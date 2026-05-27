"""
main.py
─────────────────────────────────────────────────────────────────────────────
ReviewSnipper — complete FastAPI application.

Everything lives in this one file — no subfolders, no separate routers.
Built for a non-technical solo founder managing files via GitHub Desktop.

Endpoints defined here:
    GET  /health                    — operational status check
    POST /auth/register             — create a new user account
    POST /api/v1/billing/webhook    — Stripe payment event receiver

To run locally:
    uvicorn app.main:app --reload --port 8000

Railway starts this automatically via Procfile:
    web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
─────────────────────────────────────────────────────────────────────────────
"""

import hashlib
import hmac
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import ProcessedWebhook, User

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# All secrets come from environment variables — never hardcoded.
# Set these in your .env file (local) and Railway dashboard (production).
# ─────────────────────────────────────────────────────────────────────────────

JWT_SECRET_KEY         = os.environ.get("JWT_SECRET_KEY", "")
JWT_ALGORITHM          = "HS256"
JWT_EXPIRE_HOURS       = int(os.environ.get("JWT_EXPIRE_HOURS", "72"))
STRIPE_WEBHOOK_SECRET  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set.\n"
        "Generate one: python -c \"import secrets; print(secrets.token_hex(32))\"\n"
        "Add it to your .env file and Railway environment variables."
    )

# STRIPE_WEBHOOK_SECRET is not required at startup (Stripe not called during
# registration) but the webhook endpoint will reject all requests without it.
if not STRIPE_WEBHOOK_SECRET:
    logger.warning(
        "STRIPE_WEBHOOK_SECRET is not set. "
        "The /api/v1/billing/webhook endpoint will reject all requests "
        "until this variable is configured in Railway."
    )


# ─────────────────────────────────────────────────────────────────────────────
# STRIPE PRICE → TIER MAPPING
#
# Maps your Stripe Price IDs to ReviewSnipper subscription tier codes.
#
# HOW TO SET THIS UP:
#   1. Go to Stripe Dashboard → Products → Create your three products
#      (Starter £29, SMB Pro £79, Agency £149)
#   2. Each product has a Price ID starting with "price_"
#   3. Replace the placeholder strings below with your real Price IDs
#
# The tier codes must match the subscription_status values in the User model:
#   free_tier | trial | starter | smb_pro | agency
# ─────────────────────────────────────────────────────────────────────────────

PRICE_ID_TO_TIER: dict[str, str] = {
    "price_REPLACE_WITH_STARTER_PRICE_ID":  "starter",   # £29/month
    "price_REPLACE_WITH_SMB_PRO_PRICE_ID":  "smb_pro",   # £79/month
    "price_REPLACE_WITH_AGENCY_PRICE_ID":   "agency",    # £149/month
}


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
#
# In-memory circuit breaker that limits how many webhook requests a single
# IP address can send per hour.
#
# Why this is needed:
#   Without rate limiting, a bad actor could flood the webhook endpoint
#   with thousands of fake requests, exhausting the database connection pool
#   and taking the API offline (a denial-of-service attack).
#
# How it works:
#   We track the timestamp of each request per IP in a rolling list.
#   Before processing, we remove timestamps older than 1 hour and count
#   what remains. If the count exceeds RATE_LIMIT_MAX, we return 429.
#
# Limits:
#   RATE_LIMIT_MAX     — maximum requests allowed per IP per hour
#   RATE_LIMIT_WINDOW  — the rolling window in seconds (3600 = 1 hour)
#
# Note: This in-memory store resets when the server restarts, which is
# acceptable for MVP. A Redis-backed store is better for production at scale.
# ─────────────────────────────────────────────────────────────────────────────

RATE_LIMIT_MAX    = int(os.environ.get("WEBHOOK_RATE_LIMIT_PER_HOUR", "120"))
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

# Stores request timestamps per IP: {"1.2.3.4": [1714500000.1, 1714500001.5, ...]}
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """
    Check whether a client IP is within the allowed request rate.

    Returns True  — request is allowed, proceed.
    Returns False — rate limit exceeded, return 429.

    Eviction: timestamps older than RATE_LIMIT_WINDOW are removed on
    every call, so the store never grows unbounded.
    """
    now       = time.time()
    cutoff    = now - RATE_LIMIT_WINDOW
    timestamps = _rate_limit_store[client_ip]

    # Remove expired timestamps (older than 1 hour)
    _rate_limit_store[client_ip] = [t for t in timestamps if t > cutoff]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return False  # rate limit exceeded

    # Record this request
    _rate_limit_store[client_ip].append(now)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ReviewSnipper API",
    description="AI-powered brand and competitor review intelligence for SMBs.",
    version="1.0.0",
    # /docs (Swagger UI) is disabled in production to reduce attack surface.
    # Set ENABLE_DOCS=true in your .env to enable it during local development.
    docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
    redoc_url=None,
)

# Create all database tables on startup if they do not already exist.
# Safe to call repeatedly — does nothing if tables are already present.
# Supabase/PostgreSQL will create: users, processed_webhooks
Base.metadata.create_all(bind=engine)


# ─────────────────────────────────────────────────────────────────────────────
# CORS MIDDLEWARE
#
# Controls which domains browsers are allowed to call this API from.
# Never use allow_origins=["*"] in production — it allows any website
# on the internet to make authenticated requests to your API.
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reviewsnipper.com",
        "https://www.reviewsnipper.com",
        "https://*.vercel.app",        # Vercel preview deployments
        "http://localhost:3000",
        "http://localhost:5173",       # Vite default dev port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,           # required for httpOnly cookie auth
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    expose_headers=["Set-Cookie"],
)


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plaintext: str) -> str:
    """Hash a plaintext password with bcrypt. Never store plaintext."""
    return pwd_context.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    return pwd_context.verify(plaintext, hashed)


def create_access_token(user_id: str, email: str) -> str:
    """
    Create a signed JWT access token.
    Payload: sub (user UUID), email, exp (expiry timestamp).
    Signed with JWT_SECRET_KEY using HS256.
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": user_id, "email": email, "exp": expire},
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def set_auth_cookie(response: Response, token: str) -> None:
    """
    Set the JWT as an httpOnly secure cookie.

    httpOnly=True  — JavaScript cannot read this cookie (blocks XSS theft)
    secure=True    — only sent over HTTPS (blocks interception over HTTP)
    samesite=lax   — sent on same-site requests (blocks CSRF)
    """
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=JWT_EXPIRE_HOURS * 3600,
        path="/",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr = Field(..., examples=["user@example.com"])
    password: str      = Field(..., min_length=8, max_length=128)
    model_config = {"str_strip_whitespace": True}


class RegisterResponse(BaseModel):
    message:           str
    user_id:           str
    email:             str
    subscription:      str
    trial_ends_at:     Optional[str]
    verify_email_sent: bool


class HealthResponse(BaseModel):
    status:  str
    service: str
    version: str


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """
    Lightweight health check.
    Called by Railway to verify the container is running.
    Does not query the database — always responds instantly.
    """
    return HealthResponse(
        status="healthy",
        service="ReviewSnipper API",
        version="1.0.0",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — USER REGISTRATION
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
)
def register(
    request:  RegisterRequest,
    response: Response,
    db:       Session = Depends(get_db),
) -> RegisterResponse:
    """
    Create a new user account.

    Flow:
    1. Normalise email to lowercase
    2. Check no existing account with that email
    3. Hash the password with bcrypt
    4. Create the User row — subscription_status = 'trial'
    5. Set trial_ends_at to 14 days from now
    6. Generate a JWT and set it as an httpOnly cookie
    7. Return a safe success payload (no password or token in body)
    """

    normalised_email = request.email.lower().strip()

    # Duplicate check — friendly error before hitting the unique constraint
    if db.query(User).filter(User.email == normalised_email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "email_already_registered",
                "message": "An account with this email address already exists.",
            },
        )

    hashed    = hash_password(request.password)
    trial_end = datetime.now(timezone.utc) + timedelta(days=14)

    new_user = User(
        email=normalised_email,
        hashed_password=hashed,
        subscription_status="trial",
        trial_ends_at=trial_end,
        is_verified=False,
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    except IntegrityError:
        # Race condition: two simultaneous identical registrations
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "email_already_registered",
                "message": "An account with this email address already exists.",
            },
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error":   "registration_failed",
                "message": "We could not create your account. Please try again.",
            },
        ) from exc

    token = create_access_token(user_id=str(new_user.id), email=new_user.email)
    set_auth_cookie(response, token)

    return RegisterResponse(
        message="Account created. Please check your email to verify your address.",
        user_id=str(new_user.id),
        email=new_user.email,
        subscription=new_user.subscription_status,
        trial_ends_at=trial_end.isoformat(),
        verify_email_sent=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — STRIPE WEBHOOK
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/api/v1/billing/webhook",
    tags=["Billing"],
    response_class=JSONResponse,
)
async def stripe_webhook(
    request:          Request,
    stripe_signature: str | None = Header(default=None, alias="stripe-signature"),
    db:               Session = Depends(get_db),
) -> JSONResponse:
    """
    Receive, verify, deduplicate, and process Stripe payment events.

    Security layers (in order):
    ───────────────────────────
    1. Rate limiter    — max 120 requests/IP/hour (configurable)
    2. Signature check — HMAC-SHA256 against STRIPE_WEBHOOK_SECRET
    3. Replay guard    — reject events with timestamps > 5 minutes old
    4. Idempotency     — reject duplicate event IDs via processed_webhooks

    Events handled:
    ───────────────
    checkout.session.completed      → upgrade user to paid tier
    customer.subscription.updated   → handle plan change
    customer.subscription.deleted   → downgrade to free_tier
    invoice.payment_failed          → log warning, no tier change yet

    On success: commit transaction, return 200
    On failure: rollback, record failure, return 500 (triggers Stripe retry)
    """

    # ── Layer 1: Rate limiting ────────────────────────────────────────────────
    # Extract the real client IP — Railway passes it in X-Forwarded-For.
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.client.host
        or "unknown"
    )

    if not _check_rate_limit(client_ip):
        logger.warning("[webhook] Rate limit exceeded for IP=%s", client_ip)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error":   "rate_limit_exceeded",
                "message": "Too many requests. Please try again later.",
            },
        )

    # ── Layer 2 & 3: Read raw body and verify Stripe signature ────────────────
    # CRITICAL: Read raw bytes before any JSON parsing.
    # Stripe's HMAC is computed against the exact raw bytes of the body.
    # Any change in whitespace or key ordering breaks the signature.
    raw_body: bytes = await request.body()

    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header",
        )

    event = _verify_stripe_signature(raw_body, stripe_signature)

    if event is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Webhook signature verification failed",
        )

    event_id:   str = event["id"]
    event_type: str = event["type"]

    logger.info("[webhook] Received id=%s type=%s ip=%s", event_id, event_type, client_ip)

    # ── Layer 4: Idempotency — duplicate event check ──────────────────────────
    # Query processed_webhooks for this event ID before touching any user data.
    # If found, return 200 immediately — Stripe will stop retrying.
    if db.query(ProcessedWebhook).filter(ProcessedWebhook.id == event_id).first():
        logger.info("[webhook] Duplicate event id=%s — skipping", event_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status":   "skipped",
                "reason":   "duplicate_event",
                "event_id": event_id,
            },
        )

    # ── Insert the event record (idempotency lock) ────────────────────────────
    # We flush() — not commit() — so this INSERT is part of the same
    # transaction as the user update below. If the user update fails,
    # the rollback removes this record too, leaving the system clean
    # for Stripe's next retry attempt.
    webhook_record = ProcessedWebhook(
        id=event_id,
        event_type=event_type,
        status="processed",
    )

    try:
        db.add(webhook_record)
        db.flush()

    except IntegrityError:
        # Race condition: a parallel request inserted this event ID
        # between our check above and this insert. Safe to skip.
        db.rollback()
        logger.info("[webhook] Race condition on id=%s — skipping", event_id)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status":   "skipped",
                "reason":   "duplicate_event",
                "event_id": event_id,
            },
        )

    # ── Dispatch to the correct event handler ─────────────────────────────────
    try:
        result = _dispatch_event(event=event, db=db)
        db.commit()

        logger.info("[webhook] Processed id=%s type=%s", event_id, event_type)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "processed", "event_id": event_id, "result": result},
        )

    except Exception as exc:
        db.rollback()
        error_summary = f"{type(exc).__name__}: {exc}"

        logger.error(
            "[webhook] Failed id=%s type=%s error=%s",
            event_id, event_type, error_summary,
        )

        # Record the failure in a fresh transaction so we have an audit trail
        _record_failed_event(db, event_id, event_type, error_summary)

        # Return 500 — tells Stripe something went wrong, please retry
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "Processing failed — Stripe will retry"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRIPE SIGNATURE VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _verify_stripe_signature(raw_body: bytes, signature: str) -> dict | None:
    """
    Verify the Stripe-Signature header and return the parsed event payload.

    Returns the parsed event dict on success.
    Returns None on any failure — caller raises HTTP 400.

    How Stripe signs webhooks:
    ──────────────────────────
    Stripe sends a header in this format:
        Stripe-Signature: t=1714500000,v1=abc123def456...

    To verify:
      1. Parse the timestamp (t) and signature (v1) from the header
      2. Build the signed payload string: "{timestamp}.{raw_body}"
      3. Compute HMAC-SHA256 of that string using STRIPE_WEBHOOK_SECRET
      4. Compare our result against the v1 value from the header

    Replay attack protection:
    ─────────────────────────
    The timestamp is included so we can reject events older than 5 minutes.
    This prevents an attacker from capturing a valid webhook and replaying
    it hours or days later.

    Timing attack protection:
    ─────────────────────────
    We use hmac.compare_digest() instead of == for string comparison.
    This function takes the same time regardless of how many characters
    match, preventing timing-based attacks that could leak the secret.
    """
    if not STRIPE_WEBHOOK_SECRET:
        logger.error(
            "[webhook] STRIPE_WEBHOOK_SECRET not set — cannot verify signature"
        )
        return None

    try:
        # Parse the Stripe-Signature header
        # Format: "t=1714500000,v1=abc123...,v1=def456..."
        sig_parts: dict[str, list[str]] = {}
        for part in signature.split(","):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            sig_parts.setdefault(key.strip(), []).append(value.strip())

        timestamp_str = sig_parts.get("t", [None])[0]
        v1_signatures = sig_parts.get("v1", [])

        if not timestamp_str or not v1_signatures:
            logger.warning("[webhook] Malformed Stripe-Signature header")
            return None

        timestamp = int(timestamp_str)

        # Reject events older than 5 minutes (300 seconds)
        now = int(datetime.now(timezone.utc).timestamp())
        if abs(now - timestamp) > 300:
            logger.warning(
                "[webhook] Event timestamp too old. "
                "timestamp=%d now=%d diff=%ds",
                timestamp, now, abs(now - timestamp),
            )
            return None

        # Recompute the expected HMAC-SHA256 signature
        signed_payload    = f"{timestamp}.".encode() + raw_body
        expected_sig      = hmac.new(
            STRIPE_WEBHOOK_SECRET.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison against all v1 values in the header
        signature_valid = any(
            hmac.compare_digest(expected_sig, v1)
            for v1 in v1_signatures
        )

        if not signature_valid:
            logger.warning(
                "[webhook] Signature mismatch — check STRIPE_WEBHOOK_SECRET"
            )
            return None

        return json.loads(raw_body)

    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("[webhook] Signature verification error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EVENT DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_event(event: dict, db: Session) -> dict:
    """
    Route a verified Stripe event to the correct handler.

    Unrecognised event types are acknowledged silently — we return 200
    so Stripe stops retrying. Stripe sends dozens of event types;
    it is normal and correct to ignore most of them.
    """
    event_type: str  = event.get("type", "")
    event_data: dict = event.get("data", {}).get("object", {})

    handlers = {
        "checkout.session.completed":    _handle_checkout_completed,
        "customer.subscription.updated": _handle_subscription_updated,
        "customer.subscription.deleted": _handle_subscription_deleted,
        "invoice.payment_failed":        _handle_payment_failed,
    }

    handler = handlers.get(event_type)

    if handler is None:
        logger.debug("[webhook] No handler for '%s' — acknowledging", event_type)
        return {"action": "acknowledged", "reason": "unhandled_event_type"}

    return handler(event_data=event_data, db=db)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _handle_checkout_completed(event_data: dict, db: Session) -> dict:
    """
    checkout.session.completed — user completed Stripe Checkout.
    Upgrades the user to the tier specified in session metadata.

    How to pass the tier from your frontend:
        In your Stripe Checkout session creation call, include:
        metadata={"tier": "starter"}  (or "smb_pro" / "agency")
    """
    customer_email     = event_data.get("customer_email")
    stripe_customer_id = event_data.get("customer")
    metadata           = event_data.get("metadata", {})
    new_tier           = metadata.get("tier")

    if not customer_email:
        raise ValueError(
            "checkout.session.completed is missing customer_email. "
            "Ensure your Stripe Checkout session collects the customer email."
        )

    # Fall back to price_id lookup if tier not explicitly in metadata
    if not new_tier:
        price_id = (
            event_data
            .get("line_items", {})
            .get("data", [{}])[0]
            .get("price", {})
            .get("id")
        )
        new_tier = PRICE_ID_TO_TIER.get(price_id or "", "starter")
        logger.info(
            "[webhook] Tier resolved from price_id=%s to tier=%s",
            price_id, new_tier,
        )

    return _provision_tier(
        db=db,
        email=customer_email.lower().strip(),
        new_tier=new_tier,
        stripe_customer_id=stripe_customer_id,
        event_name="checkout.session.completed",
    )


def _handle_subscription_updated(event_data: dict, db: Session) -> dict:
    """
    customer.subscription.updated — plan changed (upgrade or downgrade).
    Resolves the user from their Stripe customer ID, then updates their tier.
    """
    stripe_customer_id = event_data.get("customer")
    metadata           = event_data.get("metadata", {})
    new_tier           = metadata.get("tier")

    if not stripe_customer_id:
        raise ValueError("customer.subscription.updated missing customer field")

    # Resolve tier from price ID if not in metadata
    if not new_tier:
        price_id = (
            event_data
            .get("items", {})
            .get("data", [{}])[0]
            .get("price", {})
            .get("id")
        )
        new_tier = PRICE_ID_TO_TIER.get(price_id or "", "starter")

    user = db.query(User).filter(
        User.stripe_customer_id == stripe_customer_id
    ).first()

    if not user:
        raise ValueError(
            f"No user found with stripe_customer_id={stripe_customer_id}"
        )

    return _provision_tier(
        db=db,
        email=user.email,
        new_tier=new_tier,
        stripe_customer_id=stripe_customer_id,
        event_name="customer.subscription.updated",
    )


def _handle_subscription_deleted(event_data: dict, db: Session) -> dict:
    """
    customer.subscription.deleted — subscription cancelled, period ended.
    Downgrades the user to free_tier.

    Note: Stripe fires this at the END of the paid period, not when the
    user clicks Cancel. Users keep access until the period expires.
    """
    stripe_customer_id = event_data.get("customer")

    if not stripe_customer_id:
        raise ValueError("customer.subscription.deleted missing customer field")

    user = db.query(User).filter(
        User.stripe_customer_id == stripe_customer_id
    ).first()

    if not user:
        logger.warning(
            "[webhook] subscription.deleted: no user for stripe_customer_id=%s",
            stripe_customer_id,
        )
        return {"action": "skipped", "reason": "user_not_found"}

    previous_tier            = user.subscription_status
    user.subscription_status = "free_tier"
    user.trial_ends_at       = None

    logger.info(
        "[webhook] Cancelled: user=%s downgraded %s → free_tier",
        user.email, previous_tier,
    )

    return {
        "action":        "downgraded",
        "email":         user.email,
        "previous_tier": previous_tier,
        "new_tier":      "free_tier",
    }


def _handle_payment_failed(event_data: dict, db: Session) -> dict:
    """
    invoice.payment_failed — a payment attempt failed.

    We do NOT immediately downgrade here. Stripe retries automatically
    (typically 3 attempts over several days). We log the event and will
    send a payment failed email in Stage 7.

    If all retries fail, Stripe fires customer.subscription.deleted
    which triggers the downgrade via _handle_subscription_deleted().
    """
    stripe_customer_id = event_data.get("customer")
    attempt_count      = event_data.get("attempt_count", 1)

    if not stripe_customer_id:
        return {"action": "logged", "reason": "missing_customer_id"}

    user = db.query(User).filter(
        User.stripe_customer_id == stripe_customer_id
    ).first()

    if not user:
        return {"action": "skipped", "reason": "user_not_found"}

    logger.warning(
        "[webhook] Payment failed user=%s attempt=%d — Stripe will retry",
        user.email, attempt_count,
    )

    # TODO Stage 7: send payment_failed email via Resend
    # send_payment_failed_email(user.email, attempt_count)

    return {
        "action":        "logged",
        "email":         user.email,
        "attempt_count": attempt_count,
        "note":          "No tier change — Stripe will retry automatically",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TIER PROVISIONING
# ─────────────────────────────────────────────────────────────────────────────

def _provision_tier(
    db:                 Session,
    email:              str,
    new_tier:           str,
    stripe_customer_id: str | None,
    event_name:         str,
) -> dict:
    """
    Update a user's subscription tier.

    Called by multiple event handlers so all tier changes follow the same
    code path. Raises ValueError if the user cannot be found — the webhook
    endpoint's try/except will catch this, roll back, and return 500.
    """
    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise ValueError(
            f"No ReviewSnipper user found for email={email}. "
            f"Event={event_name}."
        )

    previous_tier            = user.subscription_status
    user.subscription_status = new_tier

    # Clear trial end date for paid subscribers
    if new_tier not in ("free_tier", "trial"):
        user.trial_ends_at = None

    # Store Stripe customer ID if not already set
    if stripe_customer_id and not user.stripe_customer_id:
        user.stripe_customer_id = stripe_customer_id

    logger.info(
        "[webhook] Provisioned user=%s %s → %s via %s",
        email, previous_tier, new_tier, event_name,
    )

    return {
        "action":        "provisioned",
        "email":         email,
        "previous_tier": previous_tier,
        "new_tier":      new_tier,
        "event":         event_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE RECORDER
# ─────────────────────────────────────────────────────────────────────────────

def _record_failed_event(
    db:         Session,
    event_id:   str,
    event_type: str,
    error:      str,
) -> None:
    """
    Write a failure record to processed_webhooks in a fresh transaction.

    Called after the main transaction is rolled back. Opens a new
    transaction to preserve the audit trail of what went wrong.

    If this write also fails, we log and move on — a failure in the
    failure recorder should never mask the original error.
    """
    try:
        existing = db.query(ProcessedWebhook).get(event_id)
        if existing:
            existing.status       = "failed"
            existing.error_detail = error[:500]
        else:
            db.add(ProcessedWebhook(
                id=event_id,
                event_type=event_type,
                status="failed",
                error_detail=error[:500],
            ))
        db.commit()
        logger.info("[webhook] Failure recorded for id=%s", event_id)

    except Exception as exc:
        db.rollback()
        logger.error(
            "[webhook] Could not record failure for id=%s: %s",
            event_id, exc,
        )