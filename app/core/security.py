"""
app/core/security.py

Credential primitives for the gateway (Phase 7).

Two authentication paths share this module (ADR-011 keeps *policy* elsewhere;
this is only the credential mechanics):

  • API key (core) — the caller presents a shared secret. We never compare
    secrets with ``==`` (early-return leaks length/prefix via timing); every
    comparison goes through ``hmac.compare_digest`` on a fixed-width SHA-256
    digest, so the compare time is independent of where the mismatch is.

  • JWT (showcase, AUTH_MODE=jwt) — signed bearer tokens carrying a subject and
    a role claim. Thin wrappers over python-jose; symmetric HS256 by default.

Nothing here reads configuration or touches transport — callers pass the
expected secrets in. That keeps the module pure and trivially unit-testable.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jose import JWTError, jwt

# ── Roles / principal ────────────────────────────────────────────────────────


class Role(StrEnum):
    """Coarse authorisation level. Single-user instance → just user vs admin."""

    USER = "user"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller, attached to ``request.state.principal``."""

    subject: str
    role: Role

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN


# ── API-key hashing & constant-time verification ─────────────────────────────


def hash_api_key(key: str) -> str:
    """Return the hex SHA-256 digest of an API key.

    Used both to derive a fixed-width value for constant-time comparison and to
    key rate-limit counters without storing the raw secret.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_api_key(candidate: str, expected: str) -> bool:
    """Constant-time equality check for two API keys.

    Both sides are reduced to a 32-byte SHA-256 digest first, so the comparison
    runs over fixed-width input and reveals neither key length nor the position
    of the first differing byte.
    """
    candidate_digest = hashlib.sha256(candidate.encode("utf-8")).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(candidate_digest, expected_digest)


def identify_api_key(candidate: str, *, user_key: str, admin_key: str) -> Principal | None:
    """Match a presented key against the configured user/admin secrets.

    Both branches are always evaluated (no short-circuit) so a caller cannot
    infer *which* key matched from response timing. The admin key wins if it
    happens to equal the user key. Returns ``None`` when nothing matches.
    """
    is_admin = verify_api_key(candidate, admin_key)
    is_user = verify_api_key(candidate, user_key)
    if is_admin:
        return Principal(subject=hash_api_key(candidate)[:16], role=Role.ADMIN)
    if is_user:
        return Principal(subject=hash_api_key(candidate)[:16], role=Role.USER)
    return None


# ── JWT (showcase) ───────────────────────────────────────────────────────────


def create_access_token(
    subject: str,
    role: Role,
    *,
    secret: str,
    algorithm: str = "HS256",
    expires_s: int = 3600,
    now: int | None = None,
) -> str:
    """Mint a signed JWT carrying ``sub`` and a ``role`` claim."""
    issued_at = now if now is not None else int(time.time())
    claims: dict[str, Any] = {
        "sub": subject,
        "role": str(role),
        "iat": issued_at,
        "exp": issued_at + expires_s,
    }
    return jwt.encode(claims, secret, algorithm=algorithm)


def decode_access_token(
    token: str,
    *,
    secret: str,
    algorithm: str = "HS256",
) -> Principal | None:
    """Verify a JWT's signature/expiry and return its Principal, or ``None``.

    A missing/unknown role defaults to USER — a valid signature never grants
    admin implicitly.
    """
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm])
    except JWTError:
        return None

    subject = claims.get("sub")
    if not subject:
        return None
    raw_role = claims.get("role", Role.USER)
    try:
        role = Role(raw_role)
    except ValueError:
        role = Role.USER
    return Principal(subject=str(subject), role=role)
