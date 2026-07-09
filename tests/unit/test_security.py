"""
tests/unit/test_security.py

Unit coverage for the credential primitives (app/core/security.py): API-key
hashing, constant-time verification, user/admin identification and the JWT
showcase round-trip. Pure functions — no I/O, no config.
"""

from __future__ import annotations

import time

import pytest

from app.core.security import (
    Principal,
    Role,
    create_access_token,
    decode_access_token,
    hash_api_key,
    identify_api_key,
    verify_api_key,
)

pytestmark = pytest.mark.unit

_SECRET = "unit-test-jwt-secret"


def test_hash_is_stable_and_hex() -> None:
    h1 = hash_api_key("abc")
    h2 = hash_api_key("abc")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex
    assert h1 != hash_api_key("abd")


def test_verify_api_key_matches_and_rejects() -> None:
    assert verify_api_key("secret-key", "secret-key") is True
    assert verify_api_key("secret-key", "other-key") is False
    # Length differences must not raise and must return False.
    assert verify_api_key("short", "a-much-longer-key") is False


def test_identify_api_key_resolves_roles() -> None:
    user = identify_api_key("u", user_key="u", admin_key="a")
    admin = identify_api_key("a", user_key="u", admin_key="a")
    unknown = identify_api_key("x", user_key="u", admin_key="a")

    assert user is not None and user.role is Role.USER
    assert admin is not None and admin.role is Role.ADMIN
    assert unknown is None


def test_identify_api_key_admin_wins_when_keys_equal() -> None:
    p = identify_api_key("same", user_key="same", admin_key="same")
    assert p is not None and p.role is Role.ADMIN


def test_jwt_round_trip_preserves_subject_and_role() -> None:
    token = create_access_token("user-123", Role.ADMIN, secret=_SECRET)
    principal = decode_access_token(token, secret=_SECRET)
    assert principal == Principal(subject="user-123", role=Role.ADMIN)


def test_jwt_rejects_bad_signature() -> None:
    token = create_access_token("u", Role.USER, secret=_SECRET)
    assert decode_access_token(token, secret="wrong-secret") is None


def test_jwt_rejects_expired_token() -> None:
    past = int(time.time()) - 10
    token = create_access_token("u", Role.USER, secret=_SECRET, expires_s=1, now=past)
    assert decode_access_token(token, secret=_SECRET) is None


def test_jwt_unknown_role_defaults_to_user() -> None:
    # Hand-craft a token whose role claim is not a valid Role.
    from jose import jwt

    token = jwt.encode({"sub": "u", "role": "superuser"}, _SECRET, algorithm="HS256")
    principal = decode_access_token(token, secret=_SECRET)
    assert principal is not None and principal.role is Role.USER
