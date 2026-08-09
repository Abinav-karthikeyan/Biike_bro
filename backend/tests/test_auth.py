"""
Tests for JWT authentication module.
"""

import pytest
from datetime import timedelta, datetime, timezone

from backend.auth import JWTTokenValidator


@pytest.fixture
def secret_key():
    """Fixture providing a test secret key."""
    return "test-secret-key-for-jwt-validation"


@pytest.fixture
def validator(secret_key):
    """Fixture providing a JWT token validator instance."""
    return JWTTokenValidator(secret_key)


class TestJWTTokenValidator:
    """Test suite for JWTTokenValidator class."""

    def test_create_token_basic(self, validator):
        """Test creating a basic JWT token."""
        data = {"user_id": 123, "email": "user@example.com"}
        token = validator.create_token(data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_token_with_expiration(self, validator):
        """Test creating a token with custom expiration."""
        data = {"user_id": 456}
        expires_delta = timedelta(hours=1)
        token = validator.create_token(data, expires_delta)

        assert token is not None
        payload = validator.validate_token(token)
        assert payload is not None
        assert payload["user_id"] == 456

    def test_validate_token_valid(self, validator):
        """Test validating a valid token."""
        data = {"user_id": 789, "scope": "admin"}
        token = validator.create_token(data)
        payload = validator.validate_token(token)

        assert payload is not None
        assert payload["user_id"] == 789
        assert payload["scope"] == "admin"

    def test_validate_token_expired(self, validator):
        """Test validating an expired token."""
        data = {"user_id": 999}
        # Create token that expires immediately
        expires_delta = timedelta(seconds=-1)
        token = validator.create_token(data, expires_delta)
        payload = validator.validate_token(token)

        # Should return None for expired token
        assert payload is None

    def test_validate_token_invalid_signature(self, validator):
        """Test validating token with invalid signature."""
        data = {"user_id": 111}
        token = validator.create_token(data)

        # Create a different validator with different key
        other_validator = JWTTokenValidator("different-secret-key")
        payload = other_validator.validate_token(token)

        assert payload is None

    def test_validate_token_malformed(self, validator):
        """Test validating a malformed token."""
        token = "not.a.valid.token.format"
        payload = validator.validate_token(token)

        assert payload is None

    def test_validate_token_empty_string(self, validator):
        """Test validating an empty token string."""
        payload = validator.validate_token("")

        assert payload is None

    def test_token_payload_preservation(self, validator):
        """Test that token payload is correctly preserved."""
        data = {
            "user_id": 555,
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user", "admin"]
        }
        token = validator.create_token(data)
        payload = validator.validate_token(token)

        assert payload is not None
        assert payload["user_id"] == 555
        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["roles"] == ["user", "admin"]

    def test_token_includes_expiration(self, validator):
        """Test that generated tokens include expiration claim."""
        data = {"user_id": 222}
        token = validator.create_token(data)
        payload = validator.validate_token(token)

        assert payload is not None
        assert "exp" in payload
        assert isinstance(payload["exp"], int)
