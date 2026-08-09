"""
JWT-based authentication utilities for Cycle Buddy API.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
from jwt.exceptions import DecodeError, ExpiredSignatureError

from backend.config import Settings


class JWTTokenValidator:
    """Handles JWT token validation for API authentication."""

    def __init__(self, secret_key: str):
        """
        Initialize validator with secret key.

        Args:
            secret_key: Secret key for token signing/validation
        """
        self.secret_key = secret_key
        self.algorithm = "HS256"

    def create_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT token.

        Args:
            data: Payload to encode in token
            expires_delta: Optional expiration time delta

        Returns:
            Encoded JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(hours=24)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a JWT token and return payload if valid.

        Args:
            token: JWT token string to validate

        Returns:
            Token payload if valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except ExpiredSignatureError:
            return None
        except DecodeError:
            return None
        except Exception:
            return None


def get_token_validator(settings: Settings) -> JWTTokenValidator:
    """Factory function to create JWT token validator."""
    return JWTTokenValidator(settings.SECRET_KEY)
