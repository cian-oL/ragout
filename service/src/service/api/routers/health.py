"""
Health check endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict:
    """
    Health check endpoint.
    """

    return {
        "status": "OK",
        "time": datetime.now(timezone.utc).isoformat(),
    }
