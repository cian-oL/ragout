"""
Health check endpoint.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict:
    """Return service health status and current timestamp.

    Returns:
        A dict with ``status`` and ``time`` keys.
    """
    return {
        "status": "OK",
        "time": datetime.now(timezone.utc).isoformat(),
    }
