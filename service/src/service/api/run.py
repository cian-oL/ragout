"""
FastAPI application factory and entrypoint.
"""

import uvicorn
from fastapi import FastAPI

from service.api.routers import health


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A configured FastAPI instance with all routers mounted.
    """
    prefix = "/api/v1"

    app = FastAPI(
        title="Ragout: The Tasty RAG App",
    )

    app.include_router(health.router, prefix=prefix)

    return app


app = create_app()


def main():
    """Run the service API with uvicorn auto-reload."""
    uvicorn.run(
        app="service.api.run:app",
        reload=True,
    )


if __name__ == "__main__":
    main()
