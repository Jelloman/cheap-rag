"""Nox configuration for cheap-rag testing."""

from __future__ import annotations

import nox

PYTHON_VERSIONS = ["3.13"]

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["tests", "typecheck", "lint"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run unit tests with coverage."""
    session.install(
        "pytest>=7.4.3",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.1",
        "pytest-mock>=3.12.0",
    )
    session.install("-e", ".", silent=False)
    session.run(
        "pytest",
        "-v",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "tests/",
        *session.posargs,
    )


@nox.session(python="3.13")
def typecheck(session: nox.Session) -> None:
    """Run BasedPyright type checking."""
    session.install("basedpyright>=1.34.0")
    session.install("-e", ".", silent=False)
    session.run("basedpyright", *session.posargs)


@nox.session(python="3.13")
def lint(session: nox.Session) -> None:
    """Run ruff linting and formatting checks."""
    session.install("ruff>=0.2.0")
    session.run("ruff", "check", "src/", *session.posargs)
    session.run("ruff", "format", "--check", "src/", *session.posargs)


@nox.session(python="3.13")
def format(session: nox.Session) -> None:
    """Format code with ruff."""
    session.install("ruff>=0.2.0")
    session.run("ruff", "format", "src/", "tests/", "noxfile.py", *session.posargs)
