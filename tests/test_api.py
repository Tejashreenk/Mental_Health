"""
Integration tests for the FastAPI chat endpoints.
Uses an in-memory SQLite database — no external services required.
"""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app
from database.connection import init_db, engine
from database.models import Base
from data.seed_data import seed


@pytest_asyncio.fixture(scope="module")
async def client():
    # Use in-memory SQLite for tests
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await seed()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    resp = await client.post("/api/v1/auth/register", json={
        "username": "testuser123",
        "password": "securepassword99",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    assert token

    # Login
    resp2 = await client.post("/api/v1/auth/login", json={
        "username": "testuser123",
        "password": "securepassword99",
    })
    assert resp2.status_code == 200
    assert resp2.json()["access_token"]


@pytest.mark.asyncio
async def test_duplicate_register_rejected(client: AsyncClient):
    await client.post("/api/v1/auth/register", json={
        "username": "dupuser",
        "password": "securepassword99",
    })
    resp = await client.post("/api/v1/auth/register", json={
        "username": "dupuser",
        "password": "securepassword99",
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_chat_flow(client: AsyncClient):
    # Register
    reg = await client.post("/api/v1/auth/register", json={
        "username": "chatuser1",
        "password": "password12345",
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # First message
    resp = await client.post("/api/v1/chat", json={
        "message": "I've been feeling really anxious and overwhelmed lately."
    }, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "reply" in data
    assert "recommendations" in data
    assert data["nlp_analysis"]["sentiment_label"] in ("positive", "neutral", "negative")

    session_id = data["session_id"]

    # Continue session
    resp2 = await client.post("/api/v1/chat", json={
        "message": "What can I do to calm down?",
        "session_id": session_id,
    }, headers=headers)
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_crisis_message_flagged(client: AsyncClient):
    reg = await client.post("/api/v1/auth/register", json={
        "username": "crisisuser1",
        "password": "password12345",
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    resp = await client.post("/api/v1/chat", json={
        "message": "I want to kill myself, I can't go on anymore."
    }, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_crisis"] is True
    assert data["nlp_analysis"]["crisis_severity"] in ("high", "critical")


@pytest.mark.asyncio
async def test_profile_endpoint(client: AsyncClient):
    reg = await client.post("/api/v1/auth/register", json={
        "username": "profileuser1",
        "password": "password12345",
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Chat first to create profile
    await client.post("/api/v1/chat", json={"message": "I feel sad today."}, headers=headers)

    resp = await client.get("/api/v1/profile", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "stress_level" in data
    assert "risk_level" in data
