"""
API Routes
───────────
Endpoints:
  POST /auth/register
  POST /auth/login
  POST /chat
  POST /chat/end
  POST /interactions
  GET  /profile
  GET  /resources
  GET  /health
"""
from __future__ import annotations

from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import create_access_token, decode_token, hash_password, verify_password
from api.schemas import (
    ChatRequest, ChatResponse, InteractionRequest, NLPResultSchema,
    ResourceSchema, SessionEndRequest, SessionSummaryResponse,
    TokenResponse, UserLoginRequest, UserProfileResponse, UserRegisterRequest,
)
from agents.orchestrator import OrchestratorAgent
from chatbot.conversation_manager import ConversationManager
from database import get_db
from database.models import (
    ConversationSession, Resource, ResourceInteraction,
    User, UserMentalHealthProfile,
)
from nlp.pipeline import analyze
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter()
_bearer = HTTPBearer()


# ── Dependency: resolve current user from JWT ────────────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    token = credentials.credentials
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user


# ── Auth ─────────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=201,
             summary="Register a new user")
async def register(body: UserRegisterRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    existing = await db.execute(select(User).where(User.username == body.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already taken.")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        age_group=body.age_group,
        preferred_language=body.preferred_language,
    )
    db.add(user)
    await db.flush()

    # Create empty mental health profile
    profile = UserMentalHealthProfile(user_id=user.id)
    db.add(profile)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse, summary="Login")
async def login(body: UserLoginRequest, db: Annotated[AsyncSession, Depends(get_db)]):
    result = await db.execute(select(User).where(User.username == body.username, User.is_active == True))  # noqa: E712
    user = result.scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = create_access_token(user.id)
    return TokenResponse(access_token=token)


# ── Chat ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message and get recommendations")
async def chat(
    body: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    # ── Load or create conversation session ──────────────────────────────────
    if body.session_id:
        manager = await ConversationManager.resume_session(body.session_id, db)
        if manager is None or manager.context.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        manager = await ConversationManager.create_session(current_user.id, db)

    # ── Load user profile for context ────────────────────────────────────────
    profile_result = await db.execute(
        select(UserMentalHealthProfile).where(UserMentalHealthProfile.user_id == current_user.id)
    )
    profile = profile_result.scalar_one_or_none()
    if profile is None:
        profile = UserMentalHealthProfile(user_id=current_user.id)
        db.add(profile)
        await db.flush()

    # ── Run Agentic pipeline ──────────────────────────────────────────────────
    # The OrchestratorAgent decides which specialist to invoke, runs a ReAct
    # loop (Think → Act → Observe → Reflect), consults memory, and returns
    # a fully composed reply with embedded recommendations.
    orchestrator = OrchestratorAgent()
    trace = await orchestrator.run(
        user_message=body.message,
        user_id=current_user.id,
        db=db,
        session_id=manager.context.session_id,
        user_topic_weights=current_user.topic_weights or {},
        user_modality_prefs=current_user.modality_prefs or {},
        recent_topics=manager.get_recent_topics(),
        stress_level=manager.context.stress_level,
    )

    # ── Sync conversation manager with what the agent analysed ───────────────
    agent_analysis_dict = trace.steps[0].tool_result.output if (
        trace.steps and trace.steps[0].tool_result and trace.steps[0].tool_result.success
    ) else {}

    # Re-hydrate a lightweight NLPAnalysis for profile updates + schema output
    analysis = analyze(body.message, use_transformers=False)  # fast VADER-only for profile sync
    if agent_analysis_dict:
        # Prefer richer agent analysis if available
        from nlp.sentiment_analyzer import SentimentResult
        from nlp.emotion_extractor import EmotionResult
        from nlp.intent_classifier import IntentResult
        from nlp.crisis_detector import CrisisResult
        analysis.sentiment = SentimentResult(
            label=agent_analysis_dict.get("sentiment", analysis.sentiment.label),
            score=agent_analysis_dict.get("sentiment_score", analysis.sentiment.score),
            confidence=0.8,
            compound=agent_analysis_dict.get("sentiment_score", analysis.sentiment.compound),
        )
        analysis.emotion = EmotionResult(
            dominant=agent_analysis_dict.get("emotion", analysis.emotion.dominant),
            scores=agent_analysis_dict.get("emotion_scores", analysis.emotion.scores),
        )
        analysis.intent = IntentResult(
            label=agent_analysis_dict.get("intent", analysis.intent.label),
            confidence=agent_analysis_dict.get("intent_confidence", analysis.intent.confidence),
            all_scores={},
        )
        analysis.crisis = CrisisResult(
            is_crisis=agent_analysis_dict.get("crisis_detected", analysis.crisis.is_crisis),
            severity=agent_analysis_dict.get("crisis_severity", analysis.crisis.severity),
            score=agent_analysis_dict.get("crisis_score", analysis.crisis.score),
        )

    await manager.add_user_turn(body.message, analysis, db)

    # ── Update profile ────────────────────────────────────────────────────────
    alpha = 0.3
    profile.avg_sentiment_score = (
        profile.avg_sentiment_score * (1 - alpha) + analysis.sentiment.score * alpha
    )
    profile.dominant_emotion = analysis.emotion.dominant
    profile.stress_level = manager.context.stress_level
    profile.risk_level = (
        "crisis" if trace.is_crisis and analysis.crisis.severity == "critical"
        else "high" if trace.is_crisis
        else "moderate" if analysis.sentiment.score < -0.5
        else "low"
    )

    # ── Persist assistant reply ───────────────────────────────────────────────
    reply = trace.final_answer
    await manager.add_assistant_turn(reply, db)

    # ── Build structured recommendation schemas from agent context ────────────
    raw_recs: List[dict] = trace.steps[-1].tool_result.output if (
        trace.steps and trace.steps[-1].tool_result
        and trace.steps[-1].tool_result.tool_name == "recommend_resources"
        and trace.steps[-1].tool_result.success
    ) else []

    # Fallback: pull recs injected into context by any agent step
    agent_context_recs = []
    for step in trace.steps:
        if step.tool_result and step.tool_result.tool_name == "recommend_resources" \
                and step.tool_result.success:
            agent_context_recs = step.tool_result.output or []
            break

    recs_to_use = agent_context_recs or raw_recs
    rec_schemas = [
        ResourceSchema(
            resource_id=r.get("id", ""),
            title=r.get("title", ""),
            description=r.get("description", r.get("title", "")),
            resource_type=r.get("type", "article"),
            url=r.get("url"),
            topics=r.get("topics", []),
            duration_minutes=r.get("duration_minutes"),
            hybrid_score=r.get("score", 0.0),
            reason=r.get("reason", ""),
        )
        for r in recs_to_use
    ]

    # ── Compose API response ──────────────────────────────────────────────────
    nlp_schema = NLPResultSchema(
        sentiment_label=analysis.sentiment.label,
        sentiment_score=round(analysis.sentiment.score, 4),
        emotion=analysis.emotion.dominant,
        intent=analysis.intent.label,
        crisis_detected=trace.is_crisis,
        crisis_severity=analysis.crisis.severity,
        crisis_score=analysis.crisis.score,
    )

    return ChatResponse(
        session_id=manager.context.session_id,
        reply=reply,
        nlp_analysis=nlp_schema,
        recommendations=rec_schemas,
        is_crisis=trace.is_crisis,
    )


# ── Session Management ────────────────────────────────────────────────────────

@router.post("/chat/end", response_model=SessionSummaryResponse, summary="End a conversation session")
async def end_session(
    body: SessionEndRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    manager = await ConversationManager.resume_session(body.session_id, db)
    if manager is None or manager.context.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Session not found.")
    await manager.end_session(db)
    result = await db.execute(
        select(ConversationSession).where(ConversationSession.id == body.session_id)
    )
    session = result.scalar_one()
    return SessionSummaryResponse(
        session_id=session.id,
        turn_count=session.turn_count,
        session_sentiment=session.session_sentiment,
        crisis_flagged=session.crisis_flagged,
        started_at=session.started_at,
        ended_at=session.ended_at,
    )


# ── Resource Interaction ──────────────────────────────────────────────────────

@router.post("/interactions", status_code=204, summary="Record a resource interaction")
async def record_interaction(
    body: InteractionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    interaction = ResourceInteraction(
        user_id=current_user.id,
        resource_id=body.resource_id,
        interaction_type=body.interaction_type,
        rating=body.rating,
        helpful=body.helpful,
    )
    db.add(interaction)

    # Update user's topic and modality preferences based on interaction
    resource_result = await db.execute(select(Resource).where(Resource.id == body.resource_id))
    resource = resource_result.scalar_one_or_none()
    if resource:
        topic_weights: dict = dict(current_user.topic_weights or {})
        modality_prefs: dict = dict(current_user.modality_prefs or {})
        signal = 1.0 if body.interaction_type in ("completed", "saved") else (
            -0.5 if body.interaction_type == "dismissed" else 0.3
        )
        if body.helpful is False:
            signal = -0.5
        for topic in (resource.topics or []):
            topic_weights[topic] = round(
                min(max(topic_weights.get(topic, 0.0) + signal * 0.1, 0.0), 1.0), 4
            )
        modality_prefs[resource.resource_type] = round(
            min(max(modality_prefs.get(resource.resource_type, 0.0) + signal * 0.05, 0.0), 1.0), 4
        )
        current_user.topic_weights = topic_weights
        current_user.modality_prefs = modality_prefs


# ── User Profile ──────────────────────────────────────────────────────────────

@router.get("/profile", response_model=UserProfileResponse, summary="Get user mental health profile")
async def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await db.execute(
        select(UserMentalHealthProfile).where(UserMentalHealthProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return UserProfileResponse(
        username=current_user.username,
        age_group=current_user.age_group,
        dominant_emotion=profile.dominant_emotion,
        stress_level=profile.stress_level,
        risk_level=profile.risk_level,
        frequent_topics=profile.frequent_topics or [],
        avg_sentiment_score=round(profile.avg_sentiment_score, 4),
    )


# ── Resources Browser ─────────────────────────────────────────────────────────

@router.get("/resources", response_model=List[ResourceSchema], summary="Browse available resources")
async def list_resources(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    topic: Optional[str] = None,
    resource_type: Optional[str] = None,
):
    stmt = select(Resource)
    resources_result = await db.execute(stmt)
    resources = resources_result.scalars().all()

    filtered = [
        r for r in resources
        if (topic is None or topic in (r.topics or []))
        and (resource_type is None or r.resource_type == resource_type)
    ]

    return [
        ResourceSchema(
            resource_id=r.id,
            title=r.title,
            description=r.description,
            resource_type=r.resource_type,
            url=r.url,
            topics=r.topics or [],
            duration_minutes=r.duration_minutes,
            hybrid_score=0.0,
            reason="",
        )
        for r in filtered
    ]


# ── Health Check ──────────────────────────────────────────────────────────────

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}
