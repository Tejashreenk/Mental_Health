"""
Seed data — mental health resources for the recommendation engine.
Run this script once to populate the database:
  python data/seed_data.py
"""
from __future__ import annotations

import asyncio
import sys
import os

# Allow running as a script from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import AsyncSessionLocal, init_db
from database.models import Resource

RESOURCES = [
    # ── Breathing / Relaxation Exercises ────────────────────────────────────
    {
        "title": "4-7-8 Breathing Technique",
        "description": "A simple breathing pattern that activates the parasympathetic nervous system and reduces anxiety within minutes.",
        "content": "Inhale through your nose for 4 counts. Hold your breath for 7 counts. Exhale completely through your mouth for 8 counts. Repeat 4 cycles.",
        "resource_type": "exercise",
        "topics": ["anxiety", "stress", "panic"],
        "emotions": ["fear", "anxiety"],
        "difficulty": "easy",
        "duration_minutes": 5,
    },
    {
        "title": "Box Breathing (Square Breathing)",
        "description": "Used by Navy SEALs and first responders, box breathing calms the nervous system under acute stress.",
        "content": "Breathe in for 4 counts. Hold for 4. Exhale for 4. Hold for 4. Repeat 4–8 rounds.",
        "resource_type": "exercise",
        "topics": ["stress", "anxiety", "focus"],
        "emotions": ["fear", "anger"],
        "difficulty": "easy",
        "duration_minutes": 5,
    },
    {
        "title": "Progressive Muscle Relaxation",
        "description": "Systematically tense and release muscle groups to release physical tension stored from stress.",
        "content": "Starting from your feet, tense each muscle group for 5 seconds then release for 30 seconds. Work your way up to your face.",
        "resource_type": "exercise",
        "topics": ["stress", "anxiety", "sleep"],
        "emotions": ["fear", "sadness"],
        "difficulty": "easy",
        "duration_minutes": 15,
    },

    # ── Cognitive Techniques ──────────────────────────────────────────────────
    {
        "title": "Cognitive Restructuring: Challenging Negative Thoughts",
        "description": "A core CBT technique to identify and reframe distorted thinking patterns that fuel anxiety and depression.",
        "content": "1. Identify the negative thought. 2. Ask: Is this thought fact or opinion? 3. Find evidence for and against. 4. Write a balanced alternative thought.",
        "resource_type": "technique",
        "topics": ["anxiety", "depression", "self_esteem"],
        "emotions": ["sadness", "fear"],
        "difficulty": "medium",
        "duration_minutes": 20,
    },
    {
        "title": "The 5-4-3-2-1 Grounding Technique",
        "description": "A sensory grounding exercise to interrupt panic attacks and dissociative episodes by anchoring you in the present moment.",
        "content": "Name 5 things you can see, 4 things you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
        "resource_type": "technique",
        "topics": ["anxiety", "trauma", "panic"],
        "emotions": ["fear", "disgust"],
        "difficulty": "easy",
        "duration_minutes": 3,
    },
    {
        "title": "Behavioural Activation for Depression",
        "description": "A research-backed CBT strategy to break the cycle of depression by scheduling small, meaningful activities.",
        "content": "1. List 10 activities you used to enjoy. 2. Schedule one per day, starting very small. 3. Rate your mood before and after. 4. Gradually increase activity level.",
        "resource_type": "technique",
        "topics": ["depression", "loneliness"],
        "emotions": ["sadness"],
        "difficulty": "medium",
        "duration_minutes": 30,
    },
    {
        "title": "Journaling: Expressive Writing",
        "description": "Writing about difficult emotions for 15–20 minutes has been shown to reduce PTSD symptoms and improve immune function.",
        "resource_type": "technique",
        "topics": ["stress", "trauma", "grief", "depression"],
        "emotions": ["sadness", "anger", "fear"],
        "difficulty": "easy",
        "duration_minutes": 20,
    },

    # ── Sleep Resources ───────────────────────────────────────────────────────
    {
        "title": "Sleep Hygiene: 10 Evidence-Based Tips",
        "description": "Practical evidence-based strategies for improving sleep quality without medication.",
        "resource_type": "article",
        "url": "https://www.sleepfoundation.org/sleep-hygiene",
        "topics": ["sleep", "stress", "depression"],
        "emotions": ["sadness", "neutral"],
        "difficulty": "easy",
        "duration_minutes": 10,
    },
    {
        "title": "Body Scan Meditation for Sleep",
        "description": "A 20-minute guided body scan to release tension and prepare the mind for deep, restful sleep.",
        "resource_type": "exercise",
        "topics": ["sleep", "stress", "anxiety"],
        "emotions": ["fear", "neutral"],
        "difficulty": "easy",
        "duration_minutes": 20,
    },

    # ── Mindfulness ───────────────────────────────────────────────────────────
    {
        "title": "Loving-Kindness Meditation (Metta)",
        "description": "Cultivate compassion for yourself and others. Clinically proven to reduce self-criticism and increase positive emotions.",
        "resource_type": "exercise",
        "topics": ["self_esteem", "depression", "loneliness", "anger"],
        "emotions": ["sadness", "anger"],
        "difficulty": "easy",
        "duration_minutes": 10,
    },
    {
        "title": "Mindfulness-Based Stress Reduction (MBSR) Overview",
        "description": "Introduction to the 8-week MBSR program developed by Jon Kabat-Zinn, with exercises and evidence.",
        "resource_type": "article",
        "url": "https://www.mindful.org/what-is-mbsr/",
        "topics": ["stress", "anxiety", "depression"],
        "emotions": ["fear", "sadness", "neutral"],
        "difficulty": "medium",
        "duration_minutes": 15,
    },

    # ── Social / Relationship ─────────────────────────────────────────────────
    {
        "title": "How to Reach Out When You're Struggling",
        "description": "A practical guide for asking for help from friends, family, or professionals — even when it feels impossible.",
        "resource_type": "article",
        "topics": ["loneliness", "depression", "anxiety"],
        "emotions": ["sadness", "fear"],
        "difficulty": "easy",
        "duration_minutes": 8,
    },

    # ── Grief & Loss ─────────────────────────────────────────────────────────
    {
        "title": "Understanding Grief: The Five Stages and Beyond",
        "description": "A compassionate guide to grief, dispelling myths and offering healthy ways to process loss.",
        "resource_type": "article",
        "topics": ["grief", "depression", "loneliness"],
        "emotions": ["sadness"],
        "difficulty": "easy",
        "duration_minutes": 12,
    },

    # ── Self-Assessment ───────────────────────────────────────────────────────
    {
        "title": "PHQ-9 Depression Screening",
        "description": "The PHQ-9 is a validated 9-question tool used to assess the severity of depression symptoms.",
        "resource_type": "self_assessment",
        "url": "https://www.phqscreeners.com/",
        "topics": ["depression"],
        "emotions": ["sadness"],
        "difficulty": "easy",
        "duration_minutes": 5,
    },
    {
        "title": "GAD-7 Anxiety Screening",
        "description": "The Generalized Anxiety Disorder 7-item scale helps measure anxiety severity.",
        "resource_type": "self_assessment",
        "url": "https://www.phqscreeners.com/",
        "topics": ["anxiety"],
        "emotions": ["fear"],
        "difficulty": "easy",
        "duration_minutes": 5,
    },

    # ── Crisis Resources ──────────────────────────────────────────────────────
    {
        "title": "988 Suicide & Crisis Lifeline",
        "description": "Free, confidential crisis counselling 24/7. Call or text 988. Available in English and Spanish.",
        "url": "https://988lifeline.org/",
        "resource_type": "hotline",
        "topics": ["crisis", "suicide", "self_harm"],
        "emotions": ["sadness", "fear", "anger"],
        "difficulty": "easy",
        "is_crisis_resource": True,
        "duration_minutes": None,
    },
    {
        "title": "Crisis Text Line",
        "description": "Text HOME to 741741 to connect with a trained crisis counselor. Free, 24/7, confidential.",
        "url": "https://www.crisistextline.org/",
        "resource_type": "hotline",
        "topics": ["crisis", "suicide", "self_harm", "anxiety"],
        "emotions": ["sadness", "fear"],
        "difficulty": "easy",
        "is_crisis_resource": True,
        "duration_minutes": None,
    },
    {
        "title": "International Association for Suicide Prevention — Crisis Centers",
        "description": "Find your country's crisis centre and hotline from the IASP global directory.",
        "url": "https://www.iasp.info/resources/Crisis_Centres/",
        "resource_type": "hotline",
        "topics": ["crisis"],
        "emotions": ["sadness", "fear"],
        "difficulty": "easy",
        "is_crisis_resource": True,
        "duration_minutes": None,
    },
]


async def seed():
    await init_db()
    async with AsyncSessionLocal() as session:
        for data in RESOURCES:
            resource = Resource(
                title=data["title"],
                description=data["description"],
                content=data.get("content"),
                url=data.get("url"),
                resource_type=data["resource_type"],
                topics=data.get("topics", []),
                emotions=data.get("emotions", []),
                difficulty=data.get("difficulty", "easy"),
                duration_minutes=data.get("duration_minutes"),
                is_crisis_resource=data.get("is_crisis_resource", False),
            )
            session.add(resource)
        await session.commit()
        print(f"Seeded {len(RESOURCES)} resources.")


if __name__ == "__main__":
    asyncio.run(seed())
