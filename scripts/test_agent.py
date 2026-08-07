"""End-to-end text-level test of the agent (skips audio I/O).

Run:  python -m scripts.test_agent
"""
from __future__ import annotations

import asyncio
import uuid

from app.agent.graph import respond
from app.db.database import init_db

SCENARIOS = {
    "emergency_dispatch": [
        "Hi, my AC just completely stopped working and it's over a hundred degrees. My mom is 82 and she lives with us, I'm really worried.",
        "We're at 1204 Maple Grove Lane, Austin, 78745.",
        "My name is Dana Whitfield, my number is 512-555-0184.",
        "Yes that's right, and yes the one forty nine is fine, please send someone.",
    ],
    "pricing_question": [
        "How much would you charge to replace a water heater?",
        "Do you do free estimates for that?",
    ],
    "out_of_area": [
        "Hi, I have a clogged drain, I'm out in San Marcos, can you come out?",
    ],
    "routine_booking": [
        "I'd like to schedule an AC tune-up sometime next week.",
        "I'm in Round Rock, ZIP 78727. Morning works best.",
        "Sam Ortiz, 512-555-0122, 88 Bluebonnet Trail, Round Rock 78727. The first morning slot is fine.",
    ],
    "out_of_scope": [
        "My refrigerator stopped cooling, can you fix it?",
    ],
}


async def run() -> None:
    init_db()
    for name, turns in SCENARIOS.items():
        call_id = f"test-{name}-{uuid.uuid4().hex[:6]}"
        print(f"\n{'=' * 70}\nSCENARIO: {name}\n{'=' * 70}")
        for turn in turns:
            print(f"\nCALLER: {turn}")
            reply = await respond(call_id, turn)
            print(f"AGENT:  {reply}")


if __name__ == "__main__":
    asyncio.run(run())
