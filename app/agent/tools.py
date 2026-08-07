"""LangChain tools the voice agent can call during a conversation."""
from __future__ import annotations

import re

from langchain_core.tools import tool

from app.config import SERVICE_CITIES, SERVICE_ZIP_CODES
from app.db import database
from app.rag import retriever


@tool
def search_knowledge_base(query: str) -> str:
    """Look up company information: services offered, pricing policy and price
    ranges, diagnostic fees, warranties, membership plan, hours, safety
    protocols, and general FAQ. Use this whenever the caller asks a question
    about the company or its services."""
    return retriever.search(query)


@tool
def check_service_area(zip_code_or_city: str) -> str:
    """Verify whether an address is inside the service area. Pass the ZIP code
    if you have it, otherwise the city name. ALWAYS call this before booking
    an appointment or dispatching a technician."""
    raw = zip_code_or_city.strip().lower()
    zips = re.findall(r"\b\d{5}\b", raw)
    if zips:
        if any(z in SERVICE_ZIP_CODES for z in zips):
            return f"YES — ZIP {zips[0]} is inside our service area."
        return (
            f"NO — ZIP {zips[0]} is OUTSIDE our service area. Apologize and "
            "suggest the caller find a licensed contractor in their area. "
            "Do not book the job."
        )
    if any(city in raw for city in SERVICE_CITIES):
        return f"YES — {zip_code_or_city} is inside our service area."
    return (
        f"NO — '{zip_code_or_city}' does not match our service area "
        "(Austin, Round Rock, Pflugerville, Cedar Park). If unsure, ask for "
        "the ZIP code and check again."
    )


@tool
def get_available_slots(trade: str) -> str:
    """Get the next open appointment windows for a trade.
    trade must be one of: 'hvac', 'plumbing', 'electrical'."""
    trade = trade.strip().lower()
    if trade not in ("hvac", "plumbing", "electrical"):
        return "Invalid trade. Use 'hvac', 'plumbing', or 'electrical'."
    slots = database.available_slots(trade)
    if not slots:
        return "No open slots in the next 7 days. Offer an emergency dispatch if urgent, otherwise take a callback request."
    lines = [f"- slot_id={s['id']}: {s['window_label']}" for s in slots]
    return "Open appointment windows:\n" + "\n".join(lines)


@tool
def book_appointment(slot_id: str, customer_name: str, phone: str,
                     address: str, issue_description: str, trade: str) -> str:
    """Book a standard (non-emergency) appointment into a specific slot.
    Only call this AFTER you have: verified the service area, offered slots
    via get_available_slots, and collected the caller's full name, phone
    number, and complete street address."""
    trade = trade.strip().lower()
    result = database.book_slot(
        slot_id.strip(), customer_name.strip(), phone.strip(),
        address.strip(), issue_description.strip(), trade,
    )
    if result is None:
        return ("That slot is no longer available. Call get_available_slots "
                "again and offer the caller a fresh option.")
    return (
        f"BOOKED. Confirmation number {result['booking_id']}, "
        f"window: {result['window']}. Remind the caller about the $89 "
        "diagnostic fee (waived if they proceed with the repair) unless this "
        "is a free-estimate visit."
    )


@tool
def dispatch_emergency(customer_name: str, phone: str, address: str,
                       issue_description: str, trade: str, severity: str) -> str:
    """Dispatch an on-call technician immediately for a CRITICAL emergency
    (burst pipe, sewage backup, gas smell, sparking outlet, dangerous
    no-heat/no-AC situations). Only call this AFTER you have: given the
    relevant safety instructions, verified the service area, collected name,
    phone, and full address, AND disclosed the $149 after-hours emergency fee
    with the caller's agreement.
    trade: 'hvac', 'plumbing', or 'electrical'. severity: short label like
    'burst pipe' or 'gas smell'."""
    result = database.create_dispatch(
        customer_name.strip(), phone.strip(), address.strip(),
        issue_description.strip(), trade.strip().lower(), severity.strip(),
    )
    return (
        f"DISPATCHED. Ticket {result['dispatch_id']}. A technician is being "
        f"paged now; estimated arrival within 60-90 minutes "
        f"(target ~{result['eta_minutes']} min). The technician will call "
        "the customer when en route."
    )


ALL_TOOLS = [
    search_knowledge_base,
    check_service_area,
    get_available_slots,
    book_appointment,
    dispatch_emergency,
]
