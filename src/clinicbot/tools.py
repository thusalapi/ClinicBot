"""The agent's tools: plain Python functions plus a JSON schema for each.

How tool calling works:
  1. We send the LLM a list of tool *schemas* (name, description, parameters).
  2. Instead of answering in text, the model may reply with a *tool call*:
     {"name": "check_availability", "arguments": {"doctor_id": "nimal", ...}}
  3. OUR code runs the function and sends the result back as a "tool" message.
  4. The model reads the result and either calls another tool or answers.

The model never runs anything itself. It only *asks*, so every tool is a place
where you decide what's allowed. That's why tools:
  - validate every argument (models invent IDs, mangle dates, skip fields),
  - return errors as data ({"error": ...}) so the model can read them and retry,
  - are marked as side-effecting when they change the world (book/cancel), so
    the agent loop can ask a human to approve them first.

The descriptions matter as much as the code: they're the model's only manual.
"""

import json
import re
from datetime import date, datetime, timedelta

from clinicbot import clinic_db
from clinicbot.config import RETRIEVAL_MODE

# --- Tool implementations ---------------------------------------------------------


def search_clinic_info(query: str) -> dict:
    from clinicbot.retrieval import retrieve  # loads embedding/rerank models lazily

    results = retrieve(query, k=3, mode=RETRIEVAL_MODE)
    return {
        "passages": [
            {"source": r.source, "section": r.section, "text": r.text} for r in results
        ]
    }


def list_doctors(specialty: str = "") -> dict:
    doctors = clinic_db.list_doctors()
    if specialty:
        doctors = [d for d in doctors if specialty.lower() in d["specialty"].lower()] or doctors
    return {"doctors": doctors}


def check_availability(doctor_id: str, date: str) -> dict:
    if err := _check_doctor(doctor_id) or _check_date(date):
        return err
    slots = clinic_db.free_slots(doctor_id, _parse(date))
    if not slots:
        return {
            "date": date,
            "free_slots": [],
            "note": "No free slots that day. Doctor's weekly hours: "
            + json.dumps(_hours(doctor_id)),
        }
    return {"doctor_id": doctor_id, "date": date, "free_slots": slots}


def book_appointment(
    doctor_id: str, date: str, time: str, patient_name: str, phone: str
) -> dict:
    if err := _check_doctor(doctor_id) or _check_date(date) or _check_phone(phone):
        return err
    if not patient_name.strip():
        return {"error": "patient_name is required. Ask the patient for their name."}
    if time not in clinic_db.free_slots(doctor_id, _parse(date)):
        return {
            "error": f"{time} is not a free slot on {date}. "
            "Call check_availability and offer one of the free slots."
        }
    appointment_id = clinic_db.book(
        doctor_id, _parse(date), time, patient_name.strip(), _digits(phone)
    )
    return {
        "status": "booked",
        "appointment_id": appointment_id,
        "doctor": clinic_db.DOCTOR_BY_ID[doctor_id][1],
        "date": date,
        "time": time,
        "reminder": "Arrive 10 minutes early (20 minutes if this is the first visit).",
    }


def cancel_appointment(appointment_id: int, phone: str) -> dict:
    row = clinic_db.find(int(appointment_id))
    # Same answer for "doesn't exist" and "wrong phone", so the tool can't be
    # used to discover other people's appointments.
    if row is None or row["phone"] != _digits(phone) or row["status"] != "booked":
        return {"error": "No active appointment found with that ID and phone number."}
    clinic_db.cancel(row["id"])
    # Compute the fee here instead of handing the model the rule and hoping it
    # does the date maths right. Tools should return facts, not homework.
    starts = datetime.fromisoformat(f"{row['day']}T{row['start']}")
    late = starts - datetime.now() < timedelta(hours=24)
    return {
        "status": "cancelled",
        "appointment_id": row["id"],
        "cancellation_fee": "Rs. 500 (less than 24 hours' notice)" if late else "none",
    }


def find_my_appointments(phone: str) -> dict:
    if err := _check_phone(phone):
        return err
    rows = clinic_db.appointments_for_phone(_digits(phone))
    return {
        "appointments": [
            {
                "appointment_id": r["id"],
                "doctor": clinic_db.DOCTOR_BY_ID[r["doctor_id"]][1],
                "date": r["day"],
                "time": r["start"],
            }
            for r in rows
        ]
    }


# --- Validation helpers -----------------------------------------------------------


def _check_doctor(doctor_id: str) -> dict | None:
    if doctor_id not in clinic_db.DOCTOR_BY_ID:
        return {
            "error": f"Unknown doctor_id {doctor_id!r}. "
            f"Valid ids: {sorted(clinic_db.DOCTOR_BY_ID)}. Use list_doctors if unsure."
        }
    return None


def _check_date(value: str) -> dict | None:
    try:
        day = _parse(value)
    except ValueError:
        return {"error": f"Date {value!r} must be in YYYY-MM-DD format."}
    last = date.today() + timedelta(days=clinic_db.BOOKABLE_DAYS - 1)
    if not date.today() <= day <= last:
        return {"error": f"Bookings are only possible from today until {last.isoformat()}."}
    return None


def _check_phone(phone: str) -> dict | None:
    if len(_digits(phone)) != 10:
        return {"error": "Phone number must have 10 digits, e.g. 0771234567."}
    return None


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", str(phone))


def _parse(value: str) -> date:
    return date.fromisoformat(value)


def _hours(doctor_id: str) -> dict:
    return next(d["hours"] for d in clinic_db.list_doctors() if d["doctor_id"] == doctor_id)


# --- Schemas the LLM sees ---------------------------------------------------------


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_DOCTOR_ID = {
    "type": "string",
    "description": "Doctor id: nimal, ayesha, rajiv, shanika or kevin",
}
_DATE = {"type": "string", "description": "Date in YYYY-MM-DD format"}
_PHONE = {"type": "string", "description": "Patient's 10-digit phone number"}

TOOL_SCHEMAS = [
    _schema(
        "search_clinic_info",
        "Search the clinic's documents: opening hours, fees, policies, lab test "
        "preparation, vaccinations, dental care, emergencies. Use for any factual "
        "question about the clinic.",
        {"query": {"type": "string", "description": "What to look up"}},
        ["query"],
    ),
    _schema(
        "list_doctors",
        "List the clinic's doctors with their doctor_id, specialty and weekly hours.",
        {"specialty": {"type": "string", "description": "Optional filter, e.g. 'Dental'"}},
        [],
    ),
    _schema(
        "check_availability",
        "Get the free 30-minute appointment slots for one doctor on one date.",
        {"doctor_id": _DOCTOR_ID, "date": _DATE},
        ["doctor_id", "date"],
    ),
    _schema(
        "book_appointment",
        "Book an appointment. Only call this after checking availability and "
        "after the patient has given their name and phone number.",
        {
            "doctor_id": _DOCTOR_ID,
            "date": _DATE,
            "time": {"type": "string", "description": "Slot start time HH:MM, e.g. 09:30"},
            "patient_name": {"type": "string", "description": "Patient's full name"},
            "phone": _PHONE,
        },
        ["doctor_id", "date", "time", "patient_name", "phone"],
    ),
    _schema(
        "cancel_appointment",
        "Cancel an appointment. Requires the appointment id and the phone number "
        "it was booked with.",
        {"appointment_id": {"type": "integer"}, "phone": _PHONE},
        ["appointment_id", "phone"],
    ),
    _schema(
        "find_my_appointments",
        "List a patient's upcoming appointments by their phone number.",
        {"phone": _PHONE},
        ["phone"],
    ),
]

TOOLS = {
    "search_clinic_info": search_clinic_info,
    "list_doctors": list_doctors,
    "check_availability": check_availability,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
    "find_my_appointments": find_my_appointments,
}

# Tools that change the world. The agent loop asks a human before running these.
SIDE_EFFECT_TOOLS = {"book_appointment", "cancel_appointment"}


def run_tool(name: str, arguments: dict) -> dict:
    """Execute a tool call from the model, turning every failure into data."""
    if name not in TOOLS:
        return {"error": f"Unknown tool {name!r}. Available: {sorted(TOOLS)}"}
    try:
        return TOOLS[name](**arguments)
    except TypeError as exc:  # missing/extra arguments
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # never let one bad call crash the conversation
        return {"error": f"{name} failed: {exc}"}
