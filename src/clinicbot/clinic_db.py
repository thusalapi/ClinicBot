"""A tiny mock of the clinic's booking system (SQLite).

In a real deployment this would be your clinic-management-system's REST API.
The agent never touches this module directly. It goes through tools.py, which
is the only "door" the LLM gets. That separation matters for Phase 4: the
platform governs and audits the tools, not the database.
"""

import random
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta

from clinicbot.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "clinic.db"
SLOT_MINUTES = 30
BOOKABLE_DAYS = 14

# Weekly schedules copied from data/docs/02_doctors.md.
# weekday: 0 = Monday ... 6 = Sunday
DOCTORS = [
    ("nimal", "Dr. Nimal Perera", "General Practitioner",
     {0: ("08:00", "13:00"), 2: ("08:00", "13:00"), 4: ("08:00", "13:00"), 5: ("08:00", "12:00")}),
    ("ayesha", "Dr. Ayesha Fernando", "Paediatrician",
     {1: ("15:00", "20:00"), 3: ("15:00", "20:00"), 5: ("09:00", "13:00")}),
    ("rajiv", "Dr. Rajiv Kumar", "General Practitioner",
     {0: ("16:00", "21:00"), 1: ("16:00", "21:00"), 2: ("16:00", "21:00"),
      3: ("16:00", "21:00"), 4: ("16:00", "21:00"), 6: ("09:00", "13:00")}),
    ("shanika", "Dr. Shanika Silva", "Obstetrician and Gynaecologist",
     {0: ("09:00", "14:00"), 3: ("09:00", "14:00")}),
    ("kevin", "Dr. Kevin De Silva", "Dental Surgeon",
     {d: ("09:00", "17:00") for d in (1, 2, 3, 4, 5)}),
]
DOCTOR_BY_ID = {d[0]: d for d in DOCTORS}


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Open, commit on success, and always close (sqlite3's own `with` doesn't close)."""
    first_run = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if first_run:
            _create_and_seed(conn)
        with conn:
            yield conn
    finally:
        conn.close()


def _create_and_seed(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE appointments (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               doctor_id TEXT NOT NULL,
               day TEXT NOT NULL,          -- YYYY-MM-DD
               start TEXT NOT NULL,        -- HH:MM
               patient_name TEXT NOT NULL,
               phone TEXT NOT NULL,
               status TEXT NOT NULL DEFAULT 'booked'
           )"""
    )
    # A slot can be cancelled many times but booked only once.
    conn.execute(
        "CREATE UNIQUE INDEX one_booking_per_slot ON appointments (doctor_id, day, start)"
        " WHERE status = 'booked'"
    )
    # Pre-book ~35% of slots so availability looks realistic. Fixed seed = repeatable.
    rng = random.Random(42)
    for doctor_id in DOCTOR_BY_ID:
        for day in _upcoming_days():
            for start in _all_slots(doctor_id, day):
                if rng.random() < 0.35:
                    conn.execute(
                        "INSERT INTO appointments (doctor_id, day, start, patient_name, phone)"
                        " VALUES (?, ?, ?, 'Existing patient', '0000000000')",
                        (doctor_id, day.isoformat(), start),
                    )
    conn.commit()


def reset() -> None:
    DB_PATH.unlink(missing_ok=True)
    with _connect():
        pass


def _upcoming_days() -> list[date]:
    today = date.today()
    return [today + timedelta(days=i) for i in range(BOOKABLE_DAYS)]


def _all_slots(doctor_id: str, day: date) -> list[str]:
    hours = DOCTOR_BY_ID[doctor_id][3].get(day.weekday())
    if not hours:
        return []
    start = datetime.combine(day, time.fromisoformat(hours[0]))
    end = datetime.combine(day, time.fromisoformat(hours[1]))
    slots = []
    while start + timedelta(minutes=SLOT_MINUTES) <= end:
        slots.append(start.strftime("%H:%M"))
        start += timedelta(minutes=SLOT_MINUTES)
    return slots


def list_doctors() -> list[dict]:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [
        {
            "doctor_id": doctor_id,
            "name": name,
            "specialty": specialty,
            "hours": {names[d]: f"{s}-{e}" for d, (s, e) in sorted(hours.items())},
        }
        for doctor_id, name, specialty, hours in DOCTORS
    ]


def free_slots(doctor_id: str, day: date) -> list[str]:
    with _connect() as conn:
        taken = {
            row["start"]
            for row in conn.execute(
                "SELECT start FROM appointments"
                " WHERE doctor_id = ? AND day = ? AND status = 'booked'",
                (doctor_id, day.isoformat()),
            )
        }
    now = datetime.now()
    return [
        s
        for s in _all_slots(doctor_id, day)
        if s not in taken and datetime.combine(day, time.fromisoformat(s)) > now
    ]


def book(doctor_id: str, day: date, start: str, patient_name: str, phone: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO appointments (doctor_id, day, start, patient_name, phone)"
            " VALUES (?, ?, ?, ?, ?)",
            (doctor_id, day.isoformat(), start, patient_name, phone),
        )
        return cur.lastrowid


def find(appointment_id: int) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()


def cancel(appointment_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ?", (appointment_id,)
        )


def appointments_for_phone(phone: str) -> list[sqlite3.Row]:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM appointments WHERE phone = ? AND status = 'booked'"
            " AND day >= ? ORDER BY day, start",
            (phone, date.today().isoformat()),
        ).fetchall()
