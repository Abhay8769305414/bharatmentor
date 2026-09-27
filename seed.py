"""
BharatMentor — Database Seed Script
=====================================
Populates the database with sample students and progress data.

Usage:
    python seed.py

Idempotent — safe to run multiple times.
"""

from __future__ import annotations

import logging
from app.db import Student, StudentProgress, SessionLocal, create_tables

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


SAMPLE_STUDENTS = [
    {"name": "Arjun Sharma", "email": "arjun@example.com"},
    {"name": "Priya Patel", "email": "priya@example.com"},
    {"name": "Rahul Gupta", "email": "rahul@example.com"},
]

SAMPLE_PROGRESS = [
    # Arjun (id=1)
    {"student_id": 1, "topic": "Mechanics", "questions_attempted": 20, "correct_answers": 16},
    {"student_id": 1, "topic": "Thermodynamics", "questions_attempted": 15, "correct_answers": 10},
    {"student_id": 1, "topic": "Optics", "questions_attempted": 8, "correct_answers": 5},
    # Priya (id=2)
    {"student_id": 2, "topic": "Mechanics", "questions_attempted": 25, "correct_answers": 22},
    {"student_id": 2, "topic": "Electricity", "questions_attempted": 18, "correct_answers": 14},
    # Rahul (id=3)
    {"student_id": 3, "topic": "Mechanics", "questions_attempted": 10, "correct_answers": 6},
    {"student_id": 3, "topic": "Optics", "questions_attempted": 12, "correct_answers": 9},
    {"student_id": 3, "topic": "Nuclear Physics", "questions_attempted": 5, "correct_answers": 3},
]


def seed() -> None:
    create_tables()
    db = SessionLocal()

    try:
        # ── Students ──────────────────────────────────────────────────────────
        existing_emails = {s.email for s in db.query(Student.email).all()}
        students_added = 0
        student_id_map: dict[str, int] = {}

        for s_data in SAMPLE_STUDENTS:
            if s_data["email"] not in existing_emails:
                student = Student(**s_data)
                db.add(student)
                db.flush()
                student_id_map[s_data["email"]] = student.id
                students_added += 1
            else:
                existing = db.query(Student).filter_by(email=s_data["email"]).first()
                student_id_map[s_data["email"]] = existing.id

        db.commit()
        logger.info("Students: %d added, %d already existed", students_added, len(SAMPLE_STUDENTS) - students_added)

        # ── Progress ──────────────────────────────────────────────────────────
        progress_added = 0
        for p_data in SAMPLE_PROGRESS:
            existing = (
                db.query(StudentProgress)
                .filter_by(student_id=p_data["student_id"], topic=p_data["topic"])
                .first()
            )
            if existing is None:
                accuracy = (
                    p_data["correct_answers"] / p_data["questions_attempted"]
                    if p_data["questions_attempted"] > 0
                    else 0.0
                )
                progress = StudentProgress(
                    student_id=p_data["student_id"],
                    topic=p_data["topic"],
                    questions_attempted=p_data["questions_attempted"],
                    correct_answers=p_data["correct_answers"],
                    accuracy=accuracy,
                )
                db.add(progress)
                progress_added += 1

        db.commit()
        logger.info("Progress records: %d added", progress_added)
        logger.info("✅ Seed complete.")

    except Exception as exc:
        db.rollback()
        logger.error("Seed failed: %s", exc, exc_info=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed()
    