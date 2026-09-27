"""
BharatMentor — Test Suite: Database
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base, Student, StudentProgress


class TestDatabase:
    def test_student_created(self, seeded_db):
        students = seeded_db.query(Student).all()
        assert len(students) >= 3

    def test_progress_created(self, seeded_db):
        progress = seeded_db.query(StudentProgress).all()
        assert len(progress) >= 8

    def test_student_unique_email(self, seeded_db):
        emails = [s.email for s in seeded_db.query(Student).all()]
        assert len(emails) == len(set(emails))

    def test_progress_accuracy_calculated(self, seeded_db):
        rows = seeded_db.query(StudentProgress).all()
        for row in rows:
            if row.questions_attempted > 0:
                expected = row.correct_answers / row.questions_attempted
                assert abs(row.accuracy - expected) < 0.01

    def test_filter_by_student(self, seeded_db):
        rows = seeded_db.query(StudentProgress).filter_by(student_id=1).all()
        assert len(rows) >= 3
        assert all(r.student_id == 1 for r in rows)
