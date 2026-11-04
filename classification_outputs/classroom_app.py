"""Run from the repository root with uvicorn classification_outputs.classroom_app:app."""

import csv
import io
import os
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from web_classifier import StudentInput, app, bundle


HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("EDUPREDICT_CLASSROOM_DB", HERE / "classroom.sqlite3"))
FEATURES = [field.alias for field in StudentInput.model_fields.values()]


class ClassUpload(BaseModel):
    csv_text: str = Field(min_length=1, max_length=200_000)
    slots: int = Field(ge=1, le=500)


class SupportPlan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    student_id: str = Field(min_length=1, max_length=80)
    class_name: str = Field(min_length=1, max_length=80)
    action: str = Field(min_length=1, max_length=500)
    owner: str = Field(min_length=1, max_length=120)
    review_date: date
    baseline_attendance: float = Field(ge=0, le=100)


class Review(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    attendance: float = Field(ge=0, le=100)
    note: str = Field(min_length=1, max_length=1000)


def classroom_db():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS support_plans (
            id INTEGER PRIMARY KEY,
            student_id TEXT NOT NULL,
            class_name TEXT NOT NULL,
            action TEXT NOT NULL,
            owner TEXT NOT NULL,
            review_date TEXT NOT NULL,
            baseline_attendance REAL NOT NULL,
            review_attendance REAL,
            review_note TEXT
        )
    """)
    return connection


@app.get("/classroom", include_in_schema=False)
def classroom():
    return FileResponse(HERE / "classroom.html")


@app.post("/classroom/rank")
def rank_class(payload: ClassUpload):
    if not bundle.ready:
        raise HTTPException(503, "Mô hình chưa sẵn sàng.")

    required = {"Student_ID", "Class", *FEATURES}
    reader = csv.DictReader(io.StringIO(payload.csv_text.lstrip("\ufeff")), strict=True)
    students = []
    records = []
    seen = set()
    try:
        headers = reader.fieldnames or []
        missing = sorted(required.difference(headers))
        if missing:
            raise HTTPException(422, "Thiếu cột: " + ", ".join(missing))
        if len(headers) != len(set(headers)):
            raise HTTPException(422, "Tên cột bị trùng.")

        for row in reader:
            line = reader.line_num
            if len(records) >= 500:
                raise HTTPException(422, "Mỗi lần nhập tối đa 500 học sinh.")
            if None in row or any(row[column] is None for column in required):
                raise HTTPException(422, f"Dòng {line}: số ô không khớp với tiêu đề.")
            student_id, class_name = row["Student_ID"].strip(), row["Class"].strip()
            if not student_id or not class_name or max(len(student_id), len(class_name)) > 80:
                raise HTTPException(422, f"Dòng {line}: mã học sinh và lớp phải có 1–80 ký tự.")
            key = (class_name, student_id)
            if key in seen:
                raise HTTPException(422, f"Dòng {line}: mã học sinh bị trùng trong lớp.")
            seen.add(key)

            try:
                record = StudentInput.model_validate(
                    {feature: row[feature].strip() for feature in FEATURES}
                ).model_dump(by_alias=True)
            except ValidationError as exc:
                fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
                raise HTTPException(422, f"Dòng {line}: kiểm tra cột {fields}.") from exc

            for field, rule in bundle.schema.get("required_features", {}).items():
                value = record[field]
                valid = (
                    value in rule["allowed_values"]
                    if rule["type"] == "categorical"
                    else rule["min"] <= value <= rule["max"]
                )
                if not valid:
                    raise HTTPException(422, f"Dòng {line}: giá trị ngoài phạm vi của cột {field}.")
            records.append(record)
            students.append({
                "student_id": student_id,
                "class_name": class_name,
                "attendance": record["Attendance(%)"],
                "hours": record["HoursStudied/Week"],
            })
    except csv.Error as exc:
        raise HTTPException(422, f"CSV không hợp lệ tại dòng {reader.line_num}.") from exc

    if not records:
        raise HTTPException(422, "CSV chưa có học sinh.")

    frame = pd.DataFrame(records)
    frame["Effective_Study_Hours"] = frame["HoursStudied/Week"] * (frame["Attendance(%)"] / 100)
    frame["Study_Intensity_Ratio"] = frame["HoursStudied/Week"] / (frame["Attendance(%)"] + 1)
    risk_column = list(bundle.pipeline.classes_).index(0)
    probabilities = bundle.pipeline.predict_proba(frame)[:, risk_column]
    for student, probability in zip(students, probabilities):
        student["risk"] = float(probability)
    students.sort(key=lambda student: (-student["risk"], student["class_name"], student["student_id"]))
    return {"total": len(students), "priority": students[:payload.slots]}


@app.get("/classroom/plans")
def list_plans():
    with closing(classroom_db()) as db:
        return [
            dict(row) for row in db.execute("""
                SELECT * FROM support_plans
                ORDER BY review_attendance IS NOT NULL, review_date, id
            """)
        ]


@app.post("/classroom/plans", status_code=201)
def add_plan(plan: SupportPlan):
    data = plan.model_dump(mode="json")
    with closing(classroom_db()) as db, db:
        result = db.execute("""
            INSERT INTO support_plans
                (student_id, class_name, action, owner, review_date, baseline_attendance)
            VALUES (:student_id, :class_name, :action, :owner, :review_date, :baseline_attendance)
        """, data)
        return {"id": result.lastrowid}


@app.put("/classroom/plans/{plan_id}/review")
def save_review(plan_id: int, review: Review):
    with closing(classroom_db()) as db, db:
        result = db.execute("""
            UPDATE support_plans SET review_attendance = ?, review_note = ? WHERE id = ?
        """, (review.attendance, review.note, plan_id))
        if result.rowcount == 0:
            raise HTTPException(404, "Không tìm thấy kế hoạch.")
    return {"id": plan_id}
