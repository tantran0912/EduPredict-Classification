# -*- coding: utf-8 -*-
"""
EduPredict - Hệ thống Cảnh báo Sớm Học thuật & Lộ trình Học phần Thích ứng
Backend API: FastAPI + Scikit-Learn Pipeline + Gemini Academic Engine
"""

import base64
import io
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EduPredict")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(
    os.environ.get("EDUPREDICT_CLASSIFICATION_DIR", BASE_DIR / "classification_outputs")
)
PIPELINE_PATH = ARTIFACT_DIR / "best_classification_pipeline.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "classification_schema.json"
METRICS_PATH = ARTIFACT_DIR / "classification_metrics.json"
SAMPLE_CSV_PATH = BASE_DIR / "danh_sach_lop_mau_HaUI.csv"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AQ.Ab8RN6LB7TJLxqoQUlzdQwMSnFpqM0KfGnzBQt1YqUinIhIXLA",
)
# Ưu tiên các model phản hồi nhanh (2-4s) và ổn định nhất
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

CLASS_NAMES = ["Cần Hỗ Trợ (<60)", "Đạt Chuẩn (60-80)", "Xuất Sắc (>=80)"]
CLASS_LABELS_VN = {
    0: "Nguy cơ cao",
    1: "Đạt chuẩn",
    2: "Năng lực tốt",
}
CLASS_DESC_VN = {
    0: "Mức độ chuyên cần hoặc quỹ giờ tự học đang dưới ngưỡng an toàn, nguy cơ ảnh hưởng điều kiện dự thi và điểm học phần.",
    1: "Nhịp độ học tập hiện tại đáp ứng yêu cầu qua môn, có thể tối ưu thêm thời lượng tự học để nâng mức điểm khá/giỏi.",
    2: "Duy trì kỷ luật học tập tốt, nền tảng thuận lợi để đạt kết quả xuất sắc trong học phần.",
}


class ModelBundle:
    def __init__(self) -> None:
        self.pipeline = None
        self.model_name = "Logistic Regression (Class-Weighted)"
        self.classes = CLASS_NAMES
        self.schema: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def load(self) -> None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy mô hình tại {PIPELINE_PATH}. Vui lòng chạy train_classifier.py trước."
            )
        data = joblib.load(PIPELINE_PATH)
        self.pipeline = data["pipeline"]
        self.model_name = data.get("model_name", "Classification Model")
        self.classes = data.get("classes", CLASS_NAMES)

        if SCHEMA_PATH.exists():
            self.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        if METRICS_PATH.exists():
            self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return self.pipeline is not None


bundle = ModelBundle()


@asynccontextmanager
async def lifespan(_: FastAPI):
    bundle.load()
    logger.info(f"Đã nạp mô hình: {bundle.model_name}")
    yield


app = FastAPI(
    title="EduPredict - Hệ thống Cảnh báo Sớm & Tư vấn Lộ trình Học phần",
    version="4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    gender: str = Field("Male", alias="Gender")
    hours_studied_week: float = Field(5.0, alias="HoursStudied/Week")
    tutoring: str = Field("No", alias="Tutoring")
    region: str = Field("Urban", alias="Region")
    attendance: float = Field(75.0, alias="Attendance(%)")
    parent_education: str = Field("Secondary", alias="Parent Education")

    semester_phase: Optional[str] = Field("midterm")
    struggling_subject: Optional[str] = Field("")
    specific_bottleneck: Optional[str] = Field("")


class SimulateInput(BaseModel):
    gender: str = "Male"
    region: str = "Urban"
    tutoring: str = "No"
    parent_education: str = "Secondary"
    hours_studied: float
    attendance: float


class GeminiConsultInput(BaseModel):
    risk_tier: str = "Nguy cơ cao"
    risk_prob: float = 0.5
    hours_studied: float = 5.0
    attendance: float = 75.0
    semester_phase: str = "midterm"
    struggling_subject: str = ""
    specific_bottleneck: str = ""


class BatchStudentItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    student_id: Optional[Any] = Field("", alias="MaSV")
    full_name: Optional[Any] = Field("", alias="HoTen")
    gender: str = Field("Male", alias="Gender")
    hours_studied: float = Field(5.0, alias="HoursStudied/Week")
    attendance: float = Field(75.0, alias="Attendance(%)")
    tutoring: str = Field("No", alias="Tutoring")
    region: str = Field("Urban", alias="Region")
    parent_education: str = Field("Secondary", alias="Parent Education")
    struggling_subject: Optional[Any] = Field("", alias="MonHocLoLang")


class BatchPredictInput(BaseModel):
    students: List[BatchStudentItem]


class FileUploadPayload(BaseModel):
    filename: str
    content_base64: str


def predict_proba_single(pipeline, record: Dict[str, Any]) -> np.ndarray:
    df = pd.DataFrame([record])
    df["HoursStudied/Week"] = pd.to_numeric(df["HoursStudied/Week"], errors="coerce").fillna(5.0)
    df["Attendance(%)"] = pd.to_numeric(df["Attendance(%)"], errors="coerce").fillna(75.0)
    df["Effective_Study_Hours"] = df["HoursStudied/Week"] * (df["Attendance(%)"] / 100.0)
    df["Study_Intensity_Ratio"] = df["HoursStudied/Week"] / (df["Attendance(%)"] + 1.0)
    return pipeline.predict_proba(df)[0]


def calculate_what_if_thresholds(pipeline, base_record: Dict[str, Any]) -> Dict[str, Any]:
    current_hours = float(base_record.get("HoursStudied/Week", 5.0))
    current_att = float(base_record.get("Attendance(%)", 70.0))

    target_risk = 0.25
    rec = base_record.copy()
    best_hours = current_hours
    best_att = current_att
    found_safe = False

    min_safe_att = max(80.0, current_att)
    for test_att in [min_safe_att, max(min_safe_att, 88.0)]:
        rec["Attendance(%)"] = test_att
        for add_h in np.arange(0.0, 10.5, 0.5):
            test_h = min(16.0, current_hours + add_h)
            rec["HoursStudied/Week"] = test_h
            probs = predict_proba_single(pipeline, rec)
            if float(probs[0]) <= target_risk:
                best_hours = round(test_h, 1)
                best_att = round(test_att, 1)
                found_safe = True
                break
        if found_safe:
            break

    if not found_safe:
        best_hours = min(16.0, round(current_hours + 4.5, 1))
        best_att = max(82.0, current_att)

    delta_hours = max(0.0, round(best_hours - current_hours, 1))
    daily_extra_mins = int(round((delta_hours * 60) / 7))

    return {
        "target_hours": best_hours,
        "delta_hours": delta_hours,
        "daily_extra_mins": daily_extra_mins,
        "target_attendance": best_att,
        "current_hours": current_hours,
        "current_attendance": current_att,
    }


def generate_stakeholder_actions(record: Dict[str, Any], pred_code: int, phase: str) -> Dict[str, Any]:
    hours = float(record.get("HoursStudied/Week", 0.0))
    attendance = float(record.get("Attendance(%)", 0.0))

    phase_label = {
        "early": "giai đoạn đầu kỳ",
        "midterm": "giai đoạn giữa kỳ",
        "final": "giai đoạn ôn thi cuối kỳ",
    }.get(phase, "học kỳ hiện tại")

    student_actions: List[str] = []
    advisor_actions: List[str] = []

    if pred_code == 0:
        urgency = "Cần can thiệp sớm"
        if attendance < 80.0:
            student_actions.append(
                f"Tỷ lệ chuyên cần đang ở mức {attendance:.0f}% (sát ngưỡng giới hạn điều kiện dự thi 80%). Cần đi học đầy đủ các buổi còn lại."
            )
        if hours < 7.5:
            student_actions.append(
                f"Thời lượng tự học {hours:.1f} giờ/tuần chưa đủ tích lũy; nên nâng lên tối thiểu 7.5 – 9.0 giờ/tuần theo khung giờ cố định."
            )
        student_actions.append(
            "Rà soát lại đề cương học phần và hoàn thành đầy đủ các bài kiểm tra thường xuyên/giữa kỳ để đảm bảo điểm quá trình."
        )

        advisor_actions.append(
            f"Trao đổi trực tiếp với sinh viên trong {phase_label} để nắm rõ lý do vắng học hoặc khó khăn về phương pháp tiếp thu."
        )
        advisor_actions.append(
            "Nhắc nhở quy chế chuyên cần và điều kiện dự thi kết thúc học phần; yêu cầu cam kết hoàn thành bài tập quá trình."
        )
        advisor_actions.append(
            "Kết nối sinh viên với nhóm hỗ trợ học tập của lớp hoặc trợ giảng phụ trách môn học đang gặp vướng mắc."
        )
    elif pred_code == 1:
        urgency = "Duy trì ổn định"
        student_actions.append(
            f"Duy trì mức chuyên cần {attendance:.0f}% và bổ sung khoảng 1.5 – 2.0 giờ tự học/tuần để nâng kết quả từ mức đạt lên khá/giỏi."
        )
        student_actions.append(
            "Hệ thống hóa lý thuyết theo từng chương và luyện tập các dạng bài tự luận/trắc nghiệm tổng hợp."
        )

        advisor_actions.append(
            "Ghi nhận tiến độ ổn định; định hướng sinh viên đặt mục tiêu điểm B+/A ở các học phần trọng tâm."
        )
        advisor_actions.append(
            "Khuyến khích sinh viên tham gia thảo luận nhóm để củng cố chiều sâu kiến thức."
        )
    else:
        urgency = "Phát triển nâng cao"
        student_actions.append(
            "Phát huy thói quen tự học hiệu quả hiện tại; chủ động mở rộng sang các bài toán vận dụng cao hoặc đề tài thực tiễn."
        )
        advisor_actions.append(
            "Đề xuất sinh viên làm hạt nhân hỗ trợ nhóm học tập của lớp hoặc tham gia nghiên cứu khoa học sinh viên."
        )

    return {
        "urgency_level": urgency,
        "student_actions": student_actions,
        "advisor_actions": advisor_actions,
    }


def call_gemini_api(prompt: str) -> Optional[str]:
    """Gọi Gemini API ưu tiên model tốc độ cao và JSON chuẩn."""
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.35,
                "responseMimeType": "application/json",
                "maxOutputTokens": 4096,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=28)
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            logger.warning(f"Model {model_name} trả về {resp.status_code}: {resp.text[:120]}")
        except Exception as exc:
            logger.warning(f"Lỗi kết nối model {model_name}: {exc}")
            continue
    return None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if bundle.ready else "not_ready",
        "model_name": bundle.model_name,
        "version": "4.0",
    }


@app.get("/api/download-sample-csv")
def download_sample_csv():
    if SAMPLE_CSV_PATH.exists():
        return FileResponse(
            SAMPLE_CSV_PATH,
            media_type="text/csv",
            filename="danh_sach_lop_mau_HaUI.csv",
        )
    return JSONResponse(status_code=404, content={"message": "Không tìm thấy tệp mẫu."})


@app.get("/api/get-sample-data")
def get_sample_data():
    if SAMPLE_CSV_PATH.exists():
        df = pd.read_csv(SAMPLE_CSV_PATH).fillna("")
        return JSONResponse(content={"status": "success", "data": df.to_dict(orient="records")})
    return JSONResponse(status_code=404, content={"status": "error", "message": "Không tìm thấy tệp mẫu."})


def normalize_dataframe_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Chuẩn hóa linh hoạt tên cột từ CSV/Excel người dùng tải lên."""
    col_map: Dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("masv", "ma_sv", "mã sv", "mã sinh viên", "mssv", "student_id", "id"):
            col_map[col] = "MaSV"
        elif c in ("hoten", "ho_ten", "họ tên", "họ và tên", "full_name", "name", "ten"):
            col_map[col] = "HoTen"
        elif c in ("gender", "gioitinh", "giới tính", "phai"):
            col_map[col] = "Gender"
        elif "hour" in c or "giờ" in c or "gio" in c or "tự học" in c or "tuhoc" in c:
            col_map[col] = "HoursStudied/Week"
        elif "attend" in c or "chuyên cần" in c or "chuyencan" in c:
            col_map[col] = "Attendance(%)"
        elif "tutor" in c or "học thêm" in c or "phụ đạo" in c or "hocnhom" in c:
            col_map[col] = "Tutoring"
        elif "region" in c or "khu vực" in c or "khuvuc" in c:
            col_map[col] = "Region"
        elif "parent" in c or "phụ huynh" in c or "gia đình" in c:
            col_map[col] = "Parent Education"
        elif "mon" in c or "môn" in c or "subject" in c or "hocphan" in c or "học phần" in c:
            col_map[col] = "MonHocLoLang"

    df = df.rename(columns=col_map).fillna("")
    rows: List[Dict[str, Any]] = []
    for idx, r in df.iterrows():
        gender_raw = str(r.get("Gender", "Male")).strip().lower()
        gender = "Female" if gender_raw in ("female", "nữ", "nu", "f") else "Male"

        tutor_raw = str(r.get("Tutoring", "No")).strip().lower()
        tutoring = "Yes" if tutor_raw in ("yes", "có", "co", "1", "true") else "No"

        region_raw = str(r.get("Region", "Urban")).strip().lower()
        region = "Rural" if region_raw in ("rural", "nông thôn", "nong thon", "ngoại thành", "tỉnh") else "Urban"

        parent_raw = str(r.get("Parent Education", "Secondary")).strip()
        if parent_raw not in ("None", "Primary", "Secondary", "Tertiary"):
            parent_raw = "Secondary"

        try:
            hours_val = float(r.get("HoursStudied/Week", 5.0))
        except (ValueError, TypeError):
            hours_val = 5.0

        try:
            att_val = float(str(r.get("Attendance(%)", 75.0)).replace("%", "").strip())
        except (ValueError, TypeError):
            att_val = 75.0

        rows.append(
            {
                "MaSV": str(r.get("MaSV", f"SV{idx + 1:03d}")).strip(),
                "HoTen": str(r.get("HoTen", f"Sinh viên {idx + 1}")).strip(),
                "Gender": gender,
                "HoursStudied/Week": max(0.0, min(24.0, hours_val)),
                "Attendance(%)": max(0.0, min(100.0, att_val)),
                "Tutoring": tutoring,
                "Region": region,
                "Parent Education": parent_raw,
                "MonHocLoLang": str(r.get("MonHocLoLang", "")).strip() or "Chưa ghi chú",
            }
        )
    return rows


def run_batch_inference(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = [
        {
            "Gender": it["Gender"],
            "HoursStudied/Week": float(it["HoursStudied/Week"]),
            "Tutoring": it["Tutoring"],
            "Region": it["Region"],
            "Attendance(%)": float(it["Attendance(%)"]),
            "Parent Education": it["Parent Education"],
        }
        for it in items
    ]
    df_input = pd.DataFrame(records)
    df_input["Effective_Study_Hours"] = df_input["HoursStudied/Week"] * (df_input["Attendance(%)"] / 100.0)
    df_input["Study_Intensity_Ratio"] = df_input["HoursStudied/Week"] / (df_input["Attendance(%)"] + 1.0)

    preds = bundle.pipeline.predict(df_input)
    probs = bundle.pipeline.predict_proba(df_input)

    results = []
    at_risk_count = 0
    on_track_count = 0
    excel_count = 0

    for i, it in enumerate(items):
        code = int(preds[i])
        risk_p = float(probs[i][0])
        if code == 0:
            at_risk_count += 1
        elif code == 1:
            on_track_count += 1
        else:
            excel_count += 1

        att_val = float(it["Attendance(%)"])
        results.append(
            {
                "student_id": it["MaSV"],
                "full_name": it["HoTen"],
                "gender": it["Gender"],
                "region": it["Region"],
                "tutoring": it["Tutoring"],
                "parent_education": it["Parent Education"],
                "struggling_subject": it["MonHocLoLang"],
                "attendance": att_val,
                "danger_attendance": att_val < 80.0,
                "hours_studied": float(it["HoursStudied/Week"]),
                "predicted_code": code,
                "predicted_label": CLASS_LABELS_VN.get(code, "Đạt chuẩn"),
                "risk_probability": round(risk_p * 100, 1),
            }
        )

    total = len(results)
    summary = {
        "total_students": total,
        "at_risk_count": at_risk_count,
        "on_track_count": on_track_count,
        "excel_count": excel_count,
        "at_risk_rate": round((at_risk_count / total) * 100, 1) if total > 0 else 0,
    }
    return {"summary": summary, "students": results}


@app.post("/api/upload-class-file")
def upload_class_file(payload: FileUploadPayload) -> JSONResponse:
    if not bundle.ready:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Mô hình chưa sẵn sàng."})

    try:
        raw_bytes = base64.b64decode(payload.content_base64)
        fname = payload.filename.lower()
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            try:
                text = raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw_bytes.decode("latin-1")
            df = pd.read_csv(io.StringIO(text))

        if df.empty:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Tệp tải lên không có dữ liệu."})

        normalized_items = normalize_dataframe_columns(df)
        out = run_batch_inference(normalized_items)
        return JSONResponse(content={"status": "success", **out})
    except Exception as exc:
        logger.error(f"Lỗi đọc tệp danh sách lớp: {exc}")
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Không thể đọc tệp danh sách lớp: {exc}"},
        )


@app.post("/api/batch-predict")
def batch_predict(payload: BatchPredictInput) -> JSONResponse:
    if not bundle.ready:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Mô hình chưa sẵn sàng."})
    if not payload.students:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Danh sách sinh viên trống."})

    items = [
        {
            "MaSV": str(s.student_id) if s.student_id else "N/A",
            "HoTen": str(s.full_name) if s.full_name else "Sinh viên",
            "Gender": s.gender,
            "HoursStudied/Week": float(s.hours_studied),
            "Tutoring": s.tutoring,
            "Region": s.region,
            "Attendance(%)": float(s.attendance),
            "Parent Education": s.parent_education,
            "MonHocLoLang": str(s.struggling_subject) if s.struggling_subject else "Chưa ghi chú",
        }
        for s in payload.students
    ]
    out = run_batch_inference(items)
    return JSONResponse(content={"status": "success", **out})


@app.post("/predict")
def predict(payload: StudentInput) -> JSONResponse:
    if not bundle.ready:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Mô hình chưa sẵn sàng."})

    record = payload.model_dump(by_alias=True)
    clean_record = {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}

    try:
        df_input = pd.DataFrame([clean_record])
        df_input["HoursStudied/Week"] = pd.to_numeric(df_input["HoursStudied/Week"], errors="coerce")
        df_input["Attendance(%)"] = pd.to_numeric(df_input["Attendance(%)"], errors="coerce")
        df_input["Effective_Study_Hours"] = df_input["HoursStudied/Week"] * (df_input["Attendance(%)"] / 100.0)
        df_input["Study_Intensity_Ratio"] = df_input["HoursStudied/Week"] / (df_input["Attendance(%)"] + 1.0)

        pred_code = int(bundle.pipeline.predict(df_input)[0])
        pred_probs_raw = bundle.pipeline.predict_proba(df_input)[0]

        probs_dict = {
            bundle.classes[i]: round(float(pred_probs_raw[i]), 4)
            for i in range(len(bundle.classes))
        }

        recs = generate_stakeholder_actions(clean_record, pred_code, payload.semester_phase or "midterm")
        what_if_data = calculate_what_if_thresholds(bundle.pipeline, clean_record)

    except Exception as exc:
        logger.error(f"Lỗi suy luận: {exc}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

    return JSONResponse(
        content={
            "status": "success",
            "predicted_class_code": pred_code,
            "predicted_class_name": CLASS_LABELS_VN.get(pred_code, bundle.classes[pred_code]),
            "class_description": CLASS_DESC_VN.get(pred_code, ""),
            "urgency_level": recs["urgency_level"],
            "class_probabilities": probs_dict,
            "what_if_analysis": what_if_data,
            "stakeholder_recommendations": {
                "for_student": recs["student_actions"],
                "for_advisor": recs["advisor_actions"],
            },
        }
    )


@app.post("/api/simulate-hours")
def simulate_hours(payload: SimulateInput) -> JSONResponse:
    if not bundle.ready:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Mô hình chưa sẵn sàng."})

    rec = {
        "Gender": payload.gender,
        "HoursStudied/Week": payload.hours_studied,
        "Tutoring": payload.tutoring,
        "Region": payload.region,
        "Attendance(%)": payload.attendance,
        "Parent Education": payload.parent_education,
    }
    probs = predict_proba_single(bundle.pipeline, rec)
    return JSONResponse(
        content={
            "status": "success",
            "risk_probability": round(float(probs[0]), 4),
            "safe_probability": round(float(probs[1]), 4),
            "excel_probability": round(float(probs[2]), 4),
        }
    )


@app.post("/api/gemini-consult")
def gemini_consult(payload: GeminiConsultInput) -> JSONResponse:
    subject = payload.struggling_subject.strip()
    bottleneck = payload.specific_bottleneck.strip()

    if not subject and not bottleneck:
        return JSONResponse(
            content={
                "status": "success",
                "source": "empty_query",
                "data": {
                    "topic_title": "Định hướng phương pháp tự học đại học",
                    "direct_explanation": "Bạn chưa nhập tên học phần hoặc lý thuyết cụ thể. Hãy nhập tên môn học (ví dụ: Giải tích 1, Cơ sở dữ liệu, Kinh tế vi mô...) và phần kiến thức đang vướng mắc để nhận giải đáp chuyên môn và lộ trình học phần chi tiết.",
                    "key_takeaways": [
                        "Xác định rõ đề cương học phần và trọng số điểm quá trình ngay từ đầu kỳ.",
                        "Chia nhỏ thời lượng tự học thành các phiên 45 phút tập trung giải bài tập mẫu.",
                    ],
                    "course_roadmap": [],
                    "study_method_tip": "Nhập tên môn học cụ thể ở khung bên trái và nhấn Phân tích để tạo lộ trình.",
                    "advisor_note": "Cần trao đổi để xác định môn học cụ thể sinh viên đang gặp khó khăn.",
                },
            }
        )

    subject_display = subject if subject else "Học phần liên quan đến nội dung câu hỏi"
    question_display = (
        bottleneck
        if bottleneck
        else f"Tổng quan trọng tâm kiến thức cốt lõi và cách học hiệu quả môn {subject}"
    )

    phase_text = {
        "early": "Đầu học kỳ (Tuần 1 - 4: Xây dựng kiến thức nền tảng)",
        "midterm": "Giữa học kỳ (Tuần 5 - 9: Kiểm tra điều kiện & Bài tập lớn)",
        "final": "Cuối học kỳ (Tuần 10+: Hệ thống hóa & Ôn thi kết thúc học phần)",
    }.get(payload.semester_phase, "Trong học kỳ")

    prompt = f"""Bạn là giảng viên đại học giàu kinh nghiệm chuyên môn. Hãy giải đáp ĐÚNG TRỌNG TÂM học thuật và thiết kế LỘ TRÌNH HỌC PHẦN thực tế, cá nhân hóa cho sinh viên dưới đây.

THÔNG TIN ĐẦU VÀO CỦA SINH VIÊN:
- Học phần / Môn học: {subject_display}
- Vấn đề / Lý thuyết cần giải đáp: "{question_display}"
- Thời điểm hiện tại: {phase_text}
- Thời lượng tự học hiện có: {payload.hours_studied:.1f} giờ/tuần | Chuyên cần: {payload.attendance:.0f}%
- Mức độ cảnh báo học tập từ mô hình ML: {payload.risk_tier} (xác suất rủi ro {payload.risk_prob * 100:.1f}%)

YÊU CẦU BẮT BUỘC:
1. Đi thẳng vào kiến thức chuyên môn chính xác của môn "{subject_display}" và vấn đề "{question_display}". Tuyệt đối KHÔNG trả lời chung chung sáo rỗng, KHÔNG dùng văn mẫu có sẵn. Dù sinh viên hỏi bất kỳ môn tự nhiên, kỹ thuật, kinh tế, ngôn ngữ hay lý luận chính trị nào, phải giải thích chuẩn xác lý thuyết/công thức/nguyên lý của đúng môn đó.
2. Lộ trình học phần ("course_roadmap") phải được chia thành 3 hoặc 4 chặng rõ ràng, bám sát nội dung thực tế của học phần "{subject_display}", phân bổ thời lượng hợp lý với quỹ thời gian {payload.hours_studied:.1f} giờ/tuần và giai đoạn "{phase_text}".

TRẢ VỀ DUY NHẤT 1 JSON HỢP LỆ THEO ĐÚNG CẤU TRÚC SAU:
{{
  "topic_title": "Tên chuẩn xác của chủ đề lý thuyết và học phần",
  "direct_explanation": "Giải thích trực diện, rõ ràng, chuẩn lý thuyết chuyên ngành về vấn đề sinh viên hỏi (nêu rõ bản chất, định nghĩa, công thức hoặc quy tắc cốt lõi dễ hiểu nhất trong 4-6 câu).",
  "key_takeaways": [
    "Ý trọng tâm 1: Công thức / nguyên lý cốt lõi hoặc cách phân biệt khái niệm dễ nhầm",
    "Ý trọng tâm 2: Quy tắc áp dụng khi giải bài tập hoặc thực hành",
    "Ý trọng tâm 3: Sai lầm / bẫy đề thi phổ biến nhất ở phần này cần tránh"
  ],
  "course_roadmap": [
    {{
      "stage": "Chặng 1: Tên chặng cụ thể theo nội dung môn học",
      "duration": "Thời lượng phân bổ phù hợp (ví dụ: Tuần 1 - 2 giờ hoặc Buổi 1-2)",
      "focus": "Kiến thức / chương mục cụ thể của học phần cần nắm vững",
      "action": "Hoạt động thực hành / dạng bài tập cụ thể cần hoàn thành"
    }},
    {{
      "stage": "Chặng 2: ...",
      "duration": "...",
      "focus": "...",
      "action": "..."
    }},
    {{
      "stage": "Chặng 3: ...",
      "duration": "...",
      "focus": "...",
      "action": "..."
    }}
  ],
  "study_method_tip": "Lời khuyên ngắn gọn (1-2 câu) về phương pháp học đặc thù phù hợp nhất cho môn học này.",
  "advisor_note": "Ghi chú ngắn (1-2 câu) cho Giảng viên/Cố vấn học tập khi hỗ trợ sinh viên ở học phần này."
}}"""

    ai_raw = call_gemini_api(prompt)
    if ai_raw:
        try:
            cleaned = ai_raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            res_json = json.loads(cleaned.strip())
            return JSONResponse(content={"status": "success", "data": res_json, "source": "gemini_live"})
        except Exception as exc:
            logger.warning(f"Lỗi parse JSON Gemini: {exc}")
            match = re.search(r"\{[\s\S]*\}", ai_raw)
            if match:
                try:
                    res_json = json.loads(match.group(0))
                    return JSONResponse(content={"status": "success", "data": res_json, "source": "gemini_live"})
                except Exception:
                    pass

    return JSONResponse(
        status_code=502,
        content={
            "status": "error",
            "message": "Không thể kết nối tới dịch vụ AI để phân tích học phần lúc này. Vui lòng kiểm tra kết nối mạng và thử lại.",
        },
    )


@app.get("/", response_class=HTMLResponse)
def index():
    if not TEMPLATE_PATH.exists():
        return HTMLResponse(
            content="<h3>Không tìm thấy giao diện tại templates/index.html</h3>",
            status_code=500,
        )
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(
        content=html,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_classifier:app", host="127.0.0.1", port=8000, reload=False)
