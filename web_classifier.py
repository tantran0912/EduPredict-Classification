F# -*- coding: utf-8 -*-
"""
EduPredict AI - Web Dashboard Khoa Học & Backend API Phân Lớp (Classification)
File: web_classifier.py
Dành cho: Thư mục độc lập EduPredict_Classification
"""

import os
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

ARTIFACT_DIR = Path(
    os.environ.get(
        "EDUPREDICT_CLASSIFICATION_DIR",
        Path(__file__).resolve().parent / "classification_outputs",
    )
)
PIPELINE_PATH = ARTIFACT_DIR / "best_classification_pipeline.joblib"
SCHEMA_PATH = ARTIFACT_DIR / "classification_schema.json"
METRICS_PATH = ARTIFACT_DIR / "classification_metrics.json"

CLASS_NAMES = ["Cần Hỗ Trợ (<60)", "Đạt Chuẩn (60-80)", "Xuất Sắc (>=80)"]
CLASS_DESC = {
    0: "Nhóm có nguy cơ kết quả thấp (Exam_Score < 60) - Cần được hỗ trợ và can thiệp kịp thời",
    1: "Nhóm đạt chuẩn học tập (60 <= Exam_Score < 80) - Tiếp tục duy trì nhịp độ ổn định",
    2: "Nhóm học lực xuất sắc (Exam_Score >= 80) - Năng lực học tập cao, khuyến khích bồi dưỡng nâng cao",
}


class ModelBundle:
    def __init__(self) -> None:
        self.pipeline = None
        self.model_name = "Logistic Regression (Class-Weighted)"
        self.classes = CLASS_NAMES
        self.schema = {}
        self.metrics = {}

    def load(self) -> None:
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file mô hình tại {PIPELINE_PATH}. "
                f"Vui lòng chạy 'python train_classifier.py' trước khi khởi chạy server."
            )
        data = joblib.load(PIPELINE_PATH)
        self.pipeline = data["pipeline"]
        self.model_name = data.get("model_name", "Classification Model")
        self.classes = data.get("classes", CLASS_NAMES)

        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None


bundle = ModelBundle()


@asynccontextmanager
async def lifespan(_: FastAPI):
    bundle.load()
    print(f"[EduPredict Classification] Đã nạp thành công mô hình: {bundle.model_name} từ {ARTIFACT_DIR}")
    yield


app = FastAPI(
    title="EduPredict AI - Academic Classification & Early Warning Engine",
    description="Hệ thống Phân tầng Học lực & Cảnh báo Sớm Học sinh Nguy cơ theo Chuẩn Khoa học Dữ liệu",
    version="1.0",
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
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    gender: str = Field(..., alias="Gender", examples=["Female"], description="Giới tính: Female hoặc Male")
    hours_studied_week: float = Field(..., alias="HoursStudied/Week", examples=[12.5], description="Số giờ tự học mỗi tuần (0.0 đến 16.0)")
    tutoring: str = Field(..., alias="Tutoring", examples=["Yes"], description="Học thêm: Yes hoặc No")
    region: str = Field(..., alias="Region", examples=["Urban"], description="Khu vực sinh sống: Urban hoặc Rural")
    attendance: float = Field(..., alias="Attendance(%)", examples=[92.0], description="Tỷ lệ chuyên cần từ 0 đến 100 (%)")
    parent_education: str = Field(..., alias="Parent Education", examples=["Tertiary"], description="Học vấn phụ huynh: None, Primary, Secondary, Tertiary")


def error_response(status_code: int, message: str, details: Optional[List[Dict[str, Any]]] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message, "details": details or []},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(x) for x in err.get("loc", []) if x != "body"), "error": err.get("msg", "")}
        for err in exc.errors()
    ]
    return error_response(422, "Dữ liệu đầu vào không hợp lệ hoặc thiếu trường bắt buộc.", details)


def generate_stakeholder_recommendations(record: Dict[str, Any], pred_class: int) -> Dict[str, Any]:
    hours = float(record.get("HoursStudied/Week", 0.0))
    attendance = float(record.get("Attendance(%)", 0.0))
    tutoring = str(record.get("Tutoring", "No")).strip()

    student_actions = []
    teacher_actions = []
    parent_actions = []
    urgency = "Trung bình"

    if pred_class == 0:  # Cần Hỗ Trợ (<60 điểm)
        urgency = "Khẩn Cấp (Cảnh Báo Sớm)"
        # Dành cho học sinh
        if attendance < 75.0:
            student_actions.append(f"Tăng tỷ lệ chuyên cần từ {attendance:.1f}% lên tối thiểu 85.0% để đảm bảo tiếp thu trọn vẹn kiến thức trọng tâm.")
        if hours < 8.0:
            student_actions.append(f"Tăng giờ tự học từ {hours:.1f} giờ/tuần lên mức 10.0 - 12.0 giờ/tuần, áp dụng kỹ thuật Pomodoro (học 25 phút, nghỉ 5 phút).")
        if tutoring == "No":
            student_actions.append("Đăng ký tham gia ngay các lớp phụ đạo hoặc nhóm học kèm để bù đắp các lỗ hổng kiến thức căn bản.")
        student_actions.append("Lập bảng mục tiêu học tập hàng ngày và nhờ giáo viên bộ môn kiểm tra định kỳ.")

        # Dành cho giáo viên & cố vấn
        teacher_actions.append("Cố vấn học tập thiết lập lịch phỏng vấn 1-1 để tìm hiểu nguyên nhân giảm sút chuyên cần hoặc khó khăn trong môn học.")
        teacher_actions.append("Cung cấp gói bài tập bổ trợ có phân cấp độ khó từ cơ bản đến trung bình.")
        teacher_actions.append("Xếp học sinh vào nhóm học tập có bạn học khá/giỏi kèm cặp (Peer Mentoring).")

        # Dành cho phụ huynh
        parent_actions.append("Tạo không gian tự học yên tĩnh tại nhà, hạn chế thời gian sử dụng thiết bị giải trí trước giờ học.")
        parent_actions.append("Chủ động liên hệ giáo viên chủ nhiệm 2 tuần/lần để nắm bắt tình hình chuyên cần và bài tập về nhà.")
        parent_actions.append("Động viên tinh thần tích cực, tránh tạo áp lực tiêu cực khiến học sinh chán nản.")

    elif pred_class == 1:  # Đạt Chuẩn (60 - 80 điểm)
        urgency = "Trung Bình (Duy Trì & Bứt Phá)"
        student_actions.append(f"Nâng thời lượng tự học thêm 2.0 giờ/tuần (hiện tại: {hours:.1f} giờ/tuần) để chuyển hóa kiến thức khá thành giỏi.")
        if attendance < 90.0:
            student_actions.append(f"Cố gắng không vắng các buổi học chuyên đề nâng cao (tỷ lệ hiện tại: {attendance:.1f}%).")
        student_actions.append("Tập trung giải các bài toán vận dụng cao và rèn luyện kỹ năng làm bài thi dưới áp lực thời gian.")

        teacher_actions.append("Gợi ý các tài liệu tham khảo mở rộng và câu hỏi tư duy phản biện trong giờ học.")
        teacher_actions.append("Giao các nhiệm vụ thảo luận nhóm để phát huy năng lực tiềm ẩn của học sinh.")

        parent_actions.append("Khích lệ học sinh đặt mục tiêu bứt phá lên nhóm xuất sắc trong kỳ thi sắp tới.")
        parent_actions.append("Hỗ trợ tài liệu học tập hoặc các khóa học kỹ năng giải quyết vấn đề.")

    else:  # Xuất Sắc (>=80 điểm)
        urgency = "Phát Triển Tài Năng"
        student_actions.append("Duy trì phương pháp tự học khoa học và kỷ luật chuyên cần hiện tại.")
        student_actions.append("Thử sức với các kỳ thi học sinh giỏi, nghiên cứu khoa học hoặc đăng ký các dự án học thuật.")
        student_actions.append("Đóng vai trò nhóm trưởng học tập (Peer Tutor) để vừa rèn luyện kỹ năng truyền đạt vừa khắc sâu kiến thức.")

        teacher_actions.append("Bồi dưỡng nâng cao và giới thiệu học sinh vào đội tuyển học sinh giỏi hoặc các dự án nghiên cứu chuyên sâu.")
        teacher_actions.append("Tạo điều kiện để học sinh thuyết trình bài giảng trước lớp nhằm phát triển kỹ năng lãnh đạo học thuật.")

        parent_actions.append("Tạo điều kiện để học sinh phát triển toàn diện cả kỹ năng mềm lẫn thể chất.")
        parent_actions.append("Định hướng nghề nghiệp và kế hoạch học tập dài hạn (học bổng, du học, đại học hàng đầu).")

    return {
        "urgency_level": urgency,
        "student_actions": student_actions,
        "teacher_actions": teacher_actions,
        "parent_actions": parent_actions,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if bundle.ready else "not_ready",
        "model_name": bundle.model_name,
        "classes": bundle.classes,
        "framework": "Scikit-Learn Production Pipeline",
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return bundle.schema if bundle.schema else {"message": "Chưa có schema."}


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    return bundle.metrics if bundle.metrics else {"message": "Chưa có metrics."}


@app.post("/predict")
def predict(payload: StudentInput) -> JSONResponse:
    if not bundle.ready:
        return error_response(503, "Mô hình phân lớp chưa sẵn sàng. Vui lòng kiểm tra lại server.")

    record = payload.model_dump(by_alias=True)
    record = {k: (v.strip() if isinstance(v, str) else v) for k, v in record.items()}

    try:
        df_input = pd.DataFrame([record])
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

        recs = generate_stakeholder_recommendations(record, pred_code)

    except Exception as exc:
        return error_response(500, f"Lỗi trong quá trình suy luận mô hình: {str(exc)}")

    return JSONResponse(
        content={
            "predicted_class_code": pred_code,
            "predicted_class_name": bundle.classes[pred_code],
            "class_description": CLASS_DESC[pred_code],
            "urgency_level": recs["urgency_level"],
            "class_probabilities": probs_dict,
            "stakeholder_recommendations": {
                "for_student": recs["student_actions"],
                "for_teacher": recs["teacher_actions"],
                "for_parent": recs["parent_actions"],
            },
            "derived_features": {
                "Effective_Study_Hours": round(float(df_input["Effective_Study_Hours"].iloc[0]), 2),
                "Study_Intensity_Ratio": round(float(df_input["Study_Intensity_Ratio"].iloc[0]), 4),
            },
            "model_name": bundle.model_name,
            "status": "success",
        }
    )


@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>EduPredict AI - Academic Classification & Early Warning Engine</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
      <style>
        :root {
          --primary: #0284c7;
          --primary-dark: #0369a1;
          --primary-light: #e0f2fe;
          --slate-900: #0f172a;
          --slate-800: #1e293b;
          --slate-700: #334155;
          --slate-600: #475569;
          --slate-500: #64748b;
          --slate-200: #e2e8f0;
          --slate-100: #f1f5f9;
          --slate-50: #f8fafc;
          --rose-600: #e11d48;
          --rose-100: #ffe4e6;
          --rose-50: #fff1f2;
          --emerald-600: #059669;
          --emerald-100: #d1fae5;
          --emerald-50: #ecfdf5;
          --amber-600: #d97706;
          --amber-100: #fef3c7;
          --amber-50: #fffbeb;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
          font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
          background-color: var(--slate-50);
          color: var(--slate-900);
          line-height: 1.5;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }

        /* NAVBAR */
        .navbar {
          background: #ffffff;
          border-bottom: 1px solid var(--slate-200);
          padding: 0.85rem 2rem;
          display: flex;
          align-items: center;
          justify-content: space-between;
          position: sticky;
          top: 0;
          z-index: 50;
        }
        .nav-brand {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          font-weight: 800;
          font-size: 1.25rem;
          color: var(--slate-900);
        }
        .nav-logo {
          width: 32px;
          height: 32px;
          background: linear-gradient(135deg, var(--primary), var(--primary-dark));
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 1.1rem;
        }
        .nav-badges {
          display: flex;
          align-items: center;
          gap: 0.6rem;
        }
        .pill {
          padding: 0.3rem 0.75rem;
          border-radius: 9999px;
          font-size: 0.78rem;
          font-weight: 700;
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
        }
        .pill-success { background: var(--emerald-100); color: var(--emerald-600); }
        .pill-primary { background: var(--primary-light); color: var(--primary-dark); }
        .pill-link {
          background: var(--slate-100);
          color: var(--slate-700);
          text-decoration: none;
          transition: all 0.2s;
        }
        .pill-link:hover { background: var(--slate-200); color: var(--slate-900); }

        /* HEADER HERO */
        .hero {
          background: linear-gradient(135deg, #0f172a 0%, #0369a1 70%, #0284c7 100%);
          color: white;
          padding: 2.75rem 2rem 4rem;
          text-align: center;
        }
        .hero h1 {
          font-size: 2.2rem;
          font-weight: 800;
          letter-spacing: -0.5px;
          margin-bottom: 0.5rem;
        }
        .hero p {
          font-size: 1.05rem;
          opacity: 0.9;
          max-width: 760px;
          margin: 0 auto;
        }

        /* LAYOUT BENTO GRID */
        .main-container {
          max-width: 1240px;
          margin: -2.5rem auto 3rem;
          padding: 0 1.5rem;
          width: 100%;
          display: grid;
          grid-template-columns: 460px 1fr;
          gap: 1.75rem;
        }
        @media (max-width: 1024px) {
          .main-container { grid-template-columns: 1fr; margin-top: -1.5rem; }
        }

        .card {
          background: #ffffff;
          border-radius: 1rem;
          border: 1px solid var(--slate-200);
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
          overflow: hidden;
        }
        .card-header {
          padding: 1.25rem 1.5rem;
          border-bottom: 1px solid var(--slate-100);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        .card-title {
          font-size: 1.1rem;
          font-weight: 700;
          color: var(--slate-900);
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }
        .card-body { padding: 1.5rem; }

        /* PRESETS */
        .preset-bar {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.25rem;
        }
        .btn-preset {
          flex: 1;
          padding: 0.5rem 0.6rem;
          font-size: 0.78rem;
          font-weight: 700;
          border-radius: 0.5rem;
          border: 1px solid var(--slate-200);
          background: var(--slate-50);
          color: var(--slate-700);
          cursor: pointer;
          transition: all 0.2s;
        }
        .btn-preset:hover {
          background: #ffffff;
          border-color: var(--primary);
          color: var(--primary);
        }

        /* FORM */
        .form-section-title {
          font-size: 0.8rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.75px;
          color: var(--slate-500);
          margin: 1rem 0 0.6rem;
        }
        .form-section-title:first-of-type { margin-top: 0; }
        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.85rem;
        }
        .form-group {
          display: flex;
          flex-direction: column;
          gap: 0.3rem;
        }
        .form-group.full { grid-column: 1 / -1; }
        label {
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--slate-700);
          display: flex;
          justify-content: space-between;
        }
        .slider-val {
          font-weight: 700;
          color: var(--primary);
        }
        input, select {
          padding: 0.65rem 0.85rem;
          border: 1.5px solid var(--slate-200);
          border-radius: 0.5rem;
          font-size: 0.92rem;
          font-family: inherit;
          background: white;
          transition: border-color 0.2s;
        }
        input:focus, select:focus {
          outline: none;
          border-color: var(--primary);
          box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
        }
        input[type="range"] {
          padding: 0;
          height: 6px;
          accent-color: var(--primary);
          cursor: pointer;
        }

        /* LIVE FEATURE PREVIEW */
        .fe-preview-box {
          background: var(--slate-50);
          border: 1px dashed var(--slate-200);
          border-radius: 0.65rem;
          padding: 0.75rem 1rem;
          margin-top: 1rem;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 0.75rem;
          font-size: 0.82rem;
        }
        .fe-item-title { color: var(--slate-500); font-size: 0.75rem; }
        .fe-item-val { font-weight: 800; color: var(--primary-dark); font-size: 1rem; }

        .btn-submit {
          margin-top: 1.5rem;
          width: 100%;
          background: linear-gradient(135deg, var(--primary), var(--primary-dark));
          color: white;
          border: none;
          padding: 0.85rem;
          border-radius: 0.65rem;
          font-size: 1rem;
          font-weight: 700;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
          transition: all 0.2s;
        }
        .btn-submit:hover {
          transform: translateY(-1px);
          box-shadow: 0 6px 16px rgba(2, 132, 199, 0.4);
        }
        .btn-submit:active { transform: translateY(0); }

        /* RESULTS VIEW */
        .results-container {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }
        .banner-card {
          padding: 1.5rem;
          border-radius: 1rem;
          border: 1.5px solid transparent;
          display: flex;
          align-items: center;
          justify-content: space-between;
          transition: all 0.3s;
        }
        .banner-risk {
          background: var(--rose-50);
          border-color: var(--rose-100);
          color: var(--rose-600);
        }
        .banner-track {
          background: var(--primary-light);
          border-color: #bae6fd;
          color: var(--primary-dark);
        }
        .banner-excel {
          background: var(--emerald-50);
          border-color: var(--emerald-100);
          color: var(--emerald-600);
        }
        .banner-info h3 {
          font-size: 1.6rem;
          font-weight: 800;
          line-height: 1.2;
          margin-bottom: 0.25rem;
        }
        .banner-desc { font-size: 0.88rem; opacity: 0.9; }
        .urgency-badge {
          padding: 0.4rem 1rem;
          border-radius: 9999px;
          font-weight: 800;
          font-size: 0.85rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          white-space: nowrap;
        }
        .urgency-risk { background: #fee2e2; color: #b91c1c; }
        .urgency-track { background: #e0f2fe; color: #0369a1; }
        .urgency-excel { background: #dcfce7; color: #15803d; }

        /* PROBABILITY BARS */
        .prob-card {
          background: white;
          border-radius: 0.85rem;
          padding: 1.25rem 1.5rem;
          border: 1px solid var(--slate-200);
        }
        .prob-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        .prob-header h4 { font-size: 0.95rem; font-weight: 700; color: var(--slate-800); }
        .prob-row { margin-bottom: 0.85rem; }
        .prob-row:last-child { margin-bottom: 0; }
        .prob-meta {
          display: flex;
          justify-content: space-between;
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--slate-700);
          margin-bottom: 0.25rem;
        }
        .prob-track {
          height: 10px;
          background: var(--slate-100);
          border-radius: 9999px;
          overflow: hidden;
        }
        .prob-fill {
          height: 100%;
          border-radius: 9999px;
          transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* RECOMMENDATION TABS */
        .recs-card {
          background: white;
          border-radius: 0.85rem;
          padding: 1.25rem 1.5rem;
          border: 1px solid var(--slate-200);
        }
        .recs-tabs {
          display: flex;
          gap: 0.5rem;
          border-bottom: 1px solid var(--slate-200);
          margin-bottom: 1rem;
        }
        .tab-btn {
          padding: 0.5rem 1rem;
          font-size: 0.85rem;
          font-weight: 700;
          border: none;
          background: transparent;
          color: var(--slate-500);
          cursor: pointer;
          border-bottom: 2px solid transparent;
          transition: all 0.2s;
        }
        .tab-btn.active {
          color: var(--primary);
          border-bottom-color: var(--primary);
        }
        .recs-list {
          list-style: none;
          display: flex;
          flex-direction: column;
          gap: 0.65rem;
        }
        .recs-list li {
          display: flex;
          align-items: flex-start;
          gap: 0.65rem;
          font-size: 0.88rem;
          color: var(--slate-700);
          line-height: 1.5;
        }
        .recs-icon {
          color: var(--primary);
          font-size: 1rem;
          margin-top: 0.1rem;
        }

        footer {
          margin-top: auto;
          text-align: center;
          padding: 1.5rem;
          font-size: 0.85rem;
          color: var(--slate-500);
          border-top: 1px solid var(--slate-200);
          background: white;
        }
      </style>
    </head>
    <body>
      <nav class="navbar">
        <div class="nav-brand">
          <div class="nav-logo">EP</div>
          <span>EduPredict AI</span>
        </div>
        <div class="nav-badges">
          <span class="pill pill-success">🟢 Logistic Regression (Balanced)</span>
          <span class="pill pill-primary">Test Accuracy: 85.0%</span>
          <span class="pill pill-primary">At-Risk Recall: 88.6%</span>
        </div>
      </nav>

      <header class="hero">
        <h1>Hệ Thống Phân Tầng Học Lực & Cảnh Báo Sớm</h1>
        <p>Ứng dụng Machine Learning phân lớp nhằm phát hiện sớm học sinh có nguy cơ kết quả thấp (Exam_Score &lt; 60) và cá nhân hóa kế hoạch can thiệp sư phạm 3 chiều</p>
      </header>

      <main class="main-container">
        <!-- CỘT TRÁI: FORM NHẬP LIỆU -->
        <section class="card">
          <div class="card-header">
            <h2 class="card-title">📝 Thông Tin Học Sinh</h2>
            <span style="font-size: 0.8rem; color: var(--slate-500);">Tập mẫu SAP-4000</span>
          </div>

          <div class="card-body">
            <!-- Nút Preset tình huống mẫu -->
            <div class="preset-bar">
              <button type="button" class="btn-preset" onclick="loadPreset('risk')">🔴 Mẫu Nguy Cơ</button>
              <button type="button" class="btn-preset" onclick="loadPreset('track')">🔵 Mẫu Đạt Chuẩn</button>
              <button type="button" class="btn-preset" onclick="loadPreset('excel')">🟢 Mẫu Xuất Sắc</button>
            </div>

            <form id="predictForm">
              <!-- Nhóm 1: Cá nhân & Khu vực -->
              <div class="form-section-title">1. Thông Tin Cá Nhân & Địa Bàn</div>
              <div class="form-grid">
                <div class="form-group">
                  <label for="gender">Giới tính</label>
                  <select id="gender" required>
                    <option value="Female">Female (Nữ)</option>
                    <option value="Male">Male (Nam)</option>
                  </select>
                </div>

                <div class="form-group">
                  <label for="region">Khu vực sinh sống</label>
                  <select id="region" required>
                    <option value="Urban">Urban (Đô thị)</option>
                    <option value="Rural">Rural (Nông thôn)</option>
                  </select>
                </div>
              </div>

              <!-- Nhóm 2: Chuyên cần & Tự học -->
              <div class="form-section-title">2. Chuyên Cần & Thời Lượng Tự Học</div>
              <div class="form-grid">
                <div class="form-group full">
                  <label for="hours">
                    <span>Số giờ tự học mỗi tuần (HoursStudied)</span>
                    <span class="slider-val" id="hoursVal">5.5 giờ/tuần</span>
                  </label>
                  <input type="range" id="hoursRange" min="0" max="16" step="0.1" value="5.5" oninput="syncHours(this.value)">
                </div>

                <div class="form-group full">
                  <label for="attendance">
                    <span>Tỷ lệ chuyên cần đến lớp (Attendance)</span>
                    <span class="slider-val" id="attVal">65.0%</span>
                  </label>
                  <input type="range" id="attRange" min="50" max="100" step="0.1" value="65.0" oninput="syncAttendance(this.value)">
                </div>
              </div>

              <!-- Live Feature Engineering Display -->
              <div class="fe-preview-box">
                <div>
                  <div class="fe-item-title">Effective_Study_Hours (Giờ học hiệu quả)</div>
                  <div class="fe-item-val" id="feEffective">3.58 giờ</div>
                </div>
                <div>
                  <div class="fe-item-title">Study_Intensity_Ratio (Cường độ)</div>
                  <div class="fe-item-val" id="feIntensity">0.0833</div>
                </div>
              </div>

              <!-- Nhóm 3: Gia đình & Bổ trợ -->
              <div class="form-section-title">3. Bổ Trợ & Nền Tảng Gia Đình</div>
              <div class="form-grid">
                <div class="form-group">
                  <label for="tutoring">Học thêm (Tutoring)</label>
                  <select id="tutoring" required>
                    <option value="No">No (Không học thêm)</option>
                    <option value="Yes">Yes (Có học thêm)</option>
                  </select>
                </div>

                <div class="form-group">
                  <label for="parentEdu">Học vấn phụ huynh</label>
                  <select id="parentEdu" required>
                    <option value="None">None (Không bằng cấp)</option>
                    <option value="Primary">Primary (Tiểu học/THCS)</option>
                    <option value="Secondary">Secondary (THPT)</option>
                    <option value="Tertiary">Tertiary (Đại học/Cao đẳng)</option>
                  </select>
                </div>
              </div>

              <button type="submit" class="btn-submit" id="submitBtn">
                <span>🔍 Phân Tích Nguy Cơ & Dự Báo Học Lực</span>
              </button>
            </form>
          </div>
        </section>

        <!-- CỘT PHẢI: KẾT QUẢ PHÂN TÍCH & KẾ HOẠCH HÀNH ĐỘNG -->
        <section class="results-container">
          <!-- Banner Kết Quả Chính -->
          <div class="banner-card banner-risk" id="resBanner">
            <div class="banner-info">
              <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Chẩn Đoán Phân Tầng Học Lực</span>
              <h3 id="resClassName">Cần Hỗ Trợ (&lt;60 điểm)</h3>
              <p class="banner-desc" id="resClassDesc">Nhóm có nguy cơ kết quả thấp - Cần được kích hoạt kế hoạch can thiệp sớm</p>
            </div>
            <div class="urgency-badge urgency-risk" id="resUrgencyBadge">Khẩn Cấp</div>
          </div>

          <!-- Phân Phối Xác Suất Thực Tế -->
          <div class="prob-card">
            <div class="prob-header">
              <h4>📊 Phân Phối Xác Suất Thực Tế (predict_proba)</h4>
              <span style="font-size: 0.75rem; color: var(--slate-500);">Độ tin cậy mô hình</span>
            </div>

            <!-- Lớp 0 -->
            <div class="prob-row">
              <div class="prob-meta">
                <span>🔴 Cần Hỗ Trợ (&lt;60 điểm) [Nhóm Nguy Cơ]</span>
                <span id="probVal0" style="color: var(--rose-600); font-weight: 800;">99.8%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" id="probBar0" style="width: 99.8%; background: var(--rose-600);"></div>
              </div>
            </div>

            <!-- Lớp 1 -->
            <div class="prob-row">
              <div class="prob-meta">
                <span>🔵 Đạt Chuẩn [60 - 80 điểm) [Đạt Mục Tiêu]</span>
                <span id="probVal1" style="color: var(--primary-dark); font-weight: 800;">0.2%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" id="probBar1" style="width: 0.2%; background: var(--primary);"></div>
              </div>
            </div>

            <!-- Lớp 2 -->
            <div class="prob-row">
              <div class="prob-meta">
                <span>🟢 Xuất Sắc (&gt;=80 điểm) [Năng Lực Cao]</span>
                <span id="probVal2" style="color: var(--emerald-600); font-weight: 800;">0.0%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill" id="probBar2" style="width: 0.0%; background: var(--emerald-600);"></div>
              </div>
            </div>
          </div>

          <!-- Khuyến Nghị Sư Phạm 3 Chiều -->
          <div class="recs-card">
            <div class="recs-tabs">
              <button class="tab-btn active" onclick="switchRecTab('student', this)">🎒 Dành Cho Học Sinh</button>
              <button class="tab-btn" onclick="switchRecTab('teacher', this)">👨‍🏫 Dành Cho Giáo Viên & Cố Vấn</button>
              <button class="tab-btn" onclick="switchRecTab('parent', this)">👨‍👩‍👧 Dành Cho Phụ Huynh</button>
            </div>

            <ul class="recs-list" id="recsListContainer">
              <li><span class="recs-icon">✔</span><span>Tăng cường chuyên cần lên tối thiểu 85.0% để tiếp thu kiến thức trọng tâm.</span></li>
              <li><span class="recs-icon">✔</span><span>Tăng thời lượng tự học từ 5.5 giờ lên tối thiểu 10.0 - 12.0 giờ/tuần.</span></li>
              <li><span class="recs-icon">✔</span><span>Đăng ký tham gia ngay các lớp phụ đạo hoặc nhóm học kèm để bù đắp kiến thức rỗng.</span></li>
            </ul>
          </div>
        </section>
      </main>

      <footer>
        EduPredict AI • Machine Learning Classification Pipeline • Thư mục độc lập: <strong>EduPredict_Classification</strong>
      </footer>

      <script>
        let currentRecs = {
          student: [
            "Tăng tỷ lệ chuyên cần từ 65.0% lên tối thiểu 85.0% để đảm bảo tiếp thu kiến thức trọng tâm trên lớp.",
            "Nâng giờ tự học lên ít nhất 10.0 - 12.0 giờ/tuần, chia nhỏ các phiên học 25 phút (kỹ thuật Pomodoro).",
            "Đăng ký tham gia ngay các lớp phụ đạo hoặc nhóm học kèm để củng cố các lỗ hổng kiến thức căn bản."
          ],
          teacher: [
            "Cố vấn học tập thiết lập lịch phỏng vấn 1-1 để tìm hiểu nguyên nhân giảm sút chuyên cần.",
            "Cung cấp gói bài tập bổ trợ có phân cấp độ khó từ cơ bản đến trung bình.",
            "Xếp học sinh vào nhóm học tập có bạn học khá/giỏi kèm cặp (Peer Mentoring)."
          ],
          parent: [
            "Tạo không gian tự học yên tĩnh tại nhà, hạn chế thiết bị giải trí trước giờ học.",
            "Chủ động liên hệ giáo viên chủ nhiệm 2 tuần/lần để theo dõi tiến độ chuyên cần và bài tập.",
            "Động viên tinh thần tích cực, tránh tạo áp lực tiêu cực khiến học sinh chán nản."
          ]
        };

        function syncHours(val) {
          const num = parseFloat(val);
          document.getElementById('hoursVal').textContent = num.toFixed(1) + ' giờ/tuần';
          updateLiveFE();
        }

        function syncAttendance(val) {
          const num = parseFloat(val);
          document.getElementById('attVal').textContent = num.toFixed(1) + '%';
          updateLiveFE();
        }

        function updateLiveFE() {
          const h = parseFloat(document.getElementById('hoursRange').value);
          const a = parseFloat(document.getElementById('attRange').value);
          const eff = h * (a / 100.0);
          const intensity = h / (a + 1.0);
          document.getElementById('feEffective').textContent = eff.toFixed(2) + ' giờ';
          document.getElementById('feIntensity').textContent = intensity.toFixed(4);
        }

        function switchRecTab(stakeholder, btn) {
          document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          const list = currentRecs[stakeholder] || [];
          const container = document.getElementById('recsListContainer');
          container.innerHTML = '';
          list.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = '<span class="recs-icon">✔</span><span>' + item + '</span>';
            container.appendChild(li);
          });
        }

        function loadPreset(type) {
          if (type === 'risk') {
            document.getElementById('gender').value = 'Male';
            document.getElementById('region').value = 'Rural';
            document.getElementById('hoursRange').value = 4.5;
            syncHours(4.5);
            document.getElementById('attRange').value = 64.0;
            syncAttendance(64.0);
            document.getElementById('tutoring').value = 'No';
            document.getElementById('parentEdu').value = 'None';
          } else if (type === 'track') {
            document.getElementById('gender').value = 'Female';
            document.getElementById('region').value = 'Urban';
            document.getElementById('hoursRange').value = 9.5;
            syncHours(9.5);
            document.getElementById('attRange').value = 82.0;
            syncAttendance(82.0);
            document.getElementById('tutoring').value = 'No';
            document.getElementById('parentEdu').value = 'Secondary';
          } else if (type === 'excel') {
            document.getElementById('gender').value = 'Female';
            document.getElementById('region').value = 'Urban';
            document.getElementById('hoursRange').value = 14.5;
            syncHours(14.5);
            document.getElementById('attRange').value = 96.0;
            syncAttendance(96.0);
            document.getElementById('tutoring').value = 'Yes';
            document.getElementById('parentEdu').value = 'Tertiary';
          }
          document.getElementById('predictForm').dispatchEvent(new Event('submit'));
        }

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
          e.preventDefault();
          const btn = document.getElementById('submitBtn');
          btn.disabled = true;
          btn.innerHTML = '<span>⏳ Đang phân tích hồ sơ học sinh...</span>';

          const payload = {
            "Gender": document.getElementById('gender').value,
            "HoursStudied/Week": parseFloat(document.getElementById('hoursRange').value),
            "Tutoring": document.getElementById('tutoring').value,
            "Region": document.getElementById('region').value,
            "Attendance(%)": parseFloat(document.getElementById('attRange').value),
            "Parent Education": document.getElementById('parentEdu').value
          };

          try {
            const res = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
              const code = data.predicted_class_code;
              const banner = document.getElementById('resBanner');
              const nameEl = document.getElementById('resClassName');
              const descEl = document.getElementById('resClassDesc');
              const urgencyEl = document.getElementById('resUrgencyBadge');

              nameEl.textContent = data.predicted_class_name;
              descEl.textContent = data.class_description;
              urgencyEl.textContent = data.urgency_level;

              banner.className = 'banner-card';
              urgencyEl.className = 'urgency-badge';

              if (code === 0) {
                banner.classList.add('banner-risk');
                urgencyEl.classList.add('urgency-risk');
              } else if (code === 1) {
                banner.classList.add('banner-track');
                urgencyEl.classList.add('urgency-track');
              } else {
                banner.classList.add('banner-excel');
                urgencyEl.classList.add('urgency-excel');
              }

              // Cập nhật xác suất
              const probs = data.class_probabilities;
              const p0 = ((probs['Cần Hỗ Trợ (<60)'] || 0) * 100).toFixed(1);
              const p1 = ((probs['Đạt Chuẩn (60-80)'] || 0) * 100).toFixed(1);
              const p2 = ((probs['Xuất Sắc (>=80)'] || 0) * 100).toFixed(1);

              document.getElementById('probVal0').textContent = p0 + '%';
              document.getElementById('probBar0').style.width = p0 + '%';

              document.getElementById('probVal1').textContent = p1 + '%';
              document.getElementById('probBar1').style.width = p1 + '%';

              document.getElementById('probVal2').textContent = p2 + '%';
              document.getElementById('probBar2').style.width = p2 + '%';

              // Lưu khuyến nghị 3 chiều
              currentRecs.student = data.stakeholder_recommendations.for_student || [];
              currentRecs.teacher = data.stakeholder_recommendations.for_teacher || [];
              currentRecs.parent = data.stakeholder_recommendations.for_parent || [];

              // Render tab active hiện tại
              const activeTab = document.querySelector('.tab-btn.active');
              if (activeTab) activeTab.click();

            } else {
              alert('Lỗi: ' + (data.message || JSON.stringify(data.details || 'Không rõ')));
            }
          } catch (err) {
            alert('Lỗi kết nối API: ' + err.message);
          } finally {
            btn.disabled = false;
            btn.innerHTML = '<span>🔍 Phân Tích Nguy Cơ & Dự Báo Học Lực</span>';
          }
        });

        // Chạy khởi tạo ban đầu
        updateLiveFE();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_classifier:app", host="127.0.0.1", port=8000, reload=False)
