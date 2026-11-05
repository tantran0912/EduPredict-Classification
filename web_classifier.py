# -*- coding: utf-8 -*-
"""
EduPredict - Hệ thống Phân tích Học tập, Cảnh báo Sớm Học vụ & Lộ trình Học phần Thích ứng
Backend API: FastAPI + Scikit-Learn Pipeline + Gemini Academic Engine (Local & Render Ready)
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
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Request
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
SCHEMA_PATH = ARTIFACT_DIR / "classification_metrics_schema.json"
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = ARTIFACT_DIR / "classification_schema.json"
METRICS_PATH = ARTIFACT_DIR / "classification_metrics.json"
SAMPLE_CSV_PATH = BASE_DIR / "danh_sach_lop_mau_HaUI.csv"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
ENV_FILE_PATH = BASE_DIR / ".env"


def load_local_env() -> None:
    """Nạp biến môi trường từ tệp .env (nếu có) mà không cần thư viện ngoài."""
    for candidate in (ENV_FILE_PATH, BASE_DIR.parent / ".env"):
        if candidate.exists():
            try:
                for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
            except Exception as exc:
                logger.warning(f"Không thể đọc {candidate}: {exc}")


load_local_env()

# Danh sách model ưu tiên tốc độ phản hồi nhanh (2-4s) và tương thích cả AI Studio mới & cũ
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
]

CLASS_NAMES = ["Cần Hỗ Trợ (<60)", "Đạt Chuẩn (60-80)", "Xuất Sắc (>=80)"]
CLASS_LABELS_VN = {
    0: "Nguy cơ cao",
    1: "Đạt chuẩn",
    2: "Năng lực tốt",
}
CLASS_DESC_VN = {
    0: "Tỷ lệ chuyên cần hoặc quỹ giờ tự học đang dưới ngưỡng an toàn tín chỉ, có nguy cơ ảnh hưởng điều kiện dự thi và điểm kết thúc học phần.",
    1: "Nhịp độ học tập hiện tại đáp ứng chuẩn đầu ra học phần; có thể tối ưu thêm thời lượng tự học để bứt phá lên mức điểm Khá/Giỏi.",
    2: "Duy trì kỷ luật chuyên cần và cường độ tự học khoa học, nền tảng vững chắc để đạt kết quả Xuất sắc trong học phần.",
}


def resolve_gemini_key(client_key: Optional[str] = None) -> str:
    """Ưu tiên khóa do người dùng cấu hình trên trình duyệt, sau đó tới biến môi trường / .env."""
    if client_key and client_key.strip():
        return client_key.strip()
    load_local_env()
    return os.environ.get("GEMINI_API_KEY", "").strip()


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
    title="EduPredict - Hệ thống Phân tích Học tập & Cảnh báo Sớm Học vụ",
    version="5.0",
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
    api_key: Optional[str] = ""


class GeminiClassConsultInput(BaseModel):
    total_students: int = 0
    at_risk_count: int = 0
    on_track_count: int = 0
    excel_count: int = 0
    avg_attendance: float = 0.0
    below_80_count: int = 0
    avg_hours: float = 0.0
    low_hours_count: int = 0
    avg_risk_probability: float = 0.0
    tutoring_rate: float = 0.0
    top_subjects: List[str] = Field(default_factory=list)
    api_key: Optional[str] = ""


class ApiKeyInput(BaseModel):
    api_key: str


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
    min_safe_att = max(80.0, current_att)
    best_att = min_safe_att
    found_safe = False

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
        "early": "giai đoạn đầu học kỳ",
        "midterm": "giai đoạn giữa học kỳ",
        "final": "giai đoạn ôn thi kết thúc học phần",
    }.get(phase, "học kỳ hiện tại")

    student_actions: List[str] = []
    advisor_actions: List[str] = []

    if pred_code == 0:
        urgency = "Cần can thiệp sớm"
        if attendance < 80.0:
            student_actions.append(
                f"Chuyên cần hiện đạt {attendance:.0f}% (dưới ngưỡng an toàn điều kiện dự thi 80%). Cần tham dự đầy đủ các buổi học còn lại."
            )
        if hours < 7.5:
            student_actions.append(
                f"Quỹ tự học {hours:.1f} giờ/tuần còn thấp so với định mức tín chỉ; nên nâng lên tối thiểu 7.5 – 9.0 giờ/tuần theo lịch cố định."
            )
        student_actions.append(
            "Rà soát đề cương chi tiết học phần, hoàn thành đầy đủ bài kiểm tra thường xuyên và bài tập quá trình để giữ điểm điều kiện."
        )

        advisor_actions.append(
            f"Trao đổi trực tiếp với sinh viên trong {phase_label} nhằm xác định nguyên nhân vắng học hoặc vướng mắc phương pháp tiếp thu."
        )
        advisor_actions.append(
            "Nhắc nhở quy chế chuyên cần (tối thiểu 80%) và yêu cầu sinh viên cam kết tiến độ hoàn thành bài tập quá trình."
        )
        advisor_actions.append(
            "Ghép sinh viên vào nhóm học tập đồng đẳng (Peer Tutoring) hoặc kết nối trợ giảng phụ trách học phần đang gặp khó khăn."
        )
    elif pred_code == 1:
        urgency = "Duy trì ổn định"
        student_actions.append(
            f"Duy trì mức chuyên cần {attendance:.0f}% và bổ sung thêm 1.5 – 2.0 giờ tự học/tuần để nâng kết quả từ mức Đạt lên Khá/Giỏi."
        )
        student_actions.append(
            "Hệ thống hóa lý thuyết theo từng chương bằng sơ đồ tư duy và tự giải lại các dạng bài kiểm tra tổng hợp."
        )

        advisor_actions.append(
            "Ghi nhận tiến độ học tập ổn định; định hướng sinh viên đặt mục tiêu điểm B+/A ở các học phần cốt lõi chuyên ngành."
        )
        advisor_actions.append(
            "Khuyến khích sinh viên tham gia thảo luận nhóm để củng cố chiều sâu vận dụng lý thuyết."
        )
    else:
        urgency = "Phát triển nâng cao"
        student_actions.append(
            "Phát huy kỷ luật tự học khoa học hiện tại; chủ động mở rộng sang các bài toán vận dụng cao, đồ án hoặc đề tài nghiên cứu."
        )
        advisor_actions.append(
            "Đề xuất sinh viên làm nhóm trưởng học tập hỗ trợ các bạn trong lớp hoặc tham gia nhóm Nghiên cứu khoa học sinh viên."
        )

    return {
        "urgency_level": urgency,
        "student_actions": student_actions,
        "advisor_actions": advisor_actions,
    }


def call_gemini_api(prompt: str, custom_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Gọi Gemini API với cơ chế phát hiện nhanh lỗi khóa API (400/401/403).
    Trả về: (text_json, error_reason)
    """
    active_key = resolve_gemini_key(custom_key)
    if not active_key:
        return None, "missing_key"

    last_status = None
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={active_key}"
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
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip(), None

            err_body = resp.text[:300]
            logger.warning(f"Model {model_name} trả về {resp.status_code}: {err_body}")

            # Nếu khóa không hợp lệ hoặc bị thu hồi (401 / 403 / API_KEY_INVALID), dừng ngay không thử lại vô ích
            if resp.status_code in (401, 403) or "API_KEY_INVALID" in err_body or "API key not valid" in err_body:
                return None, "invalid_key"
            if resp.status_code == 429:
                last_status = 429
                continue
        except Exception as exc:
            logger.warning(f"Lỗi kết nối model {model_name}: {exc}")
            continue

    if last_status == 429:
        return None, "quota_exceeded"
    return None, "service_unavailable"


def gemini_error_message(reason: Optional[str]) -> str:
    if reason == "missing_key":
        return (
            "Chưa cấu hình Gemini API Key. Hãy nhấn nút 'Cấu hình AI' ở thanh trên cùng để dán API Key mới "
            "(hoặc đặt biến môi trường GEMINI_API_KEY trên Render / tệp .env)."
        )
    if reason == "invalid_key":
        return (
            "API Key hiện tại không hợp lệ hoặc đã bị Google tự động khóa do từng đẩy công khai lên GitHub. "
            "Bạn hãy tạo khóa mới tại Google AI Studio (aistudio.google.com/apikey) rồi nhấn 'Cấu hình AI' ở góc phải phía trên để cập nhật."
        )
    if reason == "quota_exceeded":
        return "API Key hiện tại đang vượt giới hạn tần suất gọi miễn phí (Rate Limit 429). Vui lòng đợi khoảng 30 giây rồi thử lại."
    return "Không thể kết nối tới dịch vụ Gemini lúc này. Vui lòng kiểm tra kết nối mạng hoặc khóa API trong mục 'Cấu hình AI'."


@app.get("/health")
def health() -> Dict[str, Any]:
    key = resolve_gemini_key()
    return {
        "status": "ok" if bundle.ready else "not_ready",
        "model_name": bundle.model_name,
        "version": "5.0",
        "ai_key_configured": bool(key),
    }


@app.get("/api/key-status")
def key_status() -> Dict[str, Any]:
    key = resolve_gemini_key()
    masked = ""
    if len(key) > 10:
        masked = f"{key[:6]}...{key[-4:]}"
    return {"configured": bool(key), "masked": masked}


@app.post("/api/verify-key")
def verify_and_save_key(payload: ApiKeyInput, request: Request) -> JSONResponse:
    candidate = payload.api_key.strip()
    if not candidate or len(candidate) < 15:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Khóa API không đúng định dạng."})

    test_prompt = 'Trả về JSON: {"ok": true}'
    raw, err = call_gemini_api(test_prompt, custom_key=candidate)
    if err is not None or not raw:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": gemini_error_message(err or "invalid_key")},
        )

    # Nếu đang chạy ở máy local (localhost / 127.0.0.1), lưu vào tệp .env (đã được .gitignore bảo vệ khỏi GitHub)
    client_host = request.client.host if request.client else ""
    saved_local = False
    if client_host in ("127.0.0.1", "::1", "localhost"):
        try:
            os.environ["GEMINI_API_KEY"] = candidate
            ENV_FILE_PATH.write_text(f"GEMINI_API_KEY={candidate}\n", encoding="utf-8")
            saved_local = True
        except Exception as exc:
            logger.warning(f"Không thể lưu .env local: {exc}")

    return JSONResponse(
        content={
            "status": "success",
            "saved_to_local_env": saved_local,
            "masked": f"{candidate[:6]}...{candidate[-4:]}",
            "message": "Xác thực khóa Gemini thành công! Trí tuệ nhân tạo đã sẵn sàng hoạt động.",
        }
    )


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
    """Chuẩn hóa linh hoạt tên cột từ CSV/Excel người dùng tải lên, tránh xung đột tên cột."""
    col_map: Dict[str, str] = {}
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("masv", "ma_sv", "mã sv", "mã sinh viên", "mssv", "student_id", "id"):
            col_map[col] = "MaSV"
        elif c in ("hoten", "ho_ten", "họ tên", "họ và tên", "full_name", "name", "ten", "tên"):
            col_map[col] = "HoTen"
        elif c in ("gender", "gioitinh", "giới tính", "phai", "phái"):
            col_map[col] = "Gender"
        elif c in ("region", "khuvuc", "khu vực", "khu_vuc", "vung", "vùng"):
            col_map[col] = "Region"
        elif (
            c in ("hoursstudied/week", "hours_studied", "hours", "so_gio", "số giờ")
            or "hour" in c
            or "giờ" in c
            or "tự học" in c
            or "tuhoc" in c
            or "gio hoc" in c
            or "gio tu hoc" in c
        ):
            col_map[col] = "HoursStudied/Week"
        elif "attend" in c or "chuyên cần" in c or "chuyencan" in c:
            col_map[col] = "Attendance(%)"
        elif "tutor" in c or "học thêm" in c or "phụ đạo" in c or "hocnhom" in c or "học nhóm" in c:
            col_map[col] = "Tutoring"
        elif "parent" in c or "phụ huynh" in c or "gia đình" in c or "giadinh" in c:
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


def build_class_overview_and_recommendations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp phân tích năng lực toàn lớp (chuyên cần, tự học, rủi ro, môn học điểm nóng) & khuyến nghị cho giảng viên."""
    total = len(results)
    if total == 0:
        return {}

    atts = [float(r["attendance"]) for r in results]
    hours = [float(r["hours_studied"]) for r in results]
    risks = [float(r["risk_probability"]) for r in results]

    at_risk_count = sum(1 for r in results if r["predicted_code"] == 0)
    on_track_count = sum(1 for r in results if r["predicted_code"] == 1)
    excel_count = sum(1 for r in results if r["predicted_code"] == 2)

    at_risk_rate = round((at_risk_count / total) * 100, 1)
    on_track_rate = round((on_track_count / total) * 100, 1)
    excel_rate = round((excel_count / total) * 100, 1)
    avg_risk_prob = round(float(np.mean(risks)), 1)

    # Phân tích chuyên cần
    avg_att = round(float(np.mean(atts)), 1)
    min_att = round(float(np.min(atts)), 1)
    max_att = round(float(np.max(atts)), 1)
    critical_att_count = sum(1 for a in atts if a < 70.0)
    warn_att_count = sum(1 for a in atts if 70.0 <= a < 80.0)
    below_80_count = critical_att_count + warn_att_count
    below_80_rate = round((below_80_count / total) * 100, 1)
    safe_80_plus_count = total - below_80_count
    safe_80_plus_rate = round((safe_80_plus_count / total) * 100, 1)

    # Phân tích giờ tự học
    avg_h = round(float(np.mean(hours)), 1)
    min_h = round(float(np.min(hours)), 1)
    max_h = round(float(np.max(hours)), 1)
    low_h_count = sum(1 for h in hours if h < 5.0)
    mod_h_count = sum(1 for h in hours if 5.0 <= h < 8.0)
    high_h_count = sum(1 for h in hours if h >= 8.0)
    low_h_rate = round((low_h_count / total) * 100, 1)
    mod_h_rate = round((mod_h_count / total) * 100, 1)
    high_h_rate = round((high_h_count / total) * 100, 1)

    # Học nhóm / phụ đạo
    tutoring_count = sum(1 for r in results if r.get("tutoring") == "Yes")
    tutoring_rate = round((tutoring_count / total) * 100, 1)

    # Môn học điểm nóng trong lớp
    subj_stats: Dict[str, Dict[str, int]] = {}
    for r in results:
        sub = str(r.get("struggling_subject", "")).strip()
        if not sub or sub == "Chưa ghi chú":
            continue
        if sub not in subj_stats:
            subj_stats[sub] = {"total": 0, "at_risk": 0}
        subj_stats[sub]["total"] += 1
        if r["predicted_code"] == 0:
            subj_stats[sub]["at_risk"] += 1

    hotspots = sorted(
        [
            {
                "subject": k,
                "total_count": v["total"],
                "at_risk_count": v["at_risk"],
            }
            for k, v in subj_stats.items()
        ],
        key=lambda x: (x["at_risk_count"], x["total_count"]),
        reverse=True,
    )[:6]

    # Chẩn đoán sức khỏe học thuật tổng thể của lớp
    if at_risk_rate >= 35.0 or below_80_rate >= 35.0:
        health_code = 0
        health_label = "Cần can thiệp diện rộng"
        summary_text = (
            f"Năng lực lớp đang có sự phân hóa mạnh: {at_risk_count}/{total} sinh viên ({at_risk_rate}%) thuộc diện Nguy cơ cao, "
            f"trong khi nhóm Năng lực tốt và Đạt chuẩn chiếm {round(100 - at_risk_rate, 1)}%. "
            f"Nguyên nhân trọng yếu đến từ nhóm {below_80_count} sinh viên có chuyên cần dưới ngưỡng dự thi 80% (trung bình lớp {avg_att}%) "
            f"kết hợp với {low_h_count} sinh viên tự học dưới 5 giờ/tuần (trung bình lớp {avg_h} giờ/tuần)."
        )
    elif at_risk_rate >= 15.0 or below_80_count > 0:
        health_code = 1
        health_label = "Cần theo dõi trọng điểm"
        summary_text = (
            f"Mặt bằng chung của lớp tương đối khả quan với {on_track_count + excel_count}/{total} sinh viên nằm trong vùng an toàn "
            f"(chuyên cần trung bình {avg_att}%, tự học trung bình {avg_h} giờ/tuần). "
            f"Tuy nhiên vẫn còn {at_risk_count} sinh viên ({at_risk_rate}%) thuộc nhóm Nguy cơ cao cần được cố vấn học tập giám sát sát sao trước kỳ thi."
        )
    else:
        health_code = 2
        health_label = "Năng lực lớp ổn định"
        summary_text = (
            f"Tập thể lớp duy trì nền nếp học tập rất tốt: {round(100 - at_risk_rate, 1)}% sinh viên đạt chuẩn và xuất sắc, "
            f"chuyên cần trung bình đạt {avg_att}% và thời lượng tự học đạt {avg_h} giờ/tuần. "
            f"Giảng viên có thể tập trung nâng cao độ sâu bài tập vận dụng và bồi dưỡng nhóm sinh viên xuất sắc."
        )

    # Khuyến nghị hành động cụ thể cho Giảng viên / Cố vấn học tập
    recommendations: List[Dict[str, str]] = []

    # 1. Khuyến nghị về Chuyên cần & Quy chế
    if below_80_count > 0:
        crit_note = f" (đặc biệt có {critical_att_count} SV vắng nghiêm trọng dưới 70%)" if critical_att_count > 0 else ""
        recommendations.append(
            {
                "category": "Chuyên cần & Điều kiện dự thi",
                "level": "danger" if below_80_rate >= 25 else "warn",
                "title": f"Cảnh báo học vụ sớm với {below_80_count}/{total} sinh viên dưới ngưỡng 80%",
                "detail": (
                    f"Hiện có {below_80_rate}% sĩ số lớp chưa đạt ngưỡng chuyên cần an toàn 80%{crit_note}. "
                    f"Giảng viên nên công bố danh sách nhắc nhở ngay trong tuần này và yêu cầu nhóm này ký cam kết đi học đầy đủ các buổi còn lại."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "category": "Chuyên cần & Điều kiện dự thi",
                "level": "success",
                "title": "100% sinh viên đảm bảo ngưỡng chuyên cần dự thi (>= 80%)",
                "detail": (
                    f"Tỷ lệ chuyên cần trung bình của lớp đạt {avg_att}% (thấp nhất {min_att}%). "
                    f"Tiếp tục duy trì điểm danh kết hợp câu hỏi kiểm tra nhanh đầu giờ để giữ vững kỷ luật lên lớp."
                ),
            }
        )

    # 2. Khuyến nghị về Giờ tự học & Thiết kế bài tập quá trình
    if avg_h < 7.5 or low_h_count > 0:
        recommendations.append(
            {
                "category": "Tự học & Kiểm tra quá trình",
                "level": "warn" if low_h_rate >= 25 else "info",
                "title": f"Chia nhỏ bài tập theo tuần để kéo nhóm {low_h_count} sinh viên tự học thấp (< 5h/tuần)",
                "detail": (
                    f"Quỹ tự học trung bình của lớp đạt {avg_h} giờ/tuần (định mức khuyến nghị đại học: 8.0 giờ/tuần). "
                    f"Thay vì giao bài tập lớn dồn vào cuối kỳ, giảng viên nên chia thành các mốc nộp bài ngắn hàng tuần trên LMS để buộc nhóm tự học ít phải tích lũy đều đặn."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "category": "Tự học & Kiểm tra quá trình",
                "level": "success",
                "title": f"Cường độ tự học tốt ({avg_h} giờ/tuần) — Tăng cường bài toán vận dụng",
                "detail": (
                    f"Có {high_h_count}/{total} sinh viên ({high_h_rate}%) tự học từ 8 giờ/tuần trở lên. "
                    f"Giảng viên có thể bổ sung các câu hỏi phân loại mức độ Khá/Giỏi và chủ đề thảo luận thực tiễn."
                ),
            }
        )

    # 3. Khuyến nghị về Mô hình Nhóm học tập đồng đẳng (Peer Tutoring)
    core_mentors = excel_count + on_track_count
    if at_risk_count > 0 and core_mentors > 0:
        recommendations.append(
            {
                "category": "Tổ chức lớp & Kèm cặp đồng đẳng",
                "level": "info",
                "title": f"Thiết lập nhóm học tập: Phát huy {core_mentors} sinh viên khá/tốt kèm {at_risk_count} sinh viên nguy cơ",
                "detail": (
                    f"Hiện mới có {tutoring_count}/{total} sinh viên ({tutoring_rate}%) tham gia học nhóm/phụ đạo. "
                    f"Cố vấn học tập nên chia lớp thành các tổ 4–5 sinh viên, phân công {excel_count} sinh viên nhóm Năng lực tốt và {on_track_count} sinh viên Đạt chuẩn làm nhóm trưởng kèm cặp trực tiếp nhóm nguy cơ."
                ),
            }
        )
    elif at_risk_count > 0:
        recommendations.append(
            {
                "category": "Tổ chức lớp & Phụ đạo",
                "level": "danger",
                "title": "Tổ chức buổi phụ đạo hệ thống hóa kiến thức nền tảng cho cả lớp",
                "detail": (
                    f"Phần lớn lớp đang nằm trong vùng rủi ro và chỉ có {tutoring_rate}% sinh viên có hỗ trợ học nhóm. "
                    f"Khoa/Bộ môn nên bố trí 1–2 buổi phụ đạo tập trung hoặc cử trợ giảng chữa đề cương trọng tâm trước kiểm tra giữa kỳ."
                ),
            }
        )
    else:
        recommendations.append(
            {
                "category": "Phát triển mũi nhọn",
                "level": "success",
                "title": f"Bồi dưỡng {excel_count} sinh viên nhóm Năng lực tốt tham gia NCKH / Olympic",
                "detail": "Lớp có lực lượng sinh viên tự giác cao; thuận lợi để thành lập câu lạc bộ học thuật hoặc nhóm nghiên cứu khoa học sinh viên."
            }
        )

    # 4. Khuyến nghị theo Học phần điểm nóng của lớp
    if hotspots:
        top_names = ", ".join(
            [f"{h['subject']} ({h['at_risk_count']}/{h['total_count']} SV nguy cơ)" for h in hotspots[:3]]
        )
        recommendations.append(
            {
                "category": "Học phần trọng điểm của lớp",
                "level": "warn" if hotspots[0]["at_risk_count"] > 0 else "info",
                "title": "Ưu tiên rà soát & hỗ trợ tài liệu ở các môn tập trung nhiều rủi ro nhất",
                "detail": (
                    f"Dữ liệu lớp cho thấy vướng mắc tập trung nhiều nhất tại: {top_names}. "
                    f"Cố vấn học tập nên trao đổi với giảng viên giảng dạy các học phần này để gửi sớm bộ câu hỏi ôn tập và chữa các lỗi sai điển hình cho lớp."
                ),
            }
        )

    return {
        "class_health": {
            "code": health_code,
            "label": health_label,
            "summary_text": summary_text,
        },
        "competency": {
            "total": total,
            "at_risk_count": at_risk_count,
            "at_risk_rate": at_risk_rate,
            "on_track_count": on_track_count,
            "on_track_rate": on_track_rate,
            "excel_count": excel_count,
            "excel_rate": excel_rate,
            "avg_risk_probability": avg_risk_prob,
        },
        "attendance_stats": {
            "avg": avg_att,
            "min": min_att,
            "max": max_att,
            "below_80_count": below_80_count,
            "below_80_rate": below_80_rate,
            "critical_below_70_count": critical_att_count,
            "warn_70_79_count": warn_att_count,
            "safe_80_plus_count": safe_80_plus_count,
            "safe_80_plus_rate": safe_80_plus_rate,
            "critical_below_70_rate": round((critical_att_count / total) * 100, 1),
            "warn_70_79_rate": round((warn_att_count / total) * 100, 1),
        },
        "study_stats": {
            "avg": avg_h,
            "min": min_h,
            "max": max_h,
            "benchmark": 8.0,
            "low_under_5_count": low_h_count,
            "low_under_5_rate": low_h_rate,
            "mod_5_8_count": mod_h_count,
            "mod_5_8_rate": mod_h_rate,
            "high_8_plus_count": high_h_count,
            "high_8_plus_rate": high_h_rate,
            "tutoring_count": tutoring_count,
            "tutoring_rate": tutoring_rate,
        },
        "subject_hotspots": hotspots,
        "faculty_recommendations": recommendations,
    }


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
    overview = build_class_overview_and_recommendations(results)
    return {"summary": summary, "overview": overview, "students": results}


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
                    "direct_explanation": "Bạn chưa nhập tên học phần hoặc lý thuyết cụ thể. Hãy nhập tên môn học (ví dụ: Giải tích 1, Quy hoạch tuyến tính, Cơ sở dữ liệu...) và phần kiến thức đang vướng mắc để nhận giải đáp chuyên môn và lộ trình học phần chi tiết.",
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

    subject_display = subject if subject else "Học phần đại học liên quan trực tiếp tới câu hỏi"
    question_display = (
        bottleneck
        if bottleneck
        else f"Tổng quan trọng tâm kiến thức cốt lõi và phương pháp học hiệu quả môn {subject}"
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
1. Đi thẳng vào kiến thức chuyên môn chính xác của môn "{subject_display}" và vấn đề "{question_display}". Tuyệt đối KHÔNG trả lời chung chung sáo rỗng, KHÔNG dùng văn mẫu có sẵn. Dù sinh viên hỏi bất kỳ môn toán, tối ưu/quy hoạch tuyến tính, kỹ thuật, CNTT, kinh tế, ngoại ngữ hay lý luận chính trị nào, phải giải thích chuẩn xác bản chất lý thuyết/công thức/quy tắc biến đổi của đúng chủ đề đó.
2. Lộ trình học phần ("course_roadmap") phải được chia thành 3 hoặc 4 chặng rõ ràng, bám sát nội dung thực tế của học phần "{subject_display}", phân bổ thời lượng hợp lý với quỹ thời gian {payload.hours_studied:.1f} giờ/tuần và giai đoạn "{phase_text}".

TRẢ VỀ DUY NHẤT 1 JSON HỢP LỆ THEO ĐÚNG CẤU TRÚC SAU:
{{
  "topic_title": "Tên chuẩn xác của chủ đề lý thuyết và học phần",
  "direct_explanation": "Giải thích trực diện, rõ ràng, chuẩn lý thuyết chuyên ngành về vấn đề sinh viên hỏi (nêu rõ bản chất, định nghĩa, công thức hoặc quy tắc chuyển đổi cốt lõi dễ hiểu nhất trong 4-6 câu).",
  "key_takeaways": [
    "Ý trọng tâm 1: Công thức / nguyên lý cốt lõi hoặc cách phân biệt khái niệm dễ nhầm",
    "Ý trọng tâm 2: Quy tắc áp dụng từng bước khi giải bài tập hoặc thực hành",
    "Ý trọng tâm 3: Sai lầm / bẫy đề thi phổ biến nhất ở phần này cần tránh"
  ],
  "course_roadmap": [
    {{
      "stage": "Chặng 1: Tên chặng cụ thể theo nội dung môn học",
      "duration": "Thời lượng phân bổ phù hợp (ví dụ: Tuần 1 - 2.5 giờ)",
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

    ai_raw, err_reason = call_gemini_api(prompt, custom_key=payload.api_key)
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
            "error_code": err_reason or "service_unavailable",
            "message": gemini_error_message(err_reason),
        },
    )


@app.post("/api/gemini-class-consult")
def gemini_class_consult(payload: GeminiClassConsultInput) -> JSONResponse:
    subjects_str = ", ".join(payload.top_subjects) if payload.top_subjects else "Các học phần đại cương và cơ sở ngành"
    prompt = f"""Bạn là chuyên gia Quản lý Giáo dục Đại học và Cố vấn Học tập giàu kinh nghiệm. Dựa trên số liệu thống kê thực tế của lớp học vừa phân tích dưới đây, hãy đề xuất CHIẾN LƯỢC SƯ PHẠM & KẾ HOẠCH CAN THIỆP LỚP HỌC cụ thể, thiết thực cho Giảng viên.

THỐNG KÊ THỰC TẾ CỦA LỚP HỌC:
- Tổng sĩ số: {payload.total_students} sinh viên
- Phân tầng năng lực: Nguy cơ cao: {payload.at_risk_count} SV | Đạt chuẩn: {payload.on_track_count} SV | Năng lực tốt: {payload.excel_count} SV
- Tỷ lệ chuyên cần trung bình lớp: {payload.avg_attendance:.1f}% (Số SV dưới ngưỡng dự thi 80%: {payload.below_80_count} SV)
- Thời lượng tự học trung bình lớp: {payload.avg_hours:.1f} giờ/tuần (Số SV tự học dưới 5h/tuần: {payload.low_hours_count} SV)
- Tỷ lệ đang tham gia học nhóm/phụ đạo: {payload.tutoring_rate:.1f}% | Xác suất rủi ro TB: {payload.avg_risk_probability:.1f}%
- Các học phần có nhiều sinh viên gặp khó khăn nhất trong lớp: {subjects_str}

TRẢ VỀ DUY NHẤT 1 JSON HỢP LỆ THEO CẤU TRÚC SAU:
{{
  "strategy_title": "Tiêu đề ngắn gọn cho chiến lược can thiệp sư phạm của lớp",
  "pedagogical_insight": "Nhận định chuyên môn sâu sắc (3-4 câu) về mối liên hệ giữa chuyên cần, tự học và các môn học điểm nóng của lớp này.",
  "action_plan": [
    {{
      "phase": "Tuần 1: Ổn định kỷ luật & Rà soát cá nhân",
      "action": "Hành động cụ thể cho giảng viên / cố vấn học tập với con số sát thực tế của lớp"
    }},
    {{
      "phase": "Tuần 2-3: Phụ đạo trọng điểm & Nhóm học tập đồng đẳng",
      "action": "Cách ghép nhóm và hỗ trợ chuyên môn cho các môn điểm nóng ({subjects_str})"
    }},
    {{
      "phase": "Đánh giá giữa kỳ / Trước thi",
      "action": "Tiêu chí kiểm tra tiến bộ của nhóm nguy cơ cao"
    }}
  ]
}}"""

    ai_raw, err_reason = call_gemini_api(prompt, custom_key=payload.api_key)
    if ai_raw:
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", ai_raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            res_json = json.loads(cleaned.strip())
            return JSONResponse(content={"status": "success", "data": res_json})
        except Exception:
            match = re.search(r"\{[\s\S]*\}", ai_raw)
            if match:
                try:
                    return JSONResponse(content={"status": "success", "data": json.loads(match.group(0))})
                except Exception:
                    pass

    return JSONResponse(
        status_code=502,
        content={
            "status": "error",
            "error_code": err_reason or "service_unavailable",
            "message": gemini_error_message(err_reason),
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

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web_classifier:app", host=host, port=port, reload=False)
