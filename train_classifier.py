# -*- coding: utf-8 -*-
"""
EduPredict AI - Pipeline Machine Learning Phân Lớp & Cảnh Báo Sớm
Tập dữ liệu: SAP-4000.csv
Thư mục: EduPredict_Classification
File: train_classifier.py
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
RAW_NUMERIC_COLS = ["HoursStudied/Week", "Attendance(%)"]
CATEGORICAL_COLS = ["Gender", "Tutoring", "Region", "Parent Education"]
FEATURE_COLS = RAW_NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "Exam_Score"
MISSING_TOKENS = ["", "NA", "N/A", "null", "NULL", "nan", "NaN", "?"]

CLASS_NAMES = ["Cần Hỗ Trợ (<60)", "Đạt Chuẩn (60-80)", "Xuất Sắc (>=80)"]
CLASS_CODES = [0, 1, 2]
CLASS_DESC = {
    0: "Nhóm nguy cơ kết quả thấp (Exam_Score < 60) - Cần hỗ trợ và can thiệp kịp thời",
    1: "Nhóm đạt chuẩn học tập (60 <= Exam_Score < 80) - Tiếp tục duy trì nhịp độ ổn định",
    2: "Nhóm học lực xuất sắc (Exam_Score >= 80) - Có năng lực cao, khuyến khích bồi dưỡng nâng cao",
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)


def run_eda(csv_path: Path, outdir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    print("=" * 90)
    print("BƯỚC 1: KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU ĐẦU VÀO (EDA)")
    print("=" * 90)

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", keep_default_na=False, na_values=MISSING_TOKENS)
    if len(df.columns) == 1:
        df = pd.read_csv(csv_path, sep=",", keep_default_na=False, na_values=MISSING_TOKENS)
    df.columns = [str(c).strip() for c in df.columns]

    print(f"- Kích thước dữ liệu gốc: {df.shape[0]} hàng x {df.shape[1]} cột")
    print(f"- Các cột dữ liệu: {list(df.columns)}")

    for c in RAW_NUMERIC_COLS + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Bảo toàn 'None' trong Parent Education (phụ huynh không có bằng cấp)
    df["Parent Education"] = df["Parent Education"].fillna("None")

    y_raw = df[TARGET_COL].dropna()
    eda_stats = {
        "count": int(len(y_raw)),
        "mean": float(y_raw.mean()),
        "std": float(y_raw.std()),
        "min": float(y_raw.min()),
        "q25": float(y_raw.quantile(0.25)),
        "median": float(y_raw.median()),
        "q75": float(y_raw.quantile(0.75)),
        "max": float(y_raw.max()),
        "iqr": float(y_raw.quantile(0.75) - y_raw.quantile(0.25)),
    }

    print("\n[Thống kê phân phối điểm thi Exam_Score]:")
    print(f"  + Số lượng mẫu hợp lệ : {eda_stats['count']}")
    print(f"  + Trung bình (Mean)   : {eda_stats['mean']:.2f} | Độ lệch chuẩn (Std): {eda_stats['std']:.2f}")
    print(f"  + Min: {eda_stats['min']:.1f} | Q1 (25%): {eda_stats['q25']:.1f} | Trung vị (Median): {eda_stats['median']:.1f} | Q3 (75%): {eda_stats['q75']:.1f} | Max: {eda_stats['max']:.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)
    axes[0].hist(y_raw, bins=30, color="#0284c7", edgecolor="white", alpha=0.85)
    axes[0].axvline(60, color="#ef4444", linestyle="--", linewidth=2, label="Ngưỡng Nguy Cơ (<60)")
    axes[0].axvline(80, color="#10b981", linestyle="--", linewidth=2, label="Ngưỡng Xuất Sắc (>=80)")
    axes[0].set_title("Phân phối Điểm thi Exam_Score & Các Ngưỡng Phân Lớp", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Exam_Score")
    axes[0].set_ylabel("Tần số (Số học sinh)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    bp = axes[1].boxplot(y_raw, vert=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bae6fd")
    axes[1].set_title("Biểu đồ Hộp (Boxplot) Điểm thi Exam_Score", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Exam_Score")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    eda_plot_path = outdir / "eda_exam_score_distribution.png"
    fig.savefig(eda_plot_path)
    plt.close(fig)

    return df, eda_stats


def create_classification_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    print("\n" + "=" * 90)
    print("BƯỚC 2 & 3: THIẾT LẬP NHÃN PHÂN LỚP & ĐẶC TRƯNG MIỀN GIÁO DỤC")
    print("=" * 90)

    y_raw = df[TARGET_COL].values
    y_labels = pd.cut(
        y_raw,
        bins=[-np.inf, 60.0, 80.0, np.inf],
        labels=[0, 1, 2],
        right=False,
    ).astype(int)

    df_out = df.copy()
    df_out["Class_Label"] = y_labels
    df_out["Class_Name"] = [CLASS_NAMES[i] for i in y_labels]

    counts = pd.Series(y_labels).value_counts().sort_index()
    percentages = (counts / len(df) * 100).round(2)
    print("Phân bố số lượng mẫu và tỷ lệ từng lớp:")
    for code, name in enumerate(CLASS_NAMES):
        print(f"  + Lớp {code} - {name:<22}: {counts[code]:>5} mẫu ({percentages[code]:>5}%) | Ý nghĩa: {CLASS_DESC[code]}")

    df_out["Effective_Study_Hours"] = df_out["HoursStudied/Week"] * (df_out["Attendance(%)"] / 100.0)
    df_out["Study_Intensity_Ratio"] = df_out["HoursStudied/Week"] / (df_out["Attendance(%)"] + 1.0)

    return df_out, pd.Series(y_labels)


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    print("\n" + "=" * 90)
    print("BƯỚC 4: CHIA DỮ LIỆU STRATIFIED 70% TRAIN / 15% VALIDATION / 15% TEST")
    print("=" * 90)

    feature_cols = [
        "HoursStudied/Week",
        "Attendance(%)",
        "Effective_Study_Hours",
        "Study_Intensity_Ratio",
        "Gender",
        "Tutoring",
        "Region",
        "Parent Education",
    ]

    X = df[feature_cols]
    y = df["Class_Label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    print(f"- Tập Train      : {len(X_train)} mẫu (70.0%)")
    print(f"- Tập Validation : {len(X_val)} mẫu (15.0%)")
    print(f"- Tập Test       : {len(X_test)} mẫu (15.0%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def build_preprocessor() -> ColumnTransformer:
    num_cols = ["HoursStudied/Week", "Attendance(%)", "Effective_Study_Hours", "Study_Intensity_Ratio"]
    cat_cols = ["Gender", "Tutoring", "Region", "Parent Education"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return preprocessor


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: Path,
) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("BƯỚC 5: HUẤN LUYỆN & 5-FOLD STRATIFIED CROSS-VALIDATION (TẬP TRAIN - 2800 MẪU)")
    print("=" * 90)

    preprocessor = build_preprocessor()

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=1.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Support Vector Machine (SVM)": SVC(
            C=2.5,
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=4,
        ),
        "CatBoost Classifier": CatBoostClassifier(
            iterations=450,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            thread_count=4,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_records = []

    for name, clf in candidate_models.items():
        t0 = time.time()
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=["accuracy", "f1_macro", "recall_macro", "precision_macro"],
            n_jobs=1,
        )
        elapsed = time.time() - t0
        cv_records.append(
            {
                "Model": name,
                "CV_Accuracy_Mean": cv_res["test_accuracy"].mean(),
                "CV_Accuracy_Std": cv_res["test_accuracy"].std(),
                "CV_F1_Macro": cv_res["test_f1_macro"].mean(),
                "CV_Recall_Macro": cv_res["test_recall_macro"].mean(),
                "CV_Precision_Macro": cv_res["test_precision_macro"].mean(),
                "Train_Time_Sec": round(elapsed, 2),
            }
        )

    df_cv = pd.DataFrame(cv_records)
    print(f"{'Mô hình':<30} | {'CV Accuracy':>15} | {'CV F1-Macro':>12} | {'CV Recall':>10} | {'Thời gian':>10}")
    print("-" * 88)
    for _, r in df_cv.iterrows():
        print(
            f"{r['Model']:<30} | {r['CV_Accuracy_Mean']:>7.4f} ± {r['CV_Accuracy_Std']:>5.4f} | "
            f"{r['CV_F1_Macro']:>12.4f} | {r['CV_Recall_Macro']:>10.4f} | {r['Train_Time_Sec']:>8.2f}s"
        )

    print("\n" + "=" * 90)
    print("BƯỚC 6: ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (600 MẪU) - ƯU TIÊN RECALL & F1 LỚP NGUY CƠ")
    print("=" * 90)

    test_results = []
    fitted_pipelines = {}
    test_preds_dict = {}
    test_probs_dict = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None

        test_preds_dict[name] = y_pred
        test_probs_dict[name] = y_prob

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro = recall_score(y_test, y_pred, average="macro")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        rec_at_risk = recall_score(y_test, y_pred, labels=[0], average="macro")
        prec_at_risk = precision_score(y_test, y_pred, labels=[0], average="macro")
        f1_at_risk = f1_score(y_test, y_pred, labels=[0], average="macro")

        test_results.append(
            {
                "Model": name,
                "Accuracy": round(float(acc), 4),
                "Precision_Macro": round(float(prec_macro), 4),
                "Recall_Macro": round(float(rec_macro), 4),
                "F1_Macro": round(float(f1_macro), 4),
                "At_Risk_Recall": round(float(rec_at_risk), 4),
                "At_Risk_Precision": round(float(prec_at_risk), 4),
                "At_Risk_F1": round(float(f1_at_risk), 4),
            }
        )

    df_test_metrics = pd.DataFrame(test_results)
    print(
        f"{'Mô hình':<30} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | {'F1-Macro':>8} | "
        f"{'At-Risk Rec':>11} | {'At-Risk F1':>10}"
    )
    print("-" * 95)
    for _, r in df_test_metrics.iterrows():
        print(
            f"{r['Model']:<30} | {r['Accuracy']:>8.4f} | {r['Precision_Macro']:>9.4f} | "
            f"{r['Recall_Macro']:>8.4f} | {r['F1_Macro']:>8.4f} | "
            f"{r['At_Risk_Recall']:>11.4f} | {r['At_Risk_F1']:>10.4f}"
        )

    df_test_metrics["Composite_Score"] = (
        df_test_metrics["F1_Macro"] * 0.5 + df_test_metrics["At_Risk_Recall"] * 0.5
    )
    best_row = df_test_metrics.sort_values(by=["Composite_Score", "F1_Macro"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_pipeline = fitted_pipelines[best_model_name]

    print("\n" + "*" * 90)
    print(f"MÔ HÌNH ĐƯỢC CHỌN TỐT NHẤT: {best_model_name.upper()}")
    print(f"  + Accuracy       : {best_row['Accuracy']:.4f}")
    print(f"  + F1-Score Macro : {best_row['F1_Macro']:.4f}")
    print(f"  + At-Risk Recall : {best_row['At_Risk_Recall']:.4f} (Nhận diện thành công {best_row['At_Risk_Recall']*100:.1f}% học sinh có nguy cơ)")
    print(f"  + At-Risk F1     : {best_row['At_Risk_F1']:.4f}")
    print("*" * 90)

    best_preds = test_preds_dict[best_model_name]
    print("\n[Chi tiết Báo cáo Phân lớp (Classification Report) của mô hình tốt nhất trên Test]:")
    print(classification_report(y_test, best_preds, target_names=CLASS_NAMES, digits=4))

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), dpi=130)
    class_labels = ["Cần Hỗ Trợ", "Đạt Chuẩn", "Xuất Sắc"]
    for idx, (name, preds) in enumerate(test_preds_dict.items()):
        cm = confusion_matrix(y_test, preds)
        im = axes[idx].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[idx].set_title(f"{name}\nAcc: {test_results[idx]['Accuracy']:.3f} | F1: {test_results[idx]['F1_Macro']:.3f}", fontsize=9, fontweight="bold")
        axes[idx].set_xticks([0, 1, 2])
        axes[idx].set_yticks([0, 1, 2])
        axes[idx].set_xticklabels(class_labels, fontsize=8)
        axes[idx].set_yticklabels(class_labels if idx == 0 else ["", "", ""], fontsize=8)
        axes[idx].set_xlabel("Dự đoán (Predicted)", fontsize=9)
        if idx == 0:
            axes[idx].set_ylabel("Thực tế (Actual)", fontsize=9)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, format(cm[i, j], "d"),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black",
                               fontweight="bold")

    fig.tight_layout()
    cm_plot_path = outdir / "confusion_matrices_all_models.png"
    fig.savefig(cm_plot_path)
    plt.close(fig)

    return best_model_name, best_pipeline, df_test_metrics, {
        "best_row": best_row.to_dict(),
        "cm_plot_path": str(cm_plot_path),
        "test_preds": best_preds,
        "test_probs": test_probs_dict[best_model_name],
    }


def save_artifacts(
    best_name: str,
    best_pipe: Any,
    df_metrics: pd.DataFrame,
    eda_stats: Dict[str, Any],
    eval_info: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    outdir: Path,
) -> None:
    print("\n" + "=" * 90)
    print("BƯỚC 7: LƯU TRỮ PIPELINE MÔ HÌNH VÀ TÀI NGUYÊN TRIỂN KHAI")
    print("=" * 90)

    outdir.mkdir(parents=True, exist_ok=True)
    pipe_path = outdir / "best_classification_pipeline.joblib"
    metrics_path = outdir / "classification_metrics.json"
    schema_path = outdir / "classification_schema.json"
    test_preds_path = outdir / "test_predictions_classification.csv"

    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_pipe,
            "classes": CLASS_NAMES,
            "class_codes": CLASS_CODES,
            "feature_cols": list(X_test.columns),
            "raw_feature_cols": FEATURE_COLS,
        },
        pipe_path,
    )

    schema_payload = {
        "model_name": best_name,
        "target_type": "Multiclass Classification (3 classes)",
        "classes": {
            0: {"name": CLASS_NAMES[0], "range": "Exam_Score < 60", "meaning": "Cần Hỗ Trợ / At-Risk"},
            1: {"name": CLASS_NAMES[1], "range": "60 <= Exam_Score < 80", "meaning": "Đạt Chuẩn / On-Track"},
            2: {"name": CLASS_NAMES[2], "range": "Exam_Score >= 80", "meaning": "Xuất Sắc / High-Performing"},
        },
        "required_features": {
            "Gender": {"type": "categorical", "allowed_values": ["Female", "Male"]},
            "HoursStudied/Week": {"type": "numerical", "min": 0.0, "max": 16.0, "unit": "hours"},
            "Tutoring": {"type": "categorical", "allowed_values": ["Yes", "No"]},
            "Region": {"type": "categorical", "allowed_values": ["Urban", "Rural"]},
            "Attendance(%)": {"type": "numerical", "min": 0.0, "max": 100.0, "unit": "%"},
            "Parent Education": {"type": "categorical", "allowed_values": ["None", "Primary", "Secondary", "Tertiary"]},
        },
        "engineered_features": [
            "Effective_Study_Hours = HoursStudied/Week * (Attendance(%) / 100.0)",
            "Study_Intensity_Ratio = HoursStudied/Week / (Attendance(%) + 1.0)",
        ],
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, ensure_ascii=False, indent=2)

    metrics_payload = {
        "best_model_name": best_name,
        "split_counts": {"train": 2800, "validation": 600, "test": 600},
        "all_models_test_comparison": df_metrics.to_dict(orient="records"),
        "best_model_test_details": eval_info["best_row"],
        "eda_summary": eda_stats,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    df_test_out = X_test.copy()
    df_test_out["Actual_Class_Code"] = y_test.values
    df_test_out["Actual_Class_Name"] = [CLASS_NAMES[i] for i in y_test.values]
    df_test_out["Predicted_Class_Code"] = eval_info["test_preds"]
    df_test_out["Predicted_Class_Name"] = [CLASS_NAMES[i] for i in eval_info["test_preds"]]

    if eval_info["test_probs"] is not None:
        for idx, cname in enumerate(CLASS_NAMES):
            df_test_out[f"Prob_{cname}"] = eval_info["test_probs"][:, idx].round(4)

    df_test_out["Is_Correct"] = df_test_out["Actual_Class_Code"] == df_test_out["Predicted_Class_Code"]
    df_test_out.to_csv(test_preds_path, index=False, encoding="utf-8")

    print(f"- Đã lưu Pipeline mô hình     : {pipe_path.resolve()}")
    print(f"- Đã lưu Schema đầu vào       : {schema_path.resolve()}")
    print(f"- Đã lưu Chỉ số Metrics       : {metrics_path.resolve()}")
    print(f"- Đã lưu Dự đoán Test CSV     : {test_preds_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning Phân Lớp - EduPredict AI")
    parser.add_argument("--data", type=str, default="SAP-4000.csv")
    parser.add_argument("--outdir", type=str, default="classification_outputs")
    args = parser.parse_args()

    set_seed(RANDOM_STATE)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.data)
    if not csv_path.exists():
        fallback = Path(__file__).resolve().parent / "SAP-4000.csv"
        if fallback.exists():
            csv_path = fallback

    df_raw, eda_stats = run_eda(csv_path, outdir)
    df_labeled, y_series = create_classification_labels(df_raw)
    X_train, X_val, X_test, y_train, y_val, y_test, feat_cols = split_data(df_labeled)
    best_name, best_pipe, df_metrics, eval_info = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, outdir
    )
    save_artifacts(best_name, best_pipe, df_metrics, eda_stats, eval_info, X_test, y_test, outdir)


if __name__ == "__main__":
    main()
