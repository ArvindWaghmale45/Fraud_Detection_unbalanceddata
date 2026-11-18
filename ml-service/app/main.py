import json
import os
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask


SERVICE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = SERVICE_DIR / "models" / "random_forest_fraud_model.joblib"
OUTPUTS_DIR = SERVICE_DIR / "outputs"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
DATA_PATH = SERVICE_DIR / "data" / "creditcard_transactions.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
COMPARISON_PATH = OUTPUTS_DIR / "model_comparison.csv"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
LATEST_UPLOAD_METADATA_PATH = UPLOADS_DIR / "latest.json"


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: float = Field(validation_alias=AliasChoices("time", "Time"), ge=0)
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float = Field(validation_alias=AliasChoices("amount", "Amount"), ge=0)

    def feature_row(self) -> list[float]:
        values = {
            "Time": self.time,
            **{f"V{i}": getattr(self, f"v{i}") for i in range(1, 29)},
            "Amount": self.amount,
        }
        return [values[name] for name in FEATURES]


app = FastAPI(title="Fraud Detection ML Service", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Upload-ID",
        "X-Total-Transactions",
        "X-Suspicious-Transactions",
        "X-Genuine-Transactions",
    ],
)

model: Any | None = None
metrics: dict[str, Any] = {}
current_upload_id: str | None = None
current_upload_path: Path | None = None
current_upload_summary: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def load_test_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Saved test predictions not found: {PREDICTIONS_PATH}",
        )
    return pd.read_csv(PREDICTIONS_PATH)


@app.on_event("startup")
def load_artifacts() -> None:
    global model, metrics
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Missing trained model: {MODEL_PATH}")
    if not METRICS_PATH.exists():
        raise RuntimeError(f"Missing model metrics: {METRICS_PATH}")
    model = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    restore_latest_upload()


def restore_latest_upload() -> None:
    global current_upload_id, current_upload_path, current_upload_summary
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not LATEST_UPLOAD_METADATA_PATH.is_file():
        return

    # Migrate older latest-only metadata so it can also appear in upload history.
    metadata = json.loads(LATEST_UPLOAD_METADATA_PATH.read_text(encoding="utf-8"))
    upload_id = metadata.get("upload_id")
    scored_filename = metadata["scored_filename"]
    if Path(scored_filename).name != scored_filename:
        raise RuntimeError("Latest upload metadata contains an invalid output filename")
    if upload_id is None:
        upload_id = Path(scored_filename).stem.removeprefix("scored_")
    if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise RuntimeError("Latest upload metadata contains an invalid upload ID")
    scored_path = UPLOADS_DIR / scored_filename
    if not scored_path.is_file():
        raise RuntimeError(f"Latest scored upload is missing: {scored_path}")
    summary = metadata["summary"]
    summary.setdefault(
        "uploaded_at",
        datetime.fromtimestamp(scored_path.stat().st_mtime, timezone.utc).isoformat(),
    )
    summary.setdefault("upload_id", upload_id)
    upload_metadata_path = UPLOADS_DIR / f"upload_{upload_id}.json"
    if not upload_metadata_path.is_file():
        save_upload_metadata(scored_path, summary, upload_id)
    current_upload_id = upload_id
    current_upload_path = scored_path
    current_upload_summary = summary


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return metrics


@app.get("/feature-importance")
def feature_importance() -> list[dict[str, Any]]:
    path = OUTPUTS_DIR / "feature_importance.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature importance file not found")
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    dataset = load_dataset()
    predictions = load_test_predictions()
    class_counts = dataset["Class"].value_counts()
    hourly = (
        predictions.assign(hour=((predictions["Time"] // 3600) % 24).astype(int))
        .groupby("hour")
        .agg(
            flagged=("Predicted_Class", "sum"),
            fraud=("Actual_Class", "sum"),
            transactions=("Time", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    return {
        "total_transactions": int(len(dataset)),
        "genuine_transactions": int(class_counts.get(0, 0)),
        "fraud_transactions": int(class_counts.get(1, 0)),
        "fraud_rate": float(class_counts.get(1, 0) / len(dataset)),
        "test_transactions": int(len(predictions)),
        "test_flagged": int(predictions["Predicted_Class"].sum()),
        "test_true_positives": int(cm[1][1]),
        "test_false_positives": int(cm[0][1]),
        "test_false_negatives": int(cm[1][0]),
        "threshold": float(metrics["threshold"]),
        "metrics": metrics,
        "hourly": hourly,
    }


@app.get("/api/model-comparison")
def model_comparison() -> list[dict[str, Any]]:
    if not COMPARISON_PATH.is_file():
        raise HTTPException(status_code=404, detail="Model comparison file not found.")
    return pd.read_csv(COMPARISON_PATH).to_dict(orient="records")


@app.get("/api/transactions")
def transactions(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=100),
    status: str = Query(default="ALL", pattern="^(ALL|Genuine|Suspicious)$"),
    search: str = "",
) -> dict[str, Any]:
    frame = load_test_predictions().copy()
    if status != "ALL":
        frame = frame[frame["Prediction_Status"] == status]
    if search:
        frame = frame[
            frame["Time"].astype(str).str.contains(search, case=False)
            | frame["Amount"].astype(str).str.contains(search, case=False)
        ]
    total = int(len(frame))
    start = page * size
    records = frame.iloc[start:start + size].copy()
    records.insert(0, "txn_ref", [f"TEST-{index + 1:06d}" for index in records.index])
    return {
        "items": records.to_dict(orient="records"),
        "page": page,
        "size": size,
        "total": total,
        "pages": (total + size - 1) // size,
    }


@app.get("/api/alerts")
def alerts(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    frame = load_test_predictions()
    flagged = frame[frame["Predicted_Class"] == 1].sort_values(
        "Fraud_Probability",
        ascending=False,
    )
    total = int(len(flagged))
    start = page * size
    records = flagged.iloc[start:start + size].copy()
    records.insert(0, "txn_ref", [f"TEST-{index + 1:06d}" for index in records.index])
    return {
        "items": records.to_dict(orient="records"),
        "page": page,
        "size": size,
        "total": total,
        "pages": (total + size - 1) // size,
    }


def write_upload_metadata_file(path: Path, metadata: dict[str, Any]) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=UPLOADS_DIR,
            delete=False,
        ) as metadata_file:
            temporary_path = metadata_file.name
            json.dump(metadata, metadata_file)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def save_upload_metadata(
    scored_path: Path,
    summary: dict[str, Any],
    upload_id: str,
) -> None:
    record = {
        "upload_id": upload_id,
        "scored_filename": scored_path.name,
        "summary": summary,
    }
    write_upload_metadata_file(UPLOADS_DIR / f"upload_{upload_id}.json", record)
    write_upload_metadata_file(LATEST_UPLOAD_METADATA_PATH, record)


def current_upload_or_404() -> tuple[Path, dict[str, Any]]:
    global current_upload_id, current_upload_path, current_upload_summary
    if (
        current_upload_id is None
        or current_upload_path is None
        or current_upload_summary is None
        or not current_upload_path.is_file()
    ):
        current_upload_id = None
        current_upload_path = None
        current_upload_summary = None
        raise HTTPException(
            status_code=404,
            detail="No dataset has been uploaded yet. Upload a compatible CSV first.",
        )
    return current_upload_path, current_upload_summary


def upload_record_or_404(upload_id: str) -> tuple[Path, dict[str, Any]]:
    if not re.fullmatch(r"[a-f0-9]{32}", upload_id):
        raise HTTPException(status_code=404, detail="Upload history record not found.")
    metadata_path = UPLOADS_DIR / f"upload_{upload_id}.json"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Upload history record not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scored_filename = metadata["scored_filename"]
    if Path(scored_filename).name != scored_filename:
        raise HTTPException(status_code=500, detail="Stored upload metadata is invalid.")
    scored_path = UPLOADS_DIR / scored_filename
    if not scored_path.is_file():
        raise HTTPException(status_code=404, detail="Scored file for this upload is no longer available.")
    return scored_path, metadata["summary"]


@app.get("/api/uploads/current/dashboard")
def uploaded_dashboard() -> dict[str, Any]:
    _, summary = current_upload_or_404()
    return summary


def upload_transactions_page(
    path: Path,
    summary: dict[str, Any],
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=100),
    status: str = "ALL",
) -> dict[str, Any]:
    total = int(summary["total_transactions"])
    if status == "Genuine":
        total = int(summary["genuine_transactions"])
    elif status == "Suspicious":
        total = int(summary["suspicious_transactions"])

    start = page * size
    items = []
    matched = 0
    row_offset = 0
    actual_column = None
    for chunk in pd.read_csv(path, chunksize=10_000):
        columns = {str(column).strip().casefold(): column for column in chunk.columns}
        actual_column = columns.get("class", columns.get("actual_class"))
        mask = (
            chunk["Prediction_Status"].eq(status)
            if status != "ALL"
            else pd.Series(True, index=chunk.index)
        )
        matching_positions = np.flatnonzero(mask.to_numpy())
        for position in matching_positions:
            if matched >= start and len(items) < size:
                row = chunk.iloc[position]
                actual_value = row[actual_column] if actual_column is not None else None
                if actual_value is not None and pd.isna(actual_value):
                    actual_value = None
                elif actual_value is not None:
                    actual_value = int(actual_value)
                items.append(
                    {
                        "txn_ref": f"UPLOAD-{row_offset + position + 1:06d}",
                        "Time": float(row["Time"]),
                        "Amount": float(row["Amount"]),
                        "Fraud_Probability": float(row["Fraud_Probability"]),
                        "Prediction_Status": str(row["Prediction_Status"]),
                        "Actual_Class": actual_value,
                    }
                )
            matched += 1
            if len(items) == size:
                break
        row_offset += len(chunk)
        if len(items) == size:
            break

    return {
        "items": items,
        "page": page,
        "size": size,
        "total": total,
        "pages": (total + size - 1) // size,
    }


@app.get("/api/uploads/current/transactions")
def uploaded_transactions(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=100),
    status: str = Query(default="ALL", pattern="^(ALL|Genuine|Suspicious)$"),
) -> dict[str, Any]:
    path, summary = current_upload_or_404()
    return upload_transactions_page(path, summary, page, size, status)


@app.get("/api/uploads/history")
def upload_history() -> list[dict[str, Any]]:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for metadata_path in UPLOADS_DIR.glob("upload_*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        upload_id = metadata.get("upload_id")
        if not isinstance(upload_id, str) or not re.fullmatch(r"[a-f0-9]{32}", upload_id):
            raise HTTPException(status_code=500, detail="Stored upload metadata has an invalid ID.")
        scored_filename = metadata["scored_filename"]
        if Path(scored_filename).name != scored_filename:
            raise HTTPException(status_code=500, detail="Stored upload metadata is invalid.")
        scored_path = UPLOADS_DIR / scored_filename
        summary = metadata["summary"]
        uploaded_at = summary.get("uploaded_at")
        if not uploaded_at:
            uploaded_at = datetime.fromtimestamp(
                scored_path.stat().st_mtime,
                timezone.utc,
            ).isoformat()
        records.append(
            {
                "upload_id": upload_id,
                "source_filename": summary["source_filename"],
                "uploaded_at": uploaded_at,
                "total_transactions": summary["total_transactions"],
                "suspicious_transactions": summary["suspicious_transactions"],
                "genuine_transactions": summary["genuine_transactions"],
                "suspicious_rate": summary["suspicious_rate"],
                "has_actual_labels": summary["has_actual_labels"],
                "actual_fraud_transactions": summary["actual_fraud_transactions"],
                "download_available": scored_path.is_file(),
            }
        )
    return sorted(records, key=lambda record: record["uploaded_at"], reverse=True)


@app.get("/api/uploads/{upload_id}/dashboard")
def historical_upload_dashboard(upload_id: str) -> dict[str, Any]:
    _, summary = upload_record_or_404(upload_id)
    return summary


@app.get("/api/uploads/{upload_id}/transactions")
def historical_upload_transactions(
    upload_id: str,
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=100),
    status: str = Query(default="ALL", pattern="^(ALL|Genuine|Suspicious)$"),
) -> dict[str, Any]:
    path, summary = upload_record_or_404(upload_id)
    return upload_transactions_page(path, summary, page, size, status)


@app.get("/api/uploads/{upload_id}/download")
def download_scored_upload(upload_id: str) -> FileResponse:
    path, _ = upload_record_or_404(upload_id)
    filename = f"scored_transactions_{upload_id[:8]}.csv"
    return FileResponse(
        path,
        media_type="text/csv",
        filename=filename,
    )


def remove_temporary_files(*paths: str) -> None:
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@app.post("/predict-csv")
async def predict_csv(
    request: Request,
    filename: str = Query(default="Uploaded CSV", max_length=255),
) -> FileResponse:
    global current_upload_id, current_upload_path, current_upload_summary
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_path = None
    output_path: str | None = None
    completed = False
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as uploaded_file:
            input_path = uploaded_file.name

        with open(input_path, "wb") as uploaded_file:
            async for data in request.stream():
                uploaded_file.write(data)

        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        upload_id = uuid4().hex
        output_path = str(UPLOADS_DIR / f"scored_{upload_id}.csv")
        row_count = 0
        flagged_count = 0
        probability_sum = 0.0
        labeled_count = 0
        actual_fraud_count = 0
        confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
        hourly: dict[int, dict[str, int]] = {}
        threshold = float(metrics.get("threshold", 0.4954))
        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            for chunk in pd.read_csv(input_path, chunksize=10_000):
                normalized = {}
                for column in chunk.columns:
                    key = str(column).strip().casefold()
                    if key in normalized:
                        raise HTTPException(
                            status_code=422,
                            detail=f"CSV has duplicate columns after trimming/case normalization: {column}",
                        )
                    normalized[key] = column

                missing = [
                    name for name in FEATURES
                    if name.casefold() not in normalized
                ]
                if missing:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "CSV is missing required model features: "
                            + ", ".join(missing)
                            + ". Required columns are Time, V1–V28, and Amount."
                        ),
                    )

                source_columns = [normalized[name.casefold()] for name in FEATURES]
                actual_column = normalized.get("class", normalized.get("actual_class"))
                feature_values = chunk[source_columns].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                feature_values.columns = FEATURES
                values = feature_values.to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise HTTPException(
                        status_code=422,
                        detail="Required feature columns must contain only finite numeric values.",
                    )
                if (feature_values["Time"] < 0).any():
                    raise HTTPException(status_code=422, detail="Time values must be non-negative.")
                if (feature_values["Amount"] < 0).any():
                    raise HTTPException(status_code=422, detail="Amount values must be non-negative.")

                actual_values = None
                valid_labels = None
                if actual_column is not None:
                    actual_values = pd.to_numeric(chunk[actual_column], errors="coerce")
                    valid_labels = actual_values.notna().to_numpy()
                    invalid_label_text = (
                        chunk[actual_column].notna()
                        & actual_values.isna()
                    ).to_numpy()
                    if invalid_label_text.any():
                        raise HTTPException(
                            status_code=422,
                            detail="Optional Class/Actual_Class values must be numeric 0 or 1.",
                        )
                    invalid_labels = valid_labels & ~actual_values.isin([0, 1]).to_numpy()
                    if invalid_labels.any():
                        raise HTTPException(
                            status_code=422,
                            detail="Optional Class/Actual_Class values must be 0 or 1.",
                        )

                probabilities = model.predict_proba(feature_values)[:, 1]
                predicted = (probabilities >= threshold).astype(int)
                chunk = chunk.rename(
                    columns={
                        source: canonical
                        for canonical, source in zip(FEATURES, source_columns)
                        if source != canonical
                    }
                )
                chunk["Fraud_Probability"] = probabilities
                chunk["Predicted_Class"] = predicted
                chunk["Prediction_Status"] = np.where(
                    predicted == 1,
                    "Suspicious",
                    "Genuine",
                )
                chunk.to_csv(
                    output_file,
                    index=False,
                    header=(row_count == 0),
                )
                flagged_count += int(predicted.sum())
                probability_sum += float(probabilities.sum())

                hours = ((feature_values["Time"] // 3600) % 24).astype(int).to_numpy()
                for hour in np.unique(hours):
                    hour_mask = hours == hour
                    stats = hourly.setdefault(
                        int(hour),
                        {
                            "hour": int(hour),
                            "transactions": 0,
                            "flagged": 0,
                            "actual_fraud": 0,
                            "labeled_transactions": 0,
                        },
                    )
                    stats["transactions"] += int(hour_mask.sum())
                    stats["flagged"] += int(predicted[hour_mask].sum())
                    if actual_values is not None and valid_labels is not None:
                        label_mask = hour_mask & valid_labels
                        labels = actual_values.to_numpy()[label_mask].astype(int)
                        guesses = predicted[label_mask]
                        stats["labeled_transactions"] += int(label_mask.sum())
                        stats["actual_fraud"] += int(labels.sum())
                        labeled_count += int(label_mask.sum())
                        actual_fraud_count += int(labels.sum())
                        confusion["tp"] += int(((labels == 1) & (guesses == 1)).sum())
                        confusion["tn"] += int(((labels == 0) & (guesses == 0)).sum())
                        confusion["fp"] += int(((labels == 0) & (guesses == 1)).sum())
                        confusion["fn"] += int(((labels == 1) & (guesses == 0)).sum())
                row_count += len(chunk)

        if row_count == 0:
            raise HTTPException(status_code=422, detail="CSV contains no transaction rows.")

        evaluation = None
        if labeled_count:
            precision = confusion["tp"] / max(confusion["tp"] + confusion["fp"], 1)
            recall = confusion["tp"] / max(confusion["tp"] + confusion["fn"], 1)
            evaluation = {
                "precision": precision,
                "recall": recall,
                "f1_score": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                ),
                "accuracy": (confusion["tp"] + confusion["tn"]) / labeled_count,
                "confusion_matrix": [
                    [confusion["tn"], confusion["fp"]],
                    [confusion["fn"], confusion["tp"]],
                ],
            }
        source_filename = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
        summary = {
            "upload_id": upload_id,
            "source_filename": source_filename or "Uploaded CSV",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "total_transactions": row_count,
            "suspicious_transactions": flagged_count,
            "genuine_transactions": row_count - flagged_count,
            "suspicious_rate": flagged_count / row_count,
            "average_fraud_probability": probability_sum / row_count,
            "threshold": threshold,
            "has_actual_labels": labeled_count > 0,
            "labeled_transactions": labeled_count,
            "actual_fraud_transactions": actual_fraud_count if labeled_count else None,
            "evaluation": evaluation,
            "hourly": [hourly[hour] for hour in sorted(hourly)],
        }
        scored_path = Path(output_path)
        save_upload_metadata(scored_path, summary, upload_id)
        current_upload_id = upload_id
        current_upload_path = scored_path
        current_upload_summary = summary

        response = FileResponse(
            scored_path,
            media_type="text/csv",
            filename="scored_transactions.csv",
            headers={
                "X-Upload-ID": upload_id,
                "X-Total-Transactions": str(row_count),
                "X-Suspicious-Transactions": str(flagged_count),
                "X-Genuine-Transactions": str(row_count - flagged_count),
            },
            background=BackgroundTask(
                remove_temporary_files,
                input_path,
            ),
        )
        completed = True
        return response
    except pd.errors.EmptyDataError as error:
        raise HTTPException(status_code=422, detail="Upload a non-empty CSV file.") from error
    except pd.errors.ParserError as error:
        raise HTTPException(status_code=422, detail="Could not parse the upload as a CSV file.") from error
    finally:
        if not completed:
            remove_temporary_files(
                *[path for path in (input_path, output_path) if path is not None]
            )


def predict_one(transaction: Transaction) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    probability = float(model.predict_proba([transaction.feature_row()])[0][1])
    threshold = float(metrics.get("threshold", 0.4954))
    return {
        "fraud_probability": probability,
        "prediction": "Suspicious" if probability >= threshold else "Genuine",
        "threshold": threshold,
    }


@app.post("/predict")
def predict(transaction: Transaction) -> dict[str, Any]:
    return predict_one(transaction)


@app.post("/predict-batch")
def predict_batch(transactions: list[Transaction]) -> list[dict[str, Any]]:
    return [predict_one(transaction) for transaction in transactions]
