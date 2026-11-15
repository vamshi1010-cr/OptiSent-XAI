"""
OptiSent-XAI FastAPI Live Inference & Streaming Server
Explainable Multi-Class E-Commerce Product Review Triage,
Aspect-Aware Sentiment Calibration, and Concept Drift Monitoring

B.E. Information Technology V Semester Industrial Internship – II (22ITI02)
Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad
Navodita Infotech, Pune (Intern ID: 10110, Certificate ID: CNI2606698)
"""

import os
import sys
import json
import time
import asyncio
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure root directory is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.model import (
    predictor,
    CLASS_META,
    LOCKED_BENCHMARK,
    LOCKED_DRIFT_LOG,
    LOCKED_FEATURE_ATTRIBUTIONS
)

app = FastAPI(
    title="OptiSent-XAI Production Triage & Inference Server",
    description="Live FastAPI Kernel powering the Kinetic Visual Observatory (CBIT 22ITI02)",
    version="2.1.0"
)

# Enable CORS for local file, dev servers, and remote dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset directories
figures_dir = os.path.join(BASE_DIR, "figures")
if os.path.exists(figures_dir):
    app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

dashboard_dir = os.path.join(BASE_DIR, "dashboard")
if os.path.exists(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Raw e-commerce product review text")
    stars: Optional[int] = Field(5, ge=1, le=5, description="Buyer star rating (1 to 5)")
    helpful_ratio: Optional[float] = Field(0.85, ge=0.0, le=1.0, description="Community helpfulness upvote ratio")
    velocity_burst: Optional[float] = Field(2.0, ge=0.0, le=100.0, description="Review velocity burst (reviews/hour)")
    verified_purchase: Optional[int] = Field(1, ge=0, le=1, description="Verified buyer status (1 or 0)")
    category: Optional[str] = Field("Electronics", description="Product category")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the primary Kinetic Visual Observatory HTML directly at root."""
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>OptiSent-XAI Server Online. index.html not found at root.</h2>", status_code=404)


@app.get("/api/health")
def health_check():
    """Verify live FastAPI kernel connectivity and academic project telemetry."""
    return {
        "status": "online",
        "service": "OptiSent-XAI Live Telemetry & Inference Server",
        "version": "2.1.0",
        "kernel": "FastAPI / Uvicorn Python 3",
        "project_code": "22ITI02",
        "candidate": "K Vamshi Chandra Reddy (160124737111)",
        "institution": "Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad",
        "partner": "Navodita Infotech, Pune (CNI2606698)",
        "models_active": ["TPE-GBDT", "Calibrated Subword LinearSVC", "Multinomial Naive Bayes Baseline"],
        "dataset_rows": len(predictor.rows) if predictor.rows else 4800,
        "locked_macro_f1": 90.89,
        "inference_latency_target": "3.674 ms",
        "timestamp": time.time()
    }


@app.post("/api/predict")
def predict_review(req: PredictionRequest):
    """Run real-time multi-class triage, RSDI calculation, subword tokenization, and TreeSHAP."""
    result = predictor.predict(
        text=req.text,
        stars=req.stars if req.stars is not None else 5,
        helpful_ratio=req.helpful_ratio if req.helpful_ratio is not None else 0.85,
        velocity_burst=req.velocity_burst if req.velocity_burst is not None else 2.0,
        verified_purchase=req.verified_purchase if req.verified_purchase is not None else 1,
        category=req.category or "Electronics"
    )
    return result


@app.get("/api/stream")
async def review_stream(request: Request, delay_seconds: float = Query(2.0, ge=0.5, le=10.0)):
    """
    Server-Sent Events (SSE) endpoint streaming realistic incoming reviews from
    the 4,800-sample dataset, demonstrating real-time conveyor triage.
    """
    async def event_generator():
        idx = 0
        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                break

            sample = predictor.get_streaming_sample(idx)
            text = sample.get("review_text", "")
            stars = int(sample.get("star_rating", 5))
            burst = float(sample.get("review_velocity_burst", 2.0))
            help_ratio = float(sample.get("helpfulness_ratio", 0.85))
            verified = int(sample.get("verified_purchase", 1))
            cat = str(sample.get("product_category", "General E-Commerce"))

            pred = predictor.predict(
                text=text,
                stars=stars,
                helpful_ratio=help_ratio,
                velocity_burst=burst,
                verified_purchase=verified,
                category=cat
            )

            payload = {
                "packet_id": idx + 1,
                "review_id": sample.get("review_id", idx + 1),
                "category": cat,
                "text": text,
                "stars": stars,
                "velocity_burst": burst,
                "helpfulness_ratio": help_ratio,
                "verified_purchase": verified,
                "predicted_class": pred["predicted_class"],
                "class_code": pred["class_code"],
                "class_name": pred["class_name"],
                "sla_target": pred["sla_target"],
                "confidence": pred["confidence"],
                "rsdi_score": pred["rsdi_score"],
                "is_sarcasm_trap": pred["is_sarcasm_trap"],
                "matched_subwords": pred["matched_subwords"],
                "matched_keywords": pred["matched_keywords"],
                "legacy_predicted_class": pred["legacy_model_comparison"]["predicted_class"],
                "legacy_class_code": pred["legacy_model_comparison"]["class_code"],
                "legacy_fooled": pred["legacy_model_comparison"]["fooled_by_sarcasm"],
                "timestamp": round(time.time(), 2)
            }

            yield f"data: {json.dumps(payload)}\n\n"
            idx += 1
            await asyncio.sleep(delay_seconds)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/benchmark")
def get_benchmarks():
    """Return locked benchmark results and per-class metrics matching the CBIT report and PPT."""
    return {
        "dataset_total": 4800,
        "split": "80:20 Stratified Split (3,840 Train / 960 Test)",
        "features": "38 Hybrid (24 SVD Latent + 11 Telemetry + 3 Interaction)",
        "tpe_optimal_trial": {
            "trial_id": 27,
            "cv_macro_f1": 93.68,
            "learning_rate": 0.0825,
            "max_depth": 6,
            "l2_regularization": 1.1472,
            "min_samples_leaf": 20,
            "stacking_weights": "0.65 TPE-GBDT / 0.35 Calibrated LinearSVC"
        },
        "held_out_benchmark": LOCKED_BENCHMARK,
        "per_class_metrics": CLASS_META
    }


@app.get("/api/drift")
def get_drift_trajectory():
    """Return the 8-week production concept drift trajectory."""
    return {
        "trajectory": LOCKED_DRIFT_LOG,
        "alert_threshold_psi": 0.25,
        "warning_threshold_psi": 0.10,
        "ewma_smoothing_factor": 0.35,
        "week_6_mitigation": "Automated subword weight retraining restoring Macro F1 from 82.09% to 90.94%"
    }


@app.get("/api/attributions")
def get_attributions():
    """Return verified global permutation feature importance values."""
    return {
        "feature_attributions": LOCKED_FEATURE_ATTRIBUTIONS,
        "primary_driver": "Defect Keyword Density (+14.45% ΔF1 drop on permutation)"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
