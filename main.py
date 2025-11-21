import os
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import TNBAdvancedPredictiveEngine

app = FastAPI(title="TNB Predictive Maintenance API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = TNBAdvancedPredictiveEngine()


class Coordinates(BaseModel):
    latitude: float
    longitude: float


class EnvironmentalFactors(BaseModel):
    vegetation_height: float | None = None
    distance_to_line: float | None = None
    elevation: float | None = None
    terrain_type: str | None = None
    humidity: float | None = None
    weather_condition: str | None = None
    temperature: float | None = None
    wind_speed: float | None = None


class TechnicalParameters(BaseModel):
    historical_trip_count: int | None = None
    tower_footing_resistance: float | None = None
    thermal_anomaly: float | None = None
    ultrasound_db: float | None = None
    partial_discharge_pc: float | None = None
    corrosion_level: str | None = None
    structural_integrity: str | None = None


class RiskAssessments(BaseModel):
    vegetation_risk: str | None = None
    defect_status: str | None = None
    weather_impact: str | None = None


class MaintenanceRecord(BaseModel):
    last_maintenance_date: str | None = None
    maintenance_frequency: int | None = None
    repair_history: List[Dict[str, Any]] | None = None
    technician_notes: str | None = None


class PoleData(BaseModel):
    pole_id: str
    coordinates: Coordinates
    environmental_factors: EnvironmentalFactors
    technical_parameters: TechnicalParameters
    risk_assessments: RiskAssessments | None = None
    maintenance_records: MaintenanceRecord | None = None


@app.get("/")
def root():
    return {"message": "TNB Predictive Maintenance Backend running"}


@app.get("/test")
def test_database():
    from database import db
    status = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        import os as _os
        status["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
        status["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
        if db is not None:
            status["database"] = "✅ Available"
            try:
                status["collections"] = db.list_collection_names()[:10]
                status["database"] = "✅ Connected & Working"
                status["connection_status"] = "Connected"
            except Exception as e:
                status["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        return status
    except Exception as e:
        status["database"] = f"❌ Error: {str(e)[:80]}"
        return status


@app.post("/api/v1/analyze-pole")
def analyze_pole(pole: PoleData):
    result = engine.analyze_pole(pole.model_dump())
    return result


class BatchRequest(BaseModel):
    poles: List[PoleData] = Field(default_factory=list)


@app.post("/api/v1/batch-analysis")
def batch_analysis(req: BatchRequest):
    results = []
    for p in req.poles:
        d = p.model_dump()
        res = engine.analyze_pole(d)
        results.append({"pole_id": d.get("pole_id"), **res})
    return {"count": len(results), "results": results}


@app.get("/api/v1/risk-heatmap")
def risk_heatmap():
    # For demo: return a few random points and risk scores
    import random
    points = []
    for i in range(15):
        lat = 6.2 + random.uniform(-0.2, 0.2)
        lon = 100.6 + random.uniform(-0.2, 0.2)
        points.append({"pole_id": f"KBKN-{10+i}", "lat": lat, "lon": lon, "risk": random.randint(20, 95)})
    return {"points": points}


class WorkOrderResources(BaseModel):
    technicians: List[str] | None = None


@app.post("/api/v1/generate-workorders")
def generate_workorders(req: BatchRequest, resources: WorkOrderResources | None = None):
    poles = []
    for p in req.poles:
        d = p.model_dump()
        analysis = engine.analyze_pole(d)
        d["risk_score"] = analysis["risk_analysis"]["overall_risk_score"]
        d["technical_parameters"] = d.get("technical_parameters", {})
        d["environmental_factors"] = d.get("environmental_factors", {})
        poles.append(d)
    res = engine.workflow_automator.generate_optimized_work_orders(
        poles, {"technicians": (resources.technicians if resources else None)}
    )
    return {"count": len(res), "work_orders": res}


@app.put("/api/v1/update-status")
def update_status():
    return {"status": "updated"}


@app.post("/api/v1/retrain-model")
def retrain_model():
    # Simulate retraining
    engine.performance_monitor.update_metrics({"last_retrain": "now"})
    return {"message": "Retraining started", "metrics": engine.performance_monitor.metrics}


@app.get("/api/v1/performance-metrics")
def performance_metrics():
    return engine.performance_monitor.metrics


@app.websocket("/api/v1/realtime-updates")
async def ws_updates(ws: WebSocket):
    await ws.accept()
    import asyncio, random
    try:
        while True:
            payload = {
                "type": "SENSOR_TICK",
                "temperature": round(25 + random.uniform(-1, 6), 2),
                "partial_discharge_pc": round(random.uniform(50, 900), 1),
                "vegetation_height": round(random.uniform(0, 6), 2),
                "timestamp": int(asyncio.get_event_loop().time()*1000),
            }
            await ws.send_json(payload)
            await asyncio.sleep(1.0)
    except Exception:
        await ws.close()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
