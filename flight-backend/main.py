from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import polars as pl
import json
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from pipeline import preprocess_flight_data

# Global variables to hold models and data in RAM
model = None
carrier0_map = None
carrier1_map = None
rt_freq_map = None
cat_mappings = None
import zipfile # <-- Add this new import at the top of the file

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    global model, carrier0_map, carrier1_map, rt_freq_map, cat_mappings
    print("Loading assets into memory...")
    
    try:
        # --- NEW: Unzip the model before loading ---
        print("Unzipping model...")
        with zipfile.ZipFile("assets/model.zip", 'r') as zip_ref:
            zip_ref.extractall("assets")
        model = xgb.Booster()
        model.load_model("assets/model.ubj")       
        # Load Polars mappings AND force the join keys to be standard Strings
        carrier0_map = pl.read_parquet("assets/carrier0_pop.parquet").with_columns(
            pl.col("legs0_segments0_marketingCarrier_code").cast(pl.String)
        )
        
        carrier1_map = pl.read_parquet("assets/carrier1_pop.parquet").with_columns(
            pl.col("legs1_segments0_marketingCarrier_code").cast(pl.String)
        )
        
        rt_freq_map = pl.read_parquet("assets/round_trip_freq.parquet").with_columns(
            pl.col("round_trip_route").cast(pl.String)
        )
        
        with open("assets/cat_mappings.json", "r") as f:
            cat_mappings = json.load(f)
            
        print("Assets loaded successfully!")
    except Exception as e:
        print(f"Failed to load assets: {e}")
        
    yield
    
    # --- Shutdown Logic ---
    print("Shutting down API and clearing memory...")
    model = None
    carrier0_map = None
    carrier1_map = None
    rt_freq_map = None
    cat_mappings = None

# Pass the lifespan function into the FastAPI app
app = FastAPI(title="Flight Recommendation API", lifespan=lifespan)

# --- ADD THIS BLOCK TO ALLOW REACT TO TALK TO FASTAPI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you'd put your Vercel URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------------

# Define the expected JSON payload from the React frontend
class FlightSearchRequest(BaseModel):
    ranker_id: int
    flights: List[Dict[str, Any]]

@app.post("/recommend")
def get_recommendations(request: FlightSearchRequest):
    try:
        if not request.flights:
            raise HTTPException(status_code=400, detail="No flights provided")
        
        # 1. Convert incoming JSON to Polars DataFrame
        df_raw = pl.DataFrame(request.flights)
        
        # --- THE NUKE: Force all text to standard String instantly ---
        string_cols = [c for c, t in zip(df_raw.columns, df_raw.dtypes) if "String" in str(t) or "Utf8" in str(t)]
        if string_cols:
            df_raw = df_raw.with_columns([pl.col(c).cast(pl.String) for c in string_cols])
        # -------------------------------------------------------------
        
        # 2. Add ranker_id for group-based features
        if "ranker_id" not in df_raw.columns:
            df_raw = df_raw.with_columns(pl.lit(request.ranker_id).alias("ranker_id"))
            
        # 3. Run feature engineering pipeline
        df_processed = preprocess_flight_data(
            df_raw, 
            carrier0_map, 
            carrier1_map, 
            rt_freq_map,
            cat_mappings
        )
            
        original_ids = df_processed["Id"].to_list() if "Id" in df_processed.columns else list(range(len(df_processed)))
        
        # 4. Filter columns to exactly what XGBoost expects
        feature_cols = model.feature_names
        
        missing_from_df = [col for col in feature_cols if col not in df_processed.columns]
        if missing_from_df:
            raise ValueError(f"Missing columns expected by XGBoost: {missing_from_df}")
            
        X = df_processed.select(feature_cols)
        
        # --- THE FINAL SHIELD: XGBoost only speaks Math ---
        # Any string columns that weren't encoded in your cat_features list 
        # (like weightMeasurementType) will crash XGBoost. We convert them 
        # to a default numeric value (-1.0) right before inference.
        string_cols = [c for c, t in zip(X.columns, X.dtypes) if "String" in str(t) or "Utf8" in str(t)]
        if string_cols:
            X = X.with_columns([pl.lit(-1.0).alias(c) for c in string_cols])
        # --------------------------------------------------
        
        # 5. Predict and generate scores
        dmatrix = xgb.DMatrix(X)
        scores = model.predict(dmatrix)
        
        # 6. Pair scores with IDs and sort
        results = [{"Id": fid, "score": float(score)} for fid, score in zip(original_ids, scores)]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "ranker_id": request.ranker_id, 
            "ranked_flights": results
        }
        
    except Exception as e:
        # This will now catch XGBoost errors and send them to your test script
        raise HTTPException(status_code=500, detail=str(e))