
import os
import mlflow
import pandas as pd
import joblib
import preprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from mlflow.tracking import MlflowClient

app = FastAPI(
    title="Airbnb Occupancy Prediction API",
    description="API for predicting Airbnb occupancy levels based on listing features.",
    version="1.0.0"
)

# Global variables
model = None
preprocessor = None

# Pydantic models (same as before)
class ListingData(BaseModel):
    # Host related
    host_since: str = Field("2020-01-01", description="Date host joined")
    host_response_time: str = Field("within an hour", description="Host response time")
    host_response_rate: str = Field("100%", description="Host response rate")
    host_acceptance_rate: str = Field("95%", description="Host acceptance rate")
    host_is_superhost: str = Field("t", description="Is superhost (t/f)")
    host_listings_count: int = Field(1, description="Total listings count")
    host_total_listings_count: int = Field(1, description="Total listings count (all)")
    host_has_profile_pic: str = Field("t", description="Has profile pic (t/f)")
    host_identity_verified: str = Field("t", description="Identity verified (t/f)")
    
    # Location
    neighbourhood_cleansed: str = Field("Palermo", description="Neighbourhood name")
    latitude: float = Field(-34.58, description="Latitude")
    longitude: float = Field(-58.42, description="Longitude")
    
    # Property details
    property_type: str = Field("Entire rental unit", description="Type of property")
    room_type: str = Field("Entire home/apt", description="Type of room")
    accommodates: int = Field(2, description="Max guests")
    bathrooms: float = Field(1.0, description="Number of bathrooms")
    bedrooms: float = Field(1.0, description="Number of bedrooms")
    beds: float = Field(1.0, description="Number of beds")
    
    # Booking settings
    price: str = Field("$50.00", description="Price string")
    minimum_nights: int = Field(2, description="Min nights")
    maximum_nights: int = Field(30, description="Max nights")
    minimum_minimum_nights: int = Field(2, description="Min min nights")
    maximum_minimum_nights: int = Field(2, description="Max min nights")
    minimum_maximum_nights: int = Field(1125, description="Min max nights")
    maximum_maximum_nights: int = Field(1125, description="Max max nights")
    minimum_nights_avg_ntm: float = Field(2.0, description="Avg min nights")
    maximum_nights_avg_ntm: float = Field(1125.0, description="Avg max nights")
    
    # Availability
    has_availability: str = Field("t", description="Has availability (t/f)")
    availability_30: int = Field(10, description="Availability next 30 days")
    availability_60: int = Field(20, description="Availability next 60 days")
    availability_90: int = Field(30, description="Availability next 90 days")
    availability_365: int = Field(100, description="Availability next 365 days")
    
    # Reviews
    number_of_reviews: int = Field(10, description="Total reviews")
    number_of_reviews_ltm: int = Field(2, description="Reviews last 12 months")
    number_of_reviews_l30d: int = Field(0, description="Reviews last 30 days")
    review_scores_rating: float = Field(4.8, description="Overall rating")
    review_scores_accuracy: float = Field(4.9, description="Accuracy score")
    review_scores_cleanliness: float = Field(4.8, description="Cleanliness score")
    review_scores_checkin: float = Field(4.9, description="Checkin score")
    review_scores_communication: float = Field(4.9, description="Communication score")
    review_scores_location: float = Field(4.9, description="Location score")
    review_scores_value: float = Field(4.7, description="Value score")
    reviews_per_month: float = Field(0.5, description="Reviews per month")
    
    # Amenities (JSON string as in raw CSV)
    amenities: str = Field("[\"Wifi\", \"Kitchen\", \"Air conditioning\"]", description="JSON list of amenities")


class PredictionRequest(BaseModel):
    data: List[ListingData]

class PredictionResponse(BaseModel):
    predictions: List[str]

@app.on_event("startup")
def load_resources():
    global model, preprocessor
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        model_name = os.getenv("MODEL_NAME", "airbnb-occupancy-classifier")
        
        client = MlflowClient()
        
        # 1. Get Production Model Version and Run ID
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            print("No Production model found.")
            return
            
        run_id = prod_versions[0].run_id
        print(f"Found Production model (Run ID: {run_id})")
        
        # 2. Download Preprocessor Artifact
        # Try to download 'preprocessor/preprocessor.joblib'
        try:
            local_path = client.download_artifacts(run_id, "preprocessor/preprocessor.joblib", dst_path="/tmp")
            preprocessor = joblib.load(local_path)
            print("Preprocessor loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load preprocessor artifact: {e}")
            print("Assuming model pipeline handles raw data or simple mapping will be used.")
            preprocessor = None

        # 3. Load Model
        model_uri = f"models:/{model_name}/Production"
        print(f"Loading model from {model_uri}...")
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
        
    except Exception as e:
        print(f"Error loading resources: {e}")

@app.post("/reload-model")
def reload_model():
    """Force reload of model and preprocessor from Production stage."""
    load_resources()
    return {
        "status": "reloaded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 1. Convert to DataFrame
        data_dicts = [item.dict() for item in request.data]
        df = pd.DataFrame(data_dicts)
        
        # 2. Apply Custom Preprocessing (Feature Engineering)
        df_processed = preprocessing.preprocess_single(df)
        
        # 3. Apply ColumnTransformer (Scaling/Encoding)
        if preprocessor:
            # We need to ensure columns match what preprocessor expects
            # Get expected features from preprocessor
            if hasattr(preprocessor, "feature_names_in_"):
                expected_cols = preprocessor.feature_names_in_
                # Add missing cols with default 0/NaN
                for col in expected_cols:
                    if col not in df_processed.columns:
                        # If numeric, 0? If categorical, 'unknown'?
                        # Ideally median/mode, but 0 is safe for inference usually
                        df_processed[col] = 0
                
                # Drop extra cols
                df_processed = df_processed[expected_cols]
            
            # Transform
            X_final = preprocessor.transform(df_processed)
        else:
            # Fallback (risky)
            X_final = df_processed
            
        # 4. Predict
        predictions = model.predict(X_final)
        
        return {"predictions": [str(p) for p in predictions]}
    except Exception as e:
        # Log error for debug
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }
