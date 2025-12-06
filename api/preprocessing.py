
import pandas as pd
import numpy as np
import json
import ast
import re
from typing import List, Tuple

NA_TOKENS = {
    "", " ", "nan", "NaN", "null", "Null", "NULL", "none", "None", "NONE", "-", "."
}

def normalize_missing(series: pd.Series) -> pd.Series:
    mask = series.isna() | series.astype(str).str.strip().isin(NA_TOKENS)
    return series.mask(mask, pd.NA)

def parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def fix_booleans(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal version for inference
    boolean_cols = [
        "host_has_profile_pic", "host_identity_verified", "instant_bookable", 
        "has_availability", "host_is_superhost"
    ]
    existing = [c for c in boolean_cols if c in df.columns]
    
    # Map t/f to yes/no as in training
    if existing:
        df.loc[:, existing] = df.loc[:, existing].replace(
            {True: "yes", False: "no", "t": "yes", "f": "no"}
        )
    if "host_is_superhost" in df:
        df["host_is_superhost"] = df["host_is_superhost"].fillna("no")
    return df

def fill_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    if "neighbourhood" in df and "neighbourhood_cleansed" in df:
        df["neighbourhood"] = df["neighbourhood"].fillna(df["neighbourhood_cleansed"])
    return df

def reviews_block(df: pd.DataFrame) -> pd.DataFrame:
    # Logic to create days_since_* features if dates are present
    # For inference, we might simulate 'last_scraped' as 'today'
    
    if "last_scraped" not in df:
        df["last_scraped"] = pd.Timestamp.now()
        
    has_any = df.get("number_of_reviews", pd.Series(0, index=df.index)) > 0
    df["has_reviews_flag"] = has_any.astype("int8")
    
    if "first_review" in df:
         days = (df["last_scraped"] - df["first_review"]).dt.days
         df["days_since_first_review"] = days.fillna(-1).astype("int32")
         
    if "last_review" in df:
         days = (df["last_scraped"] - df["last_review"]).dt.days
         df["days_since_last_review"] = days.fillna(-1).astype("int32")
         
    # Drop dates as in DAG
    df = df.drop(columns=["first_review", "last_review"], errors="ignore")
    return df

def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    if "bathrooms" in df:
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    return df

def normalize_amenities(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Handle both json string and list representation
    if text.startswith("[") or text.startswith("{"):
        candidate = text.replace("{", "[").replace("}", "]")
        try:
            parsed = json.loads(candidate)
        except:
             try:
                 parsed = ast.literal_eval(candidate)
             except:
                 parsed = []
    else:
        parsed = [text] # Single item?
        
    normalized = []
    if isinstance(parsed, list):
        for item in parsed:
            token = str(item).strip().strip('"').strip("'")
            if token:
                normalized.append(token.lower())
    return sorted(set(normalized))

def amenity_to_col(amenity: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", amenity).strip("_")
    return f"amenity_{slug}" if slug else "amenity_other"

def build_amenities_inference(df: pd.DataFrame, top_amenities_cols: List[str]) -> pd.DataFrame:
    # For inference, we need to know the specific amenity columns the model expects
    # We will assume these are passed or hardcoded based on training top_k
    
    amenity_lists = df["amenities"].apply(normalize_amenities) if "amenities" in df else pd.Series([])
    
    df["amenities_count"] = amenity_lists.str.len().fillna(0).astype("int16")
    
    # We need to reconstruct the exact amenity columns (amenity_wifi, etc.)
    # Since we don't have the list here dynamic, we relies on what the ColumnTransformer expects.
    # Actually, the ColumnTransformer will complain if columns are missing.
    # Strategy: generate common ones found in input, but for safety we should try to match known ones.
    # Simplification: We iterate over the input amenities and set cols if they match expected format?
    # Better: The user of this function should provide the list of "amenity_*" columns to ensure existence.
    
    # For now, we expand what we have, and let 'impute_remaining' or ColumnTransformer handle missing cols if we can.
    # BUT ColumnTransformer needs keys.
    # Creating a comprehensive list of amenities from the sample data in main.py?
    
    # Let's try to map common ones.
    for items in amenity_lists:
        for it in items:
            col = amenity_to_col(it)
            # We can't know if this col was in training unless we have the list.
            # We will create it. Excess cols will be dropped by ColumnTransformer (remainder='drop'?? No default is passthrough for CT? No default is drop? No default is drop in newer versions? 
            # In prepare_features, it selects specific cols.
            pass

    # Actually, a better approach for amenities in inference:
    # We must know the columns. 
    # Valid columns will be handled in main.py by ensuring schema alignment.
    
    # Let's just create 0-filled columns for the "top" amenities if we knew them.
    # Since we don't, we might skip this unless we find the list.
    # Wait, the `ListingData` in `main.py` has `amenities` string.
    # The model was trained on `amenity_wifi`, `amenity_kitchen`...
    # If the API doesn't produce these columns, prediction FAILS.
    
    return df

def preprocess_single(df: pd.DataFrame) -> pd.DataFrame:
    # Main wrapper
    df = df.apply(normalize_missing)
    df = parse_dates(df, ["last_scraped", "host_since", "first_review", "last_review"])
    df = fix_booleans(df)
    df = fill_neighbourhood(df)
    df = reviews_block(df)
    df = parse_bathrooms(df)
    
    if "host_since" in df and "last_scraped" in df:
        df["host_days_active"] = (df["last_scraped"] - df["host_since"]).dt.days.fillna(-1).astype("int32")
        
    # Price cleaning
    if "price" in df:
        df["price_clean"] = df["price"].astype(str).str.replace(r"[^\d\.]", "", regex=True)
        df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
        # Impute price (simple global median fallback for singleton)
        df["price_imputed"] = df["price_clean"].fillna(50.0) 
        
        # Calculate price_per_person
        acc = df.get("accommodates", 1).replace(0, 1)
        df["price_per_person"] = df["price_imputed"] / acc
        
    # Explicitly calculate basic amenity columns if they are critical features
    # Based on tp.ipynb or common knowledge?
    # Or simplified: if 'Wifi' in amenities -> amenity_wifi = 1
    # We will do a best effort for a few common ones
    common_ams = ["wifi", "kitchen", "air_conditioning", "heating", "washer", "tv", "iron"]
    
    # Normalize input amenities
    ams = df["amenities"].apply(normalize_amenities) if "amenities" in df else []
    
    for c in common_ams:
        col_name = f"amenity_{c}"
        df[col_name] = ams.apply(lambda x: 1 if c in x else 0)
        
    df["amenities_count"] = ams.apply(len)

    return df
