"""DAG que replica el pipeline de limpieza de la notebook tp.ipynb.

Pasos clave (resumen):
- Carga listings_big.csv con normalización de NAs.
- Dropea columnas irrelevantes y las de >95% nulos.
- Completa booleanos (has_availability, host_is_superhost) y fechas.
- Normaliza reviews y crea días desde primeras/últimas reviews.
- Imputa price (medianas por vecindario/room_type), crea price_per_person.
- Normaliza bathrooms/bedrooms/beds con modas por room_type.
- Codifica amenities (top 20) y amenity_count.
- Define target occupancy_level (bins de estimated_occupancy_l365d).
- Quita columnas con fuga (availability, estimated_occupancy, reviews*).
- Split con GroupShuffle por host_id (si existe) y guarda train/test.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from airflow.decorators import dag, task
from pendulum import datetime
from sklearn.model_selection import GroupShuffleSplit, train_test_split


RAW_DATA_URI = os.getenv("LISTINGS_S3_URI", "s3://listings/listings_big.csv")
S3_STORAGE_OPTIONS = {
    "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000")}
}
PROCESSED_DIR = Path("/opt/airflow/data/processed")
PROCESSED_S3_PREFIX = os.getenv("PROCESSED_S3_PREFIX", "s3://listings/processed")
TARGET = "occupancy_level"
NA_TOKENS = {"", " ", "nan", "NaN", "null", "Null", "NULL", "none", "None", "NONE", "-", "."}

DROP_COLS = [
    "id",
    "listing_url",
    "scrape_id",
    "name",
    "description",
    "picture_url",
    "host_url",
    "host_name",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_verifications",
]

HIGH_NULL_COLS = ["neighbourhood_group_cleansed", "calendar_updated", "license"]


def normalize_missing(series: pd.Series) -> pd.Series:
    mask = series.isna() | series.astype(str).str.strip().isin(NA_TOKENS)
    return series.mask(mask, pd.NA)


def parse_dates(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fix_booleans(df: pd.DataFrame) -> pd.DataFrame:
    availability_cols = [c for c in ["availability_30", "availability_60", "availability_90", "availability_365"] if c in df]
    availability_sum = df[availability_cols].sum(axis=1) if availability_cols else pd.Series(index=df.index, dtype=float)
    has_availability_isna = df["has_availability"].isna() if "has_availability" in df else pd.Series(False, index=df.index)
    df.loc[has_availability_isna & (availability_sum != 0), "has_availability"] = "t"
    df.loc[has_availability_isna & (availability_sum == 0), "has_availability"] = "f"

    df["was_evaluated_for_superhost"] = ~df.get("host_is_superhost", pd.Series(index=df.index)).isna()

    boolean_cols = [
        c
        for c in [
            "host_has_profile_pic",
            "host_identity_verified",
            "instant_bookable",
            "has_availability",
            "host_is_superhost",
        ]
        if c in df
    ]
    if boolean_cols:
        df.loc[:, boolean_cols] = df.loc[:, boolean_cols].replace({True: "yes", False: "no", "t": "yes", "f": "no"})
    if "host_is_superhost" in df:
        df["host_is_superhost"] = df["host_is_superhost"].fillna("no")
    return df


def fill_neighbourhood(df: pd.DataFrame) -> pd.DataFrame:
    if "neighbourhood" in df and "neighbourhood_cleansed" in df:
        df["neighbourhood"] = df["neighbourhood"].fillna(df["neighbourhood_cleansed"])
    cols = [c for c in ["neighborhood_overview", "host_neighbourhood"] if c in df]
    if cols:
        df[cols] = (
            df[cols]
            .astype("string")
            .replace(r"^\s*$", pd.NA, regex=True)
            .fillna("not-defined")
        )
    return df


def reviews_block(df: pd.DataFrame) -> pd.DataFrame:
    review_score_columns = [c for c in df.columns if c.startswith("review_scores_")]
    df[review_score_columns] = df[review_score_columns].apply(pd.to_numeric, errors="coerce")

    has_any_review_count = df.get("number_of_reviews", pd.Series(0, index=df.index)) > 0
    has_first_review_date = df.get("first_review", pd.Series(index=df.index)).notna()
    has_last_review_date = df.get("last_review", pd.Series(index=df.index)).notna()
    has_reviews_per_month = (df.get("reviews_per_month", pd.Series(0, index=df.index)).fillna(0) > 0)

    df["has_reviews_flag"] = (
        has_any_review_count | has_first_review_date | has_last_review_date | has_reviews_per_month
    ).astype("int8")

    if review_score_columns:
        df.loc[df["has_reviews_flag"] == 0, review_score_columns] = 0.0

    if "last_scraped" in df:
        days_since_first_review = (df["last_scraped"] - df.get("first_review", pd.NaT)).dt.days
        df["days_since_first_review"] = (
            days_since_first_review.where(df["has_reviews_flag"] == 1, -1)
            .fillna(-1)
            .clip(lower=-1)
            .astype("int32")
        )
        days_since_last_review = (df["last_scraped"] - df.get("last_review", pd.NaT)).dt.days
        df["days_since_last_review"] = (
            days_since_last_review.where(df["has_reviews_flag"] == 1, -1)
            .fillna(-1)
            .clip(lower=-1)
            .astype("int32")
        )

    if "reviews_per_month" in df:
        df.loc[(df["has_reviews_flag"] == 0) & (df["reviews_per_month"].isna()), "reviews_per_month"] = 0.0
    df = df.drop(columns=[c for c in ["first_review", "last_review"] if c in df], errors="ignore")
    return df


def parse_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    if "bathrooms" in df:
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
    if "bathrooms_text" in df:
        from_text = df["bathrooms_text"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
        df["bathrooms_text_num"] = pd.to_numeric(from_text, errors="coerce")
        if "bathrooms" in df:
            df["bathrooms"].fillna(df["bathrooms_text_num"], inplace=True)
    return df


def mode_impute_by(df: pd.DataFrame, target: str, ref: str) -> pd.DataFrame:
    if ref not in df or target not in df:
        return df
    ref_no_na = df[ref].dropna()
    if ref_no_na.empty:
        df[target] = df[target].fillna(df[target].median())
        return df
    moda = df.dropna(subset=[target]).groupby(ref)[target].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    missing_idx = df[df[target].isna()].index
    df.loc[missing_idx, target] = df.loc[missing_idx, ref].map(moda)
    if df[target].isna().any():
        df[target] = df[target].fillna(df[target].median())
    return df


def price_block(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def to_numeric_price(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.replace(r"[^\d\.]", "", regex=True).replace("", pd.NA)
        return pd.to_numeric(cleaned, errors="coerce")

    for d in (df_train, df_test):
        d["price_clean"] = to_numeric_price(normalize_missing(d["price"]))
        d["price_was_missing"] = d["price_clean"].isna().astype("int8")

    global_price_median = df_train["price_clean"].median()
    median_price_by_group = (
        df_train.groupby(["neighbourhood_cleansed", "room_type"])["price_clean"].median().dropna()
        if {"neighbourhood_cleansed", "room_type"}.issubset(df_train.columns)
        else pd.Series(dtype=float)
    )
    median_price_by_neigh = (
        df_train.groupby("neighbourhood_cleansed")["price_clean"].median().dropna()
        if "neighbourhood_cleansed" in df_train
        else pd.Series(dtype=float)
    )

    def impute_price(df: pd.DataFrame) -> pd.Series:
        price = df["price_clean"].copy()
        if not median_price_by_group.empty:
            group_key = list(zip(df.get("neighbourhood_cleansed", ""), df.get("room_type", "")))
            price = price.fillna(pd.Series(group_key, index=df.index).map(median_price_by_group))
        if not median_price_by_neigh.empty:
            price = price.fillna(df.get("neighbourhood_cleansed", pd.Series(index=df.index)).map(median_price_by_neigh))
        price = price.fillna(global_price_median)
        return price

    for d in (df_train, df_test):
        d["price_imputed"] = impute_price(d)
        accommodates = d.get("accommodates", pd.Series(1, index=d.index)).replace(0, 1).fillna(1)
        d["price_per_person"] = (d["price_imputed"] / accommodates).replace([pd.NA, np.inf, -np.inf], np.nan)
        d["price_per_person"] = d["price_per_person"].fillna(d["price_per_person"].median())
        d.drop(columns=[c for c in ["price_clean"] if c in d], inplace=True)
    return df_train, df_test


def normalize_amenities(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    candidate = text.replace("{", "[").replace("}", "]")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            parsed = [item.strip() for item in candidate.strip("[]").split(",")]
    normalized = []
    for item in parsed:
        token = str(item).strip().strip('"').strip("'")
        if token:
            normalized.append(token.lower())
    return sorted(set(normalized))


def amenity_to_col(amenity: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", amenity).strip("_")
    return f"amenity_{slug}" if slug else "amenity_other"


def build_amenities(df_train: pd.DataFrame, df_test: pd.DataFrame, top_k: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    amenity_lists_train = df_train["amenities"].apply(normalize_amenities) if "amenities" in df_train else pd.Series([], dtype=object)
    amenity_counts = amenity_lists_train.explode().value_counts()
    top_amenities = amenity_counts.head(top_k)

    df_train["amenities_count"] = amenity_lists_train.str.len().astype("int16") if not amenity_lists_train.empty else 0
    amenities_feature_cols: List[str] = []
    for amenity in top_amenities.index:
        base = amenity_to_col(amenity)
        name = base
        suffix = 1
        while name in amenities_feature_cols:
            suffix += 1
            name = f"{base}_{suffix}"
        amenities_feature_cols.append(name)
        df_train[name] = amenity_lists_train.apply(lambda items, t=amenity: int(t in items)).astype("int8")

    amenity_lists_test = df_test["amenities"].apply(normalize_amenities) if "amenities" in df_test else pd.Series([], dtype=object)
    if not amenity_lists_test.empty:
        df_test["amenities_count"] = amenity_lists_test.str.len().astype("int16")
        for amenity, col_name in zip(top_amenities.index, amenities_feature_cols):
            df_test[col_name] = amenity_lists_test.apply(lambda items, t=amenity: int(t in items)).astype("int8")
    for col in amenities_feature_cols:
        if col not in df_test:
            df_test[col] = 0
        df_train[col] = df_train[col].astype("int8")
        df_test[col] = df_test[col].astype("int8")

    df_train.drop(columns=[c for c in ["amenities"] if c in df_train], inplace=True)
    df_test.drop(columns=[c for c in ["amenities"] if c in df_test], inplace=True)
    return df_train, df_test, amenities_feature_cols


def build_target(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for d in (df_train, df_test):
        occ = d.get("estimated_occupancy_l365d", pd.Series(index=d.index)).clip(lower=0, upper=365)
        bins = [-1, 0, 30, 120, 366]
        labels = ["zero", "low", "mid", "high"]
        d[TARGET] = pd.cut(occ, bins=bins, labels=labels, right=True, include_lowest=True)
    return df_train, df_test


def strip_leakage(df: pd.DataFrame) -> pd.DataFrame:
    leak = [c for c in df.columns if c.startswith("availability_")] + [
        "has_availability",
        "availability_eoy",
        "calendar_last_scraped",
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "occ_rate",
    ]
    review = [
        c
        for c in df.columns
        if (
            c.startswith("number_of_reviews")
            or c.startswith("review_scores_")
            or c
            in [
                "reviews_per_month",
                "has_reviews",
                "days_since_first_review",
                "days_since_last_review",
                "reviews_per_year",
                "reviews_per_month_filled",
                "has_reviews_flag",
            ]
        )
    ]
    drop = [c for c in leak + review if c in df.columns]
    return df.drop(columns=drop, errors="ignore")


def impute_remaining(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = df_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    if TARGET in cat_cols:
        cat_cols.remove(TARGET)

    for col in numeric_cols:
        median = df_train[col].median()
        df_train[col] = df_train[col].fillna(median)
        df_test[col] = df_test[col].fillna(median)

    for col in cat_cols:
        mode = df_train[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        df_train[col] = df_train[col].fillna(fill_value)
        df_test[col] = df_test[col].fillna(fill_value)
    return df_train, df_test


@dag(schedule=None, start_date=datetime(2025, 1, 1), catchup=False, tags=["preprocess", "listings", "mlops"])
def data_treatment_dag():
    @task()
    def preprocess() -> Dict[str, str]:
        """Run the data cleaning pipeline and persist train/test splits."""
        if RAW_DATA_URI.startswith("s3://"):
            df = pd.read_csv(RAW_DATA_URI, na_values=list(NA_TOKENS), storage_options=S3_STORAGE_OPTIONS)
        else:
            path = Path(RAW_DATA_URI)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found at {RAW_DATA_URI}")
            df = pd.read_csv(path, na_values=list(NA_TOKENS))
        df = df.drop_duplicates(subset="id", keep="first")
        df = df.drop(columns=[c for c in DROP_COLS if c in df], errors="ignore")
        df = df.drop(columns=[c for c in HIGH_NULL_COLS if c in df], errors="ignore")

        df = parse_dates(df, ["last_scraped", "host_since", "first_review", "last_review"])
        df = fix_booleans(df)
        df = fill_neighbourhood(df)
        df = reviews_block(df)
        df = parse_bathrooms(df)
        df = mode_impute_by(df, target="bedrooms", ref="room_type")
        df = mode_impute_by(df, target="beds", ref="room_type")

        if "host_since" in df and "last_scraped" in df:
            df["host_days_active"] = (df["last_scraped"] - df["host_since"]).dt.days.fillna(-1).astype("int32")
        if "host_id" in df:
            df["host_id"] = df["host_id"].fillna(-1)

        # Split evitando fuga por host
        if "host_id" in df:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(df, groups=df["host_id"]))
            df_train, df_test = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
        else:
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        df_train, df_test = price_block(df_train, df_test)
        df_train, df_test, _ = build_amenities(df_train, df_test, top_k=20)

        df_train, df_test = build_target(df_train, df_test)
        df_train, df_test = strip_leakage(df_train), strip_leakage(df_test)
        df_train, df_test = impute_remaining(df_train, df_test)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        train_path = PROCESSED_DIR / "train.csv"
        test_path = PROCESSED_DIR / "test.csv"
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)

        # También almacenamos en MinIO/S3 para consumo por otros servicios
        if PROCESSED_S3_PREFIX:
            s3_train = f"{PROCESSED_S3_PREFIX.rstrip('/')}/train.csv"
            s3_test = f"{PROCESSED_S3_PREFIX.rstrip('/')}/test.csv"
            df_train.to_csv(s3_train, index=False, storage_options=S3_STORAGE_OPTIONS)
            df_test.to_csv(s3_test, index=False, storage_options=S3_STORAGE_OPTIONS)

        return {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "rows_train": len(df_train),
            "rows_test": len(df_test),
            "columns": df_train.columns.tolist(),
        }

    preprocess()


dag = data_treatment_dag()
