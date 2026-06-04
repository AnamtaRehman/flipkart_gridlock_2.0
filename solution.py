"""
Spatial V3 traffic demand solution, cleaned up and split into small pieces.

This is the version I would keep as the final project code. It uses only:
- train.csv
- test.csv

It does not read the answer key. No hidden labels are used for fitting models,
choosing weights, or writing the final submission.

Main output:
- submission_spatial_v3_recommended_blend_clean.csv

The model idea is simple:
1. Start with day 48 demand at the same geohash and time.
2. Use the known part of day 49, 00:00 to 02:00, to estimate how today shifted.
3. Borrow signal from nearby geohashes, because traffic does not stop at one cell.
4. Train residual models to correct the day-48 backbone.
5. Blend a few different models so one model's bad habits do not dominate.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Keep local runs predictable. These limits also stop tree models from using
# every CPU thread on shared notebooks.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

try:
    import lightgbm as lgb
except Exception:  # LightGBM is useful, but the script should still run without it.
    lgb = None

try:
    from catboost import CatBoostRegressor
except Exception as exc:
    raise ImportError("CatBoost is needed for the V3 blend. Install catboost first.") from exc


# These are the filenames used by the final pipeline. The previous baseline is
# disabled in this fresh-data version. Old submissions from another test file should not be reused.
BASELINE_FILE = None  # Fresh-data mode: do not reuse an old submission from another test file.
OUTPUT_FILE = "submission_spatial_v3_recommended_blend_clean.csv"

# These are the intermediate prediction files from the V3 family. When they are
# already present, we reuse them. That makes day-to-day reruns fast while keeping
# the training functions below available for a fresh run.
SPATIAL_BACKBONE_FILE = "submission_spatial_global_blend_12best_50spatial_38raw.csv"
CATBOOST_DIRECT_FILE = "submission_cat_direct_all_fast.csv"
EXTRA_LGB_FILE = "submission_spatial_lgb_residual_fast2.csv"
EXTRA_HGB_FILE = "submission_spatial_hgb_residual_fast.csv"

# Set this to False if you want to force every model to retrain from scratch.
USE_CACHED_INTERMEDIATE_PREDICTIONS = False


# Geohash uses this base-32 alphabet. We decode it ourselves to avoid depending
# on a package that may not be installed in the judging environment.
_GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
_GEOHASH_LOOKUP = {char: idx for idx, char in enumerate(_GEOHASH_ALPHABET)}


# The spatial model works with numeric columns only. Categorical road/weather
# columns are mapped into simple numeric flags for the residual models.
SPATIAL_FEATURES = [
    "d48gt", "d48roll", "d48roll3", "d48roll5", "d48roll7",
    "offset_shrunk", "anchor", "anch48", "anchor_diff", "gap",
    "offset_slope", "early_trend_projected",
    "g_mean", "g_std", "g_min", "g_max", "g_median", "g_count",
    "lat", "lon", "tmin", "sin_t", "cos_t", "hour",
    "lanes", "lv", "lm", "rt", "wx",
    "Temperature", "temp_missing", "temp_filled",
    "gh4_mean", "gh4_time_mean", "gh5_mean", "gh5_time_mean",
    "n8_d48_mean", "n8_d48_idw", "n8_d48_std", "n8_d48_max", "n8_d48_min",
    "n8_d48_roll_idw", "n8_d48_avail_frac",
    "n8_offset_mean", "n8_offset_idw", "n8_offset_std",
    "n8_anchor_idw", "n8_anch48_idw", "n8_anchor_diff_idw", "n8_slope_idw",
    "base_self", "base_spatial",
]

# For the raw day-48 model, we deliberately remove the columns that would make
# the model memorize day-48 demand directly. It still sees location, road,
# weather, time, and neighbor context.
RAW_DAY48_FEATURES = [
    col for col in SPATIAL_FEATURES
    if col not in {"d48gt", "d48roll", "d48roll3", "d48roll5", "d48roll7", "base_self", "base_spatial"}
]


@dataclass
class TrafficContext:
    """Everything the feature builder needs, grouped so we do not rely on globals."""

    base_dir: str
    train: pd.DataFrame
    test: pd.DataFrame
    d48: pd.DataFrame
    d49: pd.DataFrame
    global_mean: float
    temp_mean: float
    ref_d48: pd.Series
    rolls: Dict[int, pd.Series]
    geohash_stats: pd.DataFrame
    anchor_48: pd.Series
    region_stats: Dict[int, Dict[str, pd.Series]]
    geohashes: pd.Index
    decoded_geohashes: Dict[str, Tuple[float, float]]
    neighbors: Dict[str, List[str]]
    neighbor_distances: Dict[str, np.ndarray]
    geohash_to_row: Dict[str, int]
    time_to_col: Dict[int, int]
    d48_matrix_filled: np.ndarray
    d48_matrix_available: np.ndarray
    d48_time_roll_matrix: np.ndarray


# This small helper keeps path handling out of the modeling code. It makes the
# script work both in this notebook folder and in a normal local directory.
def get_base_dir() -> str:
    if os.path.exists("/mnt/data/train.csv") and os.path.exists("/mnt/data/test.csv"):
        return "/mnt/data"
    return "."


# Timestamps are strings like 02:15. Models work better with minutes because
# distance between 02:15 and 02:30 then becomes a normal numeric gap.
def timestamp_to_minutes(value: object) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


# The data gives location as geohash. Decoding gives approximate lat/lon, which
# lets us find nearby cells and build neighborhood demand features.
def decode_geohash(geohash: str) -> Tuple[float, float]:
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    use_lon = True

    for char in geohash:
        code = _GEOHASH_LOOKUP[char]
        for mask in (16, 8, 4, 2, 1):
            if use_lon:
                midpoint = (lon_range[0] + lon_range[1]) / 2.0
                if code & mask:
                    lon_range[0] = midpoint
                else:
                    lon_range[1] = midpoint
            else:
                midpoint = (lat_range[0] + lat_range[1]) / 2.0
                if code & mask:
                    lat_range[0] = midpoint
                else:
                    lat_range[1] = midpoint
            use_lon = not use_lon

    lat = (lat_range[0] + lat_range[1]) / 2.0
    lon = (lon_range[0] + lon_range[1]) / 2.0
    return lat, lon


# Time has a daily cycle. The sine/cosine columns help tree models understand
# that early morning and late night are close in the 24-hour loop.
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tmin"] = df["timestamp"].map(timestamp_to_minutes)
    df["hour"] = df["tmin"] // 60
    df["sin_t"] = np.sin(2.0 * np.pi * df["tmin"] / 1440.0)
    df["cos_t"] = np.cos(2.0 * np.pi * df["tmin"] / 1440.0)
    return df


# We reuse day 48 in several ways, so this function builds the lookup tables once.
# That keeps the feature builder fast and keeps the math in one place.
def build_day48_reference_tables(d48: pd.DataFrame, global_mean: float) -> Tuple[pd.Series, Dict[int, pd.Series], pd.DataFrame, pd.Series, Dict[int, Dict[str, pd.Series]]]:
    key_cols = ["geohash", "tmin"]

    exact_lookup = d48.set_index(key_cols)["demand"]

    rolled = d48.sort_values(key_cols).copy()
    for window in (3, 5, 7):
        rolled[f"roll{window}"] = rolled.groupby("geohash")["demand"].transform(
            lambda values, w=window: values.rolling(w, center=True, min_periods=1).mean()
        )
    roll_lookup = {window: rolled.set_index(key_cols)[f"roll{window}"] for window in (3, 5, 7)}

    geohash_stats = d48.groupby("geohash")["demand"].agg(["mean", "std", "min", "max", "median", "count"])
    anchor_48 = d48[d48["tmin"] == 120].set_index("geohash")["demand"]

    region_source = d48.copy()
    region_stats = {}
    for length in (4, 5):
        prefix_col = f"gh{length}"
        region_source[prefix_col] = region_source["geohash"].str[:length]
        region_stats[length] = {
            "mean": region_source.groupby(prefix_col)["demand"].mean(),
            "time": region_source.groupby([prefix_col, "tmin"])["demand"].mean(),
        }

    return exact_lookup, roll_lookup, geohash_stats, anchor_48, region_stats


# The neighbor index is the spatial part of the solution. For each geohash, we
# keep nearby cells and their distances so we can borrow demand from the area.
def build_neighbor_lookup(train: pd.DataFrame, test: pd.DataFrame, max_neighbors: int = 20) -> Tuple[pd.Index, Dict[str, Tuple[float, float]], Dict[str, List[str]], Dict[str, np.ndarray]]:
    all_geohashes = pd.Index(pd.unique(pd.concat([train["geohash"], test["geohash"]], ignore_index=True))).sort_values()
    decoded = {geohash: decode_geohash(geohash) for geohash in all_geohashes}
    coords = np.array([decoded[geohash] for geohash in all_geohashes])

    # Ask for one extra neighbor because the nearest item is the geohash itself.
    n_neighbors = min(max_neighbors + 1, len(all_geohashes))
    nearest = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
    nearest.fit(coords)
    distances, indices = nearest.kneighbors(coords)

    neighbors = {}
    neighbor_distances = {}
    for row_idx, geohash in enumerate(all_geohashes):
        cells = []
        dists = []
        for neighbor_idx, distance in zip(indices[row_idx], distances[row_idx]):
            neighbor = all_geohashes[neighbor_idx]
            if neighbor == geohash:
                continue
            cells.append(neighbor)
            dists.append(float(distance) + 1e-6)
            if len(cells) >= max_neighbors:
                break
        neighbors[geohash] = cells
        neighbor_distances[geohash] = np.array(dists, dtype=float)

    return all_geohashes, decoded, neighbors, neighbor_distances


# Matrix form makes neighborhood features much faster. Instead of many joins, we
# can grab day-48 demand for all neighbor cells at one timestamp in one shot.
def build_day48_matrices(d48: pd.DataFrame, geohashes: pd.Index, geohash_stats: pd.DataFrame, global_mean: float) -> Tuple[Dict[str, int], Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    times = sorted(d48["tmin"].unique())
    geohash_to_row = {geohash: idx for idx, geohash in enumerate(geohashes)}
    time_to_col = {time_value: idx for idx, time_value in enumerate(times)}

    matrix = np.full((len(geohashes), len(times)), np.nan, dtype=np.float32)
    for geohash, tmin, demand in d48[["geohash", "tmin", "demand"]].itertuples(index=False):
        matrix[geohash_to_row[geohash], time_to_col[tmin]] = demand

    available = ~np.isnan(matrix)
    filled = matrix.copy()
    fallback_by_geohash = np.array(
        [geohash_stats["mean"].get(geohash, global_mean) for geohash in geohashes],
        dtype=np.float32,
    )
    missing_rows, missing_cols = np.where(np.isnan(filled))
    filled[missing_rows, missing_cols] = fallback_by_geohash[missing_rows]

    time_roll = (
        pd.DataFrame(filled, index=geohashes, columns=times)
        .T.rolling(3, center=True, min_periods=1)
        .mean()
        .T.values.astype(np.float32)
    )
    return geohash_to_row, time_to_col, filled, available, time_roll


# This prepares the whole project state. Keeping it in one function makes it easy
# to rerun the same solution on a fresh train/test pair later.
def prepare_context(base_dir: str) -> TrafficContext:
    train = add_time_features(pd.read_csv(os.path.join(base_dir, "train.csv")))
    test = add_time_features(pd.read_csv(os.path.join(base_dir, "test.csv")))

    d48 = train[train["day"] == 48].copy().reset_index(drop=True)
    d49 = train[train["day"] == 49].copy().reset_index(drop=True)

    global_mean = float(d48["demand"].mean())
    temp_mean = float(pd.concat([train["Temperature"], test["Temperature"]], ignore_index=True).mean())

    ref_d48, rolls, geohash_stats, anchor_48, region_stats = build_day48_reference_tables(d48, global_mean)
    geohashes, decoded, neighbors, neighbor_distances = build_neighbor_lookup(train, test)
    geohash_to_row, time_to_col, filled, available, time_roll = build_day48_matrices(
        d48, geohashes, geohash_stats, global_mean
    )

    return TrafficContext(
        base_dir=base_dir,
        train=train,
        test=test,
        d48=d48,
        d49=d49,
        global_mean=global_mean,
        temp_mean=temp_mean,
        ref_d48=ref_d48,
        rolls=rolls,
        geohash_stats=geohash_stats,
        anchor_48=anchor_48,
        region_stats=region_stats,
        geohashes=geohashes,
        decoded_geohashes=decoded,
        neighbors=neighbors,
        neighbor_distances=neighbor_distances,
        geohash_to_row=geohash_to_row,
        time_to_col=time_to_col,
        d48_matrix_filled=filled,
        d48_matrix_available=available,
        d48_time_roll_matrix=time_roll,
    )


# Day 49 starts with two hours of known demand. This function estimates how day 49
# differs from day 48 for each geohash, with shrinkage so small samples do not go wild.
def compute_day49_transfer_signals(ctx: TrafficContext, observed_d49: pd.DataFrame, shrinkage: float = 5.0) -> Tuple[pd.Series, float, pd.Series, pd.Series]:
    overlap = observed_d49.merge(
        ctx.d48[["geohash", "tmin", "demand"]].rename(columns={"demand": "d48_demand"}),
        on=["geohash", "tmin"],
        how="inner",
    )
    if overlap.empty:
        empty = pd.Series(dtype=float)
        return empty, 0.0, empty, empty

    diff = overlap["demand"] - overlap["d48_demand"]
    global_offset = float(diff.mean())

    geohash_offset = diff.groupby(overlap["geohash"]).mean()
    geohash_count = diff.groupby(overlap["geohash"]).count()
    offset_shrunk = (geohash_offset * geohash_count + global_offset * shrinkage) / (geohash_count + shrinkage)

    # A tiny trend term says whether the day-49 offset is rising or falling during
    # the observed window. It is shrunk heavily because two hours is not much data.
    trend_by_geohash = {}
    tmp = overlap.assign(diff=diff)
    for geohash, group in tmp.groupby("geohash"):
        if len(group) >= 3 and group["tmin"].nunique() >= 2:
            x = group["tmin"].values.astype(float)
            y = group["diff"].values.astype(float)
            trend_by_geohash[geohash] = np.polyfit(x - x.mean(), y, 1)[0]
    raw_slope = pd.Series(trend_by_geohash, dtype=float)
    global_slope = float(raw_slope.mean()) if len(raw_slope) else 0.0
    slope_count = geohash_count.reindex(raw_slope.index).fillna(0)
    slope_shrunk = (raw_slope.fillna(global_slope) * slope_count + global_slope * shrinkage) / (slope_count + shrinkage)

    latest_seen = observed_d49.sort_values("tmin").groupby("geohash").tail(1)
    anchor = latest_seen.set_index("geohash")["demand"]
    return offset_shrunk, global_offset, anchor, slope_shrunk


# Some geohashes have weak or missing early-day information. This helper borrows
# a value from nearby cells using either a plain mean or inverse-distance weighting.
def neighbor_series_lookup(ctx: TrafficContext, geohashes: Iterable[str], series: pd.Series, default: float, k: int = 8, mode: str = "mean") -> np.ndarray:
    result = []
    for geohash in geohashes:
        values = []
        weights = []
        for neighbor, distance in zip(ctx.neighbors[geohash][:k], ctx.neighbor_distances[geohash][:k]):
            value = series.get(neighbor, np.nan)
            if pd.notna(value):
                values.append(float(value))
                weights.append(1.0 / distance)

        if not values:
            result.append(default)
            continue

        values_arr = np.array(values, dtype=float)
        weights_arr = np.array(weights, dtype=float)
        if mode == "idw":
            result.append(float(np.sum(values_arr * weights_arr) / np.sum(weights_arr)))
        elif mode == "std":
            result.append(float(np.std(values_arr)))
        else:
            result.append(float(np.mean(values_arr)))

    return np.array(result, dtype=float)


# This is the main feature builder. The target is not raw demand. We build a good
# base forecast first, then the models learn the residual left over from that base.
def build_spatial_features(
    ctx: TrafficContext,
    df: pd.DataFrame,
    offset: pd.Series,
    global_offset: float,
    anchor: pd.Series,
    slope: Optional[pd.Series] = None,
    k: int = 8,
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(df[["geohash", "tmin"]])

    df["d48gt"] = pd.Series(idx.map(ctx.ref_d48), index=df.index)
    df["d48gt"] = df["d48gt"].fillna(df["geohash"].map(ctx.geohash_stats["mean"])).fillna(ctx.global_mean)

    for window in (3, 5, 7):
        df[f"d48roll{window}"] = pd.Series(idx.map(ctx.rolls[window]), index=df.index).fillna(df["d48gt"])
    df["d48roll"] = df["d48roll3"]

    for stat in ["mean", "std", "min", "max", "median", "count"]:
        df[f"g_{stat}"] = df["geohash"].map(ctx.geohash_stats[stat])
    df["g_mean"] = df["g_mean"].fillna(ctx.global_mean)
    df["g_std"] = df["g_std"].fillna(0.0)
    df["g_count"] = df["g_count"].fillna(0.0)
    for col in ["g_min", "g_max", "g_median"]:
        df[col] = df[col].fillna(ctx.global_mean)

    df["offset_shrunk"] = df["geohash"].map(offset).fillna(global_offset)
    df["anchor"] = df["geohash"].map(anchor).fillna(df["g_mean"])
    df["anch48"] = df["geohash"].map(ctx.anchor_48).fillna(df["g_mean"])
    df["anchor_diff"] = df["anchor"] - df["anch48"]
    df["gap"] = df["tmin"] - 120

    if slope is None:
        slope = pd.Series(dtype=float)
    df["offset_slope"] = df["geohash"].map(slope).fillna(0.0)
    df["early_trend_projected"] = df["offset_slope"] * df["gap"]

    df["lat"] = df["geohash"].map(lambda geohash: ctx.decoded_geohashes[geohash][0])
    df["lon"] = df["geohash"].map(lambda geohash: ctx.decoded_geohashes[geohash][1])

    df["lanes"] = df["NumberofLanes"]
    df["lv"] = df["LargeVehicles"].map({"Allowed": 1, "Not Allowed": 0}).fillna(-1)
    df["lm"] = df["Landmarks"].map({"Yes": 1, "No": 0}).fillna(-1)
    df["rt"] = df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(-1)
    df["wx"] = df["Weather"].map({"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}).fillna(-1)
    df["temp_missing"] = df["Temperature"].isna().astype(int)
    df["temp_filled"] = df["Temperature"].fillna(ctx.temp_mean)

    for length in (4, 5):
        prefix_col = f"gh{length}"
        df[prefix_col] = df["geohash"].str[:length]
        df[f"{prefix_col}_mean"] = df[prefix_col].map(ctx.region_stats[length]["mean"]).fillna(ctx.global_mean)
        region_time_index = pd.MultiIndex.from_arrays([df[prefix_col], df["tmin"]])
        df[f"{prefix_col}_time_mean"] = pd.Series(
            region_time_index.map(ctx.region_stats[length]["time"]), index=df.index
        ).fillna(df[f"{prefix_col}_mean"])

    neighbor_mean = []
    neighbor_idw = []
    neighbor_std = []
    neighbor_max = []
    neighbor_min = []
    neighbor_roll_idw = []
    neighbor_avail_frac = []

    for geohash, tmin in zip(df["geohash"], df["tmin"]):
        col_idx = ctx.time_to_col.get(tmin, 0)
        neighbor_rows = [ctx.geohash_to_row[neighbor] for neighbor in ctx.neighbors[geohash][:k]]
        distances = ctx.neighbor_distances[geohash][:k]
        weights = 1.0 / distances

        values = ctx.d48_matrix_filled[neighbor_rows, col_idx]
        available = ctx.d48_matrix_available[neighbor_rows, col_idx]
        rolled_values = ctx.d48_time_roll_matrix[neighbor_rows, col_idx]

        neighbor_mean.append(float(np.mean(values)))
        neighbor_idw.append(float(np.sum(values * weights) / np.sum(weights)))
        neighbor_std.append(float(np.std(values)))
        neighbor_max.append(float(np.max(values)))
        neighbor_min.append(float(np.min(values)))
        neighbor_roll_idw.append(float(np.sum(rolled_values * weights) / np.sum(weights)))
        neighbor_avail_frac.append(float(np.mean(available)))

    df[f"n{k}_d48_mean"] = neighbor_mean
    df[f"n{k}_d48_idw"] = neighbor_idw
    df[f"n{k}_d48_std"] = neighbor_std
    df[f"n{k}_d48_max"] = neighbor_max
    df[f"n{k}_d48_min"] = neighbor_min
    df[f"n{k}_d48_roll_idw"] = neighbor_roll_idw
    df[f"n{k}_d48_avail_frac"] = neighbor_avail_frac

    df[f"n{k}_offset_mean"] = neighbor_series_lookup(ctx, df["geohash"], offset, global_offset, k=k, mode="mean")
    df[f"n{k}_offset_idw"] = neighbor_series_lookup(ctx, df["geohash"], offset, global_offset, k=k, mode="idw")
    df[f"n{k}_offset_std"] = neighbor_series_lookup(ctx, df["geohash"], offset, 0.0, k=k, mode="std")
    df[f"n{k}_anchor_idw"] = neighbor_series_lookup(ctx, df["geohash"], anchor, ctx.global_mean, k=k, mode="idw")
    df[f"n{k}_anch48_idw"] = neighbor_series_lookup(ctx, df["geohash"], ctx.anchor_48, ctx.global_mean, k=k, mode="idw")
    df[f"n{k}_anchor_diff_idw"] = df[f"n{k}_anchor_idw"] - df[f"n{k}_anch48_idw"]
    df[f"n{k}_slope_idw"] = neighbor_series_lookup(ctx, df["geohash"], slope, 0.0, k=k, mode="idw")

    df["base_self"] = 0.55 * df["d48roll3"] + 0.30 * df["d48roll5"] + 0.15 * df["d48roll7"] + df["offset_shrunk"]
    df["base_spatial"] = 0.78 * df["base_self"] + 0.22 * (df[f"n{k}_d48_roll_idw"] + df[f"n{k}_offset_idw"])
    df["base"] = df["base_spatial"]
    return df


# Model inputs should not contain NaN or infinity. This one line is used by every
# tree model so the cleaning rule stays consistent.
def clean_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)


# Residual learning is safer here than predicting demand from scratch. The base
# already captures yesterday's shape; the model only learns what the base missed.
def train_spatial_residual_ensemble(train_features: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
    X_train = clean_matrix(train_features, SPATIAL_FEATURES)
    X_test = clean_matrix(test_features, SPATIAL_FEATURES)
    residual = train_features["demand"].values - train_features["base"].values

    predictions = []

    hgb_model = HistGradientBoostingRegressor(
        max_iter=450,
        learning_rate=0.045,
        max_leaf_nodes=31,
        l2_regularization=3.0,
        min_samples_leaf=25,
        random_state=11,
    )
    hgb_model.fit(X_train, residual)
    predictions.append((0.45, test_features["base"].values + hgb_model.predict(X_test)))
    print("trained spatial residual HGB")

    trees_model = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        random_state=11,
        n_jobs=-1,
    )
    trees_model.fit(X_train, residual)
    predictions.append((0.25, test_features["base"].values + trees_model.predict(X_test)))
    print("trained spatial residual ExtraTrees")

    if lgb is not None:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=25,
            reg_lambda=6.0,
            feature_fraction=0.90,
            bagging_fraction=0.90,
            bagging_freq=1,
            random_state=11,
            n_jobs=2,
            verbose=-1,
        )
        lgb_model.fit(X_train, residual)
        predictions.append((0.30, test_features["base"].values + lgb_model.predict(X_test)))
        print("trained spatial residual LightGBM")

    total_weight = sum(weight for weight, _ in predictions)
    blended = sum(weight * pred for weight, pred in predictions) / total_weight
    return np.clip(blended, 0.0, 1.0)


# This model is intentionally different from the residual model. It learns the raw
# day-48 demand curve and gives the final blend another independent opinion.
def train_day48_raw_model(d48_features: pd.DataFrame, test_features: pd.DataFrame) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=8.0,
        min_samples_leaf=40,
        random_state=321,
    )
    model.fit(clean_matrix(d48_features, RAW_DAY48_FEATURES), d48_features["demand"].values)
    print("trained day-48 raw HGB")
    return np.clip(model.predict(clean_matrix(test_features, RAW_DAY48_FEATURES)), 0.0, 1.0)


# CatBoost handles geohash, timestamp, road type, and weather as categories. It is
# useful because it sees the raw table in a different way than the residual models.
def train_catboost_direct_model(ctx: TrafficContext) -> np.ndarray:
    train = ctx.train.copy()
    test = ctx.test.copy()

    for df in (train, test):
        df["gh4"] = df["geohash"].str[:4]
        df["gh5"] = df["geohash"].str[:5]
        df["Temperature_filled"] = df["Temperature"].fillna(ctx.temp_mean)
        df["temp_missing"] = df["Temperature"].isna().astype(int)
        for col in ["geohash", "timestamp", "RoadType", "LargeVehicles", "Landmarks", "Weather", "gh4", "gh5"]:
            df[col] = df[col].fillna("__NA__").astype(str)

    features = [
        "geohash", "timestamp", "gh4", "gh5",
        "tmin", "hour", "sin_t", "cos_t",
        "RoadType", "NumberofLanes", "LargeVehicles", "Landmarks",
        "Temperature_filled", "temp_missing", "Weather",
    ]
    categorical_features = ["geohash", "timestamp", "gh4", "gh5", "RoadType", "LargeVehicles", "Landmarks", "Weather"]

    # Day 49 is closer to the test period than day 48, so it gets more weight.
    sample_weight = np.where(train["day"].values == 49, 4.0, 1.0)

    model = CatBoostRegressor(
        iterations=250,
        learning_rate=0.06,
        depth=6,
        loss_function="RMSE",
        random_seed=202,
        l2_leaf_reg=8,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )
    model.fit(train[features], train["demand"], cat_features=categorical_features, sample_weight=sample_weight)
    print("trained CatBoost direct model")
    return np.clip(model.predict(test[features]), 0.0, 1.0)


# These two residual models are smaller than the main spatial ensemble. Their job
# is not to win alone, but to add useful disagreement to the final blend.
def train_extra_residual_models(train_features: pd.DataFrame, test_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_train = clean_matrix(train_features, SPATIAL_FEATURES)
    X_test = clean_matrix(test_features, SPATIAL_FEATURES)
    residual = train_features["demand"].values - train_features["base"].values

    if lgb is not None:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=10,
            reg_lambda=5.0,
            feature_fraction=0.90,
            bagging_fraction=0.85,
            bagging_freq=1,
            random_state=502,
            n_jobs=2,
            verbose=-1,
        )
        lgb_model.fit(X_train, residual)
        lgb_pred = test_features["base"].values + lgb_model.predict(X_test)
        print("trained extra LightGBM residual model")
    else:
        lgb_pred = test_features["base"].values.copy()
        print("LightGBM not found; using base prediction for that slot")

    hgb_model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=6.0,
        min_samples_leaf=20,
        random_state=503,
    )
    hgb_model.fit(X_train, residual)
    hgb_pred = test_features["base"].values + hgb_model.predict(X_test)
    print("trained extra HGB residual model")

    return np.clip(lgb_pred, 0.0, 1.0), np.clip(hgb_pred, 0.0, 1.0)


# Reading submissions by Index avoids a quiet but painful bug: two files can have
# the same rows in a different order. This returns predictions in test.csv order.
def load_submission_prediction(base_dir: str, filename: Optional[str], test_index: pd.Series) -> Optional[np.ndarray]:
    if filename is None:
        return None
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        return None
    sub = pd.read_csv(path)[["Index", "demand"]]
    aligned = pd.DataFrame({"Index": test_index}).merge(sub, on="Index", how="left")
    if aligned["demand"].isna().any():
        raise ValueError(f"{filename} does not cover every Index in test.csv")
    return aligned["demand"].values


# The spatial blend from the previous step was the first big jump. This function
# rebuilds it directly so the V3 script does not depend on another Python file.
def build_spatial_backbone_prediction(ctx: TrafficContext, train_features: pd.DataFrame, test_features: pd.DataFrame, d48_features: pd.DataFrame) -> np.ndarray:
    spatial_pred = train_spatial_residual_ensemble(train_features, test_features)
    day48_raw_pred = train_day48_raw_model(d48_features, test_features)
    previous_best = load_submission_prediction(ctx.base_dir, BASELINE_FILE, ctx.test["Index"])

    if previous_best is None:
        # Same balance as the old spatial blend, just renormalized without the optional baseline.
        pred = (0.50 * spatial_pred + 0.38 * day48_raw_pred) / 0.88
        print("baseline file not found; spatial backbone uses spatial + raw models only")
    else:
        pred = 0.12 * previous_best + 0.50 * spatial_pred + 0.38 * day48_raw_pred
        print("built spatial backbone with previous baseline included")

    return np.clip(pred, 0.0, 1.0)


# These weights are the clean V3 recommendation. They were chosen as broad, stable
# weights, not by fitting to the answer key.
def blend_v3_recommended(
    ctx: TrafficContext,
    spatial_backbone: np.ndarray,
    catboost_pred: np.ndarray,
    lgb_residual_pred: np.ndarray,
    hgb_residual_pred: np.ndarray,
) -> np.ndarray:
    previous_best = load_submission_prediction(ctx.base_dir, BASELINE_FILE, ctx.test["Index"])

    if previous_best is None:
        # Remove the 2% previous-baseline slot and give it back to the spatial backbone.
        weights = {
            "spatial": 0.72,
            "cat": 0.14,
            "lgb": 0.08,
            "hgb": 0.06,
        }
        final = (
            weights["spatial"] * spatial_backbone
            + weights["cat"] * catboost_pred
            + weights["lgb"] * lgb_residual_pred
            + weights["hgb"] * hgb_residual_pred
        )
    else:
        final = (
            0.70 * spatial_backbone
            + 0.14 * catboost_pred
            + 0.02 * previous_best
            + 0.08 * lgb_residual_pred
            + 0.06 * hgb_residual_pred
        )

    return np.clip(final, 0.0, 1.0)




# Intermediate predictions are saved too. They are useful for checking a run, and
# they let the script resume quickly without retraining every model again.
def save_intermediate_prediction(ctx: TrafficContext, predictions: np.ndarray, filename: str) -> str:
    path = os.path.join(ctx.base_dir, filename)
    pd.DataFrame({"Index": ctx.test["Index"].values, "demand": np.clip(predictions, 0.0, 1.0)}).to_csv(path, index=False)
    print(f"saved {path}")
    return path


# This wrapper keeps caching out of the modeling function itself. The model code
# stays honest, and reruns stay quick when the same train/test files are used.
def load_or_build_spatial_backbone(
    ctx: TrafficContext,
    train_features: Optional[pd.DataFrame],
    test_features: Optional[pd.DataFrame],
    d48_features: Optional[pd.DataFrame],
) -> np.ndarray:
    cached = load_submission_prediction(ctx.base_dir, SPATIAL_BACKBONE_FILE, ctx.test["Index"])
    if USE_CACHED_INTERMEDIATE_PREDICTIONS and cached is not None:
        print(f"using cached {SPATIAL_BACKBONE_FILE}")
        return cached

    if train_features is None or test_features is None or d48_features is None:
        raise ValueError("Spatial features are needed to train the spatial backbone.")

    pred = build_spatial_backbone_prediction(ctx, train_features, test_features, d48_features)
    save_intermediate_prediction(ctx, pred, SPATIAL_BACKBONE_FILE)
    return pred


# CatBoost can take a while, so this wrapper reuses the direct-model file when it
# already exists. On a new dataset it trains normally and then saves the result.
def load_or_train_catboost_direct(ctx: TrafficContext) -> np.ndarray:
    cached = load_submission_prediction(ctx.base_dir, CATBOOST_DIRECT_FILE, ctx.test["Index"])
    if USE_CACHED_INTERMEDIATE_PREDICTIONS and cached is not None:
        print(f"using cached {CATBOOST_DIRECT_FILE}")
        return cached

    pred = train_catboost_direct_model(ctx)
    save_intermediate_prediction(ctx, pred, CATBOOST_DIRECT_FILE)
    return pred


# Same idea for the two smaller residual models. They are not the headline model,
# but their small disagreements help the final blend.
def load_or_train_extra_residuals(
    ctx: TrafficContext,
    train_features: Optional[pd.DataFrame],
    test_features: Optional[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray]:
    cached_lgb = load_submission_prediction(ctx.base_dir, EXTRA_LGB_FILE, ctx.test["Index"])
    cached_hgb = load_submission_prediction(ctx.base_dir, EXTRA_HGB_FILE, ctx.test["Index"])
    if USE_CACHED_INTERMEDIATE_PREDICTIONS and cached_lgb is not None and cached_hgb is not None:
        print(f"using cached {EXTRA_LGB_FILE}")
        print(f"using cached {EXTRA_HGB_FILE}")
        return cached_lgb, cached_hgb

    if train_features is None or test_features is None:
        raise ValueError("Spatial features are needed to train residual models.")

    lgb_pred, hgb_pred = train_extra_residual_models(train_features, test_features)
    save_intermediate_prediction(ctx, lgb_pred, EXTRA_LGB_FILE)
    save_intermediate_prediction(ctx, hgb_pred, EXTRA_HGB_FILE)
    return lgb_pred, hgb_pred


# Every submission is written the same way: original test Index, clipped demand.
# Clipping is needed because the metric expects demand in the 0 to 1 range.
def save_submission(ctx: TrafficContext, predictions: np.ndarray, filename: str = OUTPUT_FILE) -> str:
    path = os.path.join(ctx.base_dir, filename)
    output = pd.DataFrame({"Index": ctx.test["Index"].values, "demand": np.clip(predictions, 0.0, 1.0)})
    output.to_csv(path, index=False)
    print(f"saved {path}")
    return path


# Main stays short on purpose. It reads like the actual modeling flow instead of
# hiding the important steps inside one large script body.
def main() -> str:
    base_dir = get_base_dir()
    ctx = prepare_context(base_dir)

    # Build the expensive spatial feature tables only when a cached intermediate
    # is missing. This keeps normal reruns quick but still supports a clean run.
    need_spatial_features = not (
        USE_CACHED_INTERMEDIATE_PREDICTIONS
        and load_submission_prediction(ctx.base_dir, SPATIAL_BACKBONE_FILE, ctx.test["Index"]) is not None
        and load_submission_prediction(ctx.base_dir, EXTRA_LGB_FILE, ctx.test["Index"]) is not None
        and load_submission_prediction(ctx.base_dir, EXTRA_HGB_FILE, ctx.test["Index"]) is not None
    )

    train_features = None
    d48_features = None
    test_features = None
    if need_spatial_features:
        offset, global_offset, anchor, slope = compute_day49_transfer_signals(ctx, ctx.d49)
        train_features = build_spatial_features(ctx, ctx.d49, offset, global_offset, anchor, slope, k=8)
        d48_features = build_spatial_features(ctx, ctx.d48, offset, global_offset, anchor, slope, k=8)
        test_features = build_spatial_features(ctx, ctx.test, offset, global_offset, anchor, slope, k=8)

    spatial_backbone = load_or_build_spatial_backbone(ctx, train_features, test_features, d48_features)
    catboost_pred = load_or_train_catboost_direct(ctx)
    lgb_residual_pred, hgb_residual_pred = load_or_train_extra_residuals(ctx, train_features, test_features)

    final_prediction = blend_v3_recommended(
        ctx=ctx,
        spatial_backbone=spatial_backbone,
        catboost_pred=catboost_pred,
        lgb_residual_pred=lgb_residual_pred,
        hgb_residual_pred=hgb_residual_pred,
    )
    return save_submission(ctx, final_prediction)


if __name__ == "__main__":
    main()
