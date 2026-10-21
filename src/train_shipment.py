"""Train and evaluate XGBoost shipment-delay models on a chronological split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_shipment_data.csv"
OUTPUT_DIR = ROOT / "models"
# Use a material-delay cutoff: the dataset has only nine on-time records, so "any delay" is highly imbalanced.
HIGH_DELAY_HOURS = 6.0

NUMERIC_FEATURES = [
    "is_weekend", "is_peak_season", "distance_km", "num_transit_stops",
    "package_weight_kg", "carrier_avg_delay_hr", "carrier_on_time_rate",
    "carrier_network_congestion_score", "avg_temp_origin", "precipitation_prob",
    "storm_alert", "ship_hour", "month", "distance_per_stop",
    "distance_x_congestion", "stops_x_congestion", "weather_pressure",
]
CATEGORICAL_FEATURES = [
    "origin", "destination", "route", "carrier", "shipment_type", "route_type",
    "ship_day_of_week", "weather_origin", "weather_destination", "shipping_hour_bin",
    "carrier_network_congestion",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-shipment features and drop source columns unavailable to the model."""
    required = {"ship_date", "ship_hour", "distance_km", "delay_hr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    out = df.copy()
    out["ship_date"] = pd.to_datetime(out["ship_date"], errors="raise")
    out["month"] = out["ship_date"].dt.month
    out["route"] = out["origin"].astype(str) + " → " + out["destination"].astype(str)
    out["carrier_network_congestion_score"] = out["carrier_network_congestion"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    out["distance_per_stop"] = out["distance_km"] / (out["num_transit_stops"] + 1)
    out["distance_x_congestion"] = out["distance_km"] * out["carrier_network_congestion_score"]
    out["stops_x_congestion"] = out["num_transit_stops"] * out["carrier_network_congestion_score"]
    out["weather_pressure"] = (
        out["precipitation_prob"] * (1 + out["storm_alert"])
        + out["weather_origin"].isin(["storm", "rain"]).astype(int)
        + out["weather_destination"].isin(["storm", "rain"]).astype(int)
    )
    # Keep dates for chronological splitting, but use derived month/day features rather than raw timestamps.
    return out.sort_values(["ship_date", "ship_hour"], kind="stable").reset_index(drop=True)


def eta_baseline_hours(df: pd.DataFrame) -> pd.Series:
    """Return assumed schedule duration; speeds are explicit assignment assumptions."""
    # The source has no schedule or planned duration; these speeds and stop times are disclosed placeholders.
    assumed_speed_kmh = df["route_type"].map({"highway": 55.0, "mixed": 40.0, "rural": 30.0}).fillna(40.0)
    return df["distance_km"] / assumed_speed_kmh + df["num_transit_stops"] * 4.0


def contribution_rows(model, matrix, feature_names: list[str]) -> np.ndarray:
    dmatrix = xgb.DMatrix(matrix, feature_names=feature_names)
    # XGBoost appends a bias term as the final contribution column; callers omit it from factor rankings.
    return model.get_booster().predict(dmatrix, pred_contribs=True)


def top_factors(values: np.ndarray, names: list[str], count: int = 3) -> str:
    order = np.argsort(np.abs(values))[::-1][:count]
    return json.dumps([
        {"feature": names[i], "contribution": round(float(values[i]), 4)}
        for i in order
    ])


def train() -> dict:
    raw = pd.read_csv(DATA_PATH)
    df = prepare_data(raw)
    n = len(df)
    # Keep later shipments held out; the preprocessor below is fitted on the training partition only.
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    train_df, val_df, test_df = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    x_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    x_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["delay_hr"].astype(float)
    y_val = val_df["delay_hr"].astype(float)
    y_test = test_df["delay_hr"].astype(float)

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    x_train_t = preprocessor.fit_transform(x_train)
    x_val_t = preprocessor.transform(x_val)
    x_test_t = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())

    regressor = XGBRegressor(
        objective="reg:squarederror", n_estimators=350, max_depth=3,
        learning_rate=0.035, subsample=0.85, colsample_bytree=0.85,
        reg_lambda=2.0, random_state=42, n_jobs=4, early_stopping_rounds=30,
    )
    # Validation is used only for early stopping; the final chronological test set stays untouched until scoring.
    regressor.fit(x_train_t, y_train, eval_set=[(x_val_t, y_val)], verbose=False)
    reg_pred = np.maximum(0, regressor.predict(x_test_t))

    # Both risk labels use the same business cutoff so the classifier probability has a clear interpretation.
    train_risk = (y_train >= HIGH_DELAY_HOURS).astype(int)
    val_risk = (y_val >= HIGH_DELAY_HOURS).astype(int)
    test_risk = (y_test >= HIGH_DELAY_HOURS).astype(int)
    classifier = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=350,
        max_depth=3, learning_rate=0.035, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=2.0, random_state=42, n_jobs=4,
        early_stopping_rounds=30,
    )
    classifier.fit(x_train_t, train_risk, eval_set=[(x_val_t, val_risk)], verbose=False)
    risk_prob = classifier.predict_proba(x_test_t)[:, 1]
    risk_pred = (risk_prob >= 0.5).astype(int)

    reg_contrib = contribution_rows(regressor, x_test_t, feature_names)[:, :-1]
    cls_contrib = contribution_rows(classifier, x_test_t, feature_names)[:, :-1]
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    regressor.save_model(OUTPUT_DIR / "xgb_delay_regressor.json")
    classifier.save_model(OUTPUT_DIR / "xgb_high_delay_classifier.json")

    results = pd.DataFrame({
        "ship_date": test_df["ship_date"].dt.strftime("%Y-%m-%d"),
        "origin": test_df["origin"], "destination": test_df["destination"],
        "carrier": test_df["carrier"], "actual_delay_hr": y_test.to_numpy(),
        "predicted_delay_hr": np.round(reg_pred, 2),
        "high_delay_threshold_hr": HIGH_DELAY_HOURS,
        "actual_high_delay": test_risk.to_numpy(),
        "predicted_high_delay_probability": np.round(risk_prob, 4),
        "predicted_high_delay": risk_pred,
    })
    baseline = eta_baseline_hours(test_df)
    ship_datetime = pd.to_datetime(test_df["ship_date"].dt.strftime("%Y-%m-%d") + " " + test_df["ship_hour"].astype(str) + ":00:00")
    # ETA combines the assumed planned duration with the predicted delay; it is not a schedule-derived ETA.
    results["assumed_planned_transit_hr"] = np.round(baseline.to_numpy(), 2)
    results["predicted_eta"] = (ship_datetime + pd.to_timedelta(baseline.to_numpy() + reg_pred, unit="h")).dt.strftime("%Y-%m-%d %H:%M")
    results["top_delay_factors"] = [top_factors(row, feature_names) for row in reg_contrib]
    results["top_high_delay_factors"] = [top_factors(row, feature_names) for row in cls_contrib]
    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    reg_global = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_contribution_delay_hr": np.mean(np.abs(reg_contrib), axis=0),
        "mean_contribution_delay_hr": np.mean(reg_contrib, axis=0),
        "mean_abs_contribution_high_delay": np.mean(np.abs(cls_contrib), axis=0),
        "mean_contribution_high_delay": np.mean(cls_contrib, axis=0),
    }).sort_values("mean_abs_contribution_delay_hr", ascending=False)
    reg_global.to_csv(OUTPUT_DIR / "feature_contributions.csv", index=False)

    summary = {
        "dataset": {"rows": n, "features": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES), "date_range": [str(df.ship_date.min().date()), str(df.ship_date.max().date())]},
        "split": {"train_rows": len(train_df), "validation_rows": len(val_df), "test_rows": len(test_df), "train_through": str(train_df.ship_date.max().date()), "validation_through": str(val_df.ship_date.max().date())},
        "targets": {"delay_regression": "delay_hr", "high_delay_classification": f"delay_hr >= {HIGH_DELAY_HOURS} hours", "high_delay_train_rate": round(float(train_risk.mean()), 4), "high_delay_validation_rate": round(float(val_risk.mean()), 4), "high_delay_test_rate": round(float(test_risk.mean()), 4)},
        "eta_assumption": "ETA = ship_date/ship_hour + distance_km / assumed route speed + 4 hours per transit stop + predicted delay_hr; assumed route speeds: highway 55, mixed 40, rural 30 km/h.",
        "regression_test": {"mae_hr": round(float(mean_absolute_error(y_test, reg_pred)), 4), "rmse_hr": round(float(np.sqrt(mean_squared_error(y_test, reg_pred))), 4), "r2": round(float(r2_score(y_test, reg_pred)), 4)},
        "classification_test": {"threshold": 0.5, "accuracy": round(float(accuracy_score(test_risk, risk_pred)), 4), "precision": round(float(precision_score(test_risk, risk_pred, zero_division=0)), 4), "recall": round(float(recall_score(test_risk, risk_pred, zero_division=0)), 4), "f1": round(float(f1_score(test_risk, risk_pred, zero_division=0)), 4), "roc_auc": round(float(roc_auc_score(test_risk, risk_prob)), 4), "average_precision": round(float(average_precision_score(test_risk, risk_prob)), 4), "confusion_matrix_tn_fp_fn_tp": confusion_matrix(test_risk, risk_pred).ravel().tolist()},
        "top_global_delay_factors": reg_global.head(10).to_dict(orient="records"),
    }
    (OUTPUT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved models, test predictions, and contribution reports in {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    train()
