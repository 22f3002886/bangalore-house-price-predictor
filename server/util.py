"""Artifact loading + prediction helpers for the Bangalore price predictor."""
import json, pickle, os
import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE, "..", "model")

_columns   = None   # exact column names as the model was fitted with
_loc_map   = None   # lowercase -> original-case, for case-insensitive UI input
_model     = None


def load_artifacts():
    global _columns, _loc_map, _model
    with open(os.path.join(_MODEL_DIR, "columns.json")) as f:
        _columns = json.load(f)["data_columns"]
    # first 3 are numeric (total_sqft, bath, bhk); rest are one-hot location dummies
    _loc_map = {c.lower(): c for c in _columns[3:]}
    with open(os.path.join(_MODEL_DIR, "bangalore_home_prices_model.pkl"), "rb") as f:
        _model = pickle.load(f)
    print(f"[util] loaded model + {len(_loc_map)} locations")


def get_location_names():
    if _loc_map is None:
        load_artifacts()
    # return original-case names for the dropdown
    return list(_loc_map.values())


def predict_price(location: str, sqft: float, bath: int, bhk: int) -> float:
    if _model is None:
        load_artifacts()
    loc_key = location.strip().lower()
    orig_col = _loc_map.get(loc_key)          # None if location is 'other' / unknown
    x = np.zeros(len(_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if orig_col:
        x[_columns.index(orig_col)] = 1
    df = pd.DataFrame([x], columns=_columns)
    return round(float(_model.predict(df)[0]), 2)
