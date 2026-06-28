"""
Bangalore House Price Prediction - end-to-end training pipeline.
Stages: load -> clean -> feature engineer -> reduce dims -> remove outliers
        -> encode -> split -> train/compare models -> tune -> evaluate -> save artifacts.
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RAW = "/mnt/user-data/uploads/bengaluru_house_prices.csv"
SEED = 42


def log(msg):
    print(f"[pipeline] {msg}")


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
df = pd.read_csv(RAW)
log(f"loaded raw shape={df.shape}")

# ----------------------------------------------------------------------------
# 2. CLEAN: drop low-value columns, drop sparse missing rows
# ----------------------------------------------------------------------------
# society: 41% missing & huge cardinality. availability/area_type: drop to keep
# the standard, comparable version of this project simple.
df = df.drop(columns=["area_type", "society", "availability", "balcony"])
df = df.dropna()  # removes the small number of size/bath/location nulls
log(f"after cleaning shape={df.shape}")

# ----------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
# 3a. bhk from "2 BHK" / "4 Bedroom" / "1 RK"
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))


# 3b. total_sqft -> float (handle ranges like "2100 - 2850"; drop unit oddities)
def to_sqft(x):
    toks = str(x).split("-")
    if len(toks) == 2:
        try:
            return (float(toks[0]) + float(toks[1])) / 2
        except ValueError:
            return None
    try:
        return float(x)
    except ValueError:
        return None  # "34.46Sq. Meter", "4125Perch", etc.


df["total_sqft"] = df["total_sqft"].apply(to_sqft)
df = df.dropna(subset=["total_sqft"])
log(f"after sqft clean shape={df.shape}")

# 3c. engineered signal used for outlier logic + as a real feature later dropped
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# ----------------------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION on location (1305 -> manageable)
# ----------------------------------------------------------------------------
df["location"] = df["location"].apply(lambda x: x.strip())
loc_counts = df["location"].value_counts()
rare = loc_counts[loc_counts <= 10].index
df["location"] = df["location"].apply(lambda x: "other" if x in rare else x)
log(f"locations after reduction={df['location'].nunique()}")

# ----------------------------------------------------------------------------
# 5. OUTLIER REMOVAL (domain logic)
# ----------------------------------------------------------------------------
# 5a. unrealistic sqft per room (< 300 sqft/bhk is a data error)
df = df[~(df["total_sqft"] / df["bhk"] < 300)]
log(f"after sqft/bhk filter shape={df.shape}")


# 5b. price_per_sqft beyond +/-1 std WITHIN each location
def trim_pps(d):
    out = pd.DataFrame()
    for _, sub in d.groupby("location"):
        m, s = sub["price_per_sqft"].mean(), sub["price_per_sqft"].std()
        keep = sub[(sub["price_per_sqft"] > (m - s)) & (sub["price_per_sqft"] <= (m + s))]
        out = pd.concat([out, keep], ignore_index=True)
    return out


df = trim_pps(df)
log(f"after pps filter shape={df.shape}")


# 5c. a (location, bhk) costing less than the same location's (bhk-1) mean is suspect
def trim_bhk(d):
    exclude = np.array([])
    for loc, loc_df in d.groupby("location"):
        stats = {}
        for bhk, bhk_df in loc_df.groupby("bhk"):
            stats[bhk] = {"mean": bhk_df["price_per_sqft"].mean(), "count": bhk_df.shape[0]}
        for bhk, bhk_df in loc_df.groupby("bhk"):
            prev = stats.get(bhk - 1)
            if prev and prev["count"] > 5:
                exclude = np.append(exclude, bhk_df[bhk_df["price_per_sqft"] < prev["mean"]].index.values)
    return d.drop(exclude)


df = trim_bhk(df)
log(f"after bhk filter shape={df.shape}")

# 5d. bathrooms shouldn't exceed bhk + 2 (data error)
df = df[df["bath"] < df["bhk"] + 2]
log(f"after bath filter shape={df.shape}")

# ----------------------------------------------------------------------------
# 6. PREP FOR MODELLING
# ----------------------------------------------------------------------------
df = df.drop(columns=["size", "price_per_sqft"])  # size superseded by bhk; pps was a helper
dummies = pd.get_dummies(df["location"], drop_first=True).astype(int)
df = pd.concat([df.drop(columns=["location"]), dummies], axis=1)

X = df.drop(columns=["price"])
y = df["price"]
log(f"final feature matrix={X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def evaluate(model, name):
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"  {name:22s} R2={r2:.4f}  MAE={mae:6.2f}L  RMSE={rmse:6.2f}L")
    return r2, mae, rmse


# ----------------------------------------------------------------------------
# 7. BASELINE MODELS
# ----------------------------------------------------------------------------
print("\n--- baseline models (default params) ---")
results = {}
for name, mdl in [
    ("LinearRegression", LinearRegression()),
    ("Lasso", Lasso(alpha=1.0)),
    ("DecisionTree", DecisionTreeRegressor(random_state=SEED)),
    ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)),
]:
    mdl.fit(X_train, y_train)
    results[name] = evaluate(mdl, name)

# cross-val on linear (the classic check for this project)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=SEED)
lr_cv = cross_val_score(LinearRegression(), X, y, cv=cv)
print(f"\n  LinearRegression 5-fold CV R2: {lr_cv.mean():.4f} (+/- {lr_cv.std():.4f})")

# save intermediate for the tuning step
X.to_pickle("/home/claude/bhp/_X.pkl")
y.to_pickle("/home/claude/bhp/_y.pkl")
with open("/home/claude/bhp/model/columns.json", "w") as f:
    json.dump({"data_columns": [c.lower() for c in X.columns]}, f)
log("baseline stage done; X/y cached for tuning")
