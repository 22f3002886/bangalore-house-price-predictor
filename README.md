# Bangalore House Price Predictor

End-to-end ML project: data cleaning → feature engineering → outlier removal → model training → Flask web app.

## Model performance

| Metric       | Value                     |
| ------------ | ------------------------- |
| Algorithm    | Random Forest (100 trees) |
| Test R²      | 0.888                     |
| Test MAE     | ₹14.85 Lakh               |
| Test RMSE    | ₹27.46 Lakh               |
| 5-fold CV R² | 0.82                      |

## Project structure

```
bhp/
├── model/
│   ├── bangalore_home_prices_model.pkl   ← trained Random Forest
│   └── columns.json                      ← 243 feature names
├── server/
│   ├── server.py                         ← Flask backend
│   ├── util.py                           ← load artifacts + predict_price()
│   └── templates/
│       └── index.html                    ← polished UI
├── train.py                              ← full training pipeline
└── README.md
```

## Run locally

```bash
pip install flask scikit-learn pandas numpy

cd bhp/server
python server.py
# open http://localhost:5000
```

## API endpoints

- `GET  /api/locations` — returns list of 240 Bengaluru localities
- `POST /api/predict` — returns price estimate
  ```json
  { "location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2 }
  ```
  Response:
  ```json
  {
    "estimated_price_lakh": 71.07,
    "estimated_price_inr": 7107000,
    "per_sqft": 5922
  }
  ```
