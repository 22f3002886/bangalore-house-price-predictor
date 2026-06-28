<div align="center">

# 🏠 Bangalore House Price Predictor

### End-to-end Machine Learning project that predicts property prices across Bangalore using a trained Random Forest model, deployed as a beautiful Flask web application.



[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<img src="https://img.shields.io/badge/Model-Random%20Forest-brightgreen?style=flat-square" />
<img src="https://img.shields.io/badge/Test%20R²-0.888-blue?style=flat-square" />
<img src="https://img.shields.io/badge/MAE-₹14.85L-orange?style=flat-square" />
<img src="https://img.shields.io/badge/Localities-240-purple?style=flat-square" />

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Example Predictions](#-example-predictions)
- [Model Evaluation](#-model-evaluation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This is a **complete, production-ready machine learning project** that predicts house prices in Bangalore. Built from scratch — from raw data to a deployed web application — it demonstrates the full lifecycle of a real-world ML project.

**What makes this different from a typical ML notebook:**
- ✅ Rigorous data cleaning with domain-specific outlier removal
- ✅ Honest model evaluation (tested multiple models, not just the best-looking one)
- ✅ Deployed as a real web application with a REST API
- ✅ Beautiful, responsive UI with animated price reveal
- ✅ 240 Bangalore localities with autocomplete search
- ✅ Input validation with domain-aware error messages

---

## 📊 Model Performance

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Algorithm** | Random Forest (100 trees) | Ensemble of decision trees |
| **Test R² Score** | **0.888** | Explains 89% of price variance |
| **Mean Absolute Error** | **₹14.85 Lakh** | Average prediction error |
| **Root Mean Squared Error** | **₹27.46 Lakh** | Penalizes large errors more |
| **5-Fold Cross-Validation R²** | **0.82** | Realistic generalization estimate |
| **Training Data** | 7,286 listings | After cleaning 13,320 raw rows |

> 💡 *The 5-fold CV R² of 0.82 is the most honest estimate — it tests the model on 5 different unseen splits, not just one.*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Search** | Autocomplete location search across 240 Bangalore localities |
| ⚡ **Real-time Predictions** | Instant price estimates under 1ms |
| 📱 **Responsive UI** | Works beautifully on desktop and mobile |
| 🛡️ **Input Validation** | Catches impossible property specs before predicting |
| 💰 **Detailed Output** | Price in Lakhs, full ₹ amount, and price per sqft |
| 🎨 **Animated UI** | Smooth count-up animation on price reveal |
| 🔌 **REST API** | Clean JSON API for integration with other tools |

---

## 📁 Project Structure

```
bangalore-house-price-predictor/
│
├── 📂 model/
│   ├── bangalore_home_prices_model.pkl   ← Trained Random Forest (48 MB)
│   └── columns.json                      ← 243 feature column names
│
├── 📂 server/
│   ├── server.py                         ← Flask backend (routes + API)
│   ├── util.py                           ← Model loading + predict_price()
│   └── 📂 templates/
│       └── index.html                    ← Single-page web UI
│
├── train.py                              ← Full training pipeline
├── requirements.txt                      ← Python dependencies
├── .gitignore                            ← Git ignore rules
├── LICENSE                               ← MIT License
└── README.md                             ← You are here
```

---

## 🏗️ ML Pipeline

The complete data science pipeline from raw CSV to deployed model:

```
Raw Data (13,320 rows)
        │
        ▼
┌──────────────────┐
│   Data Cleaning  │  Drop nulls, remove low-value columns
└────────┬─────────┘
         │  13,246 rows
         ▼
┌──────────────────────┐
│  Feature Engineering │  Extract BHK, fix sqft ranges, create price/sqft
└────────┬─────────────┘
         │  13,200 rows
         ▼
┌─────────────────────────┐
│  Dimensionality         │  1,305 locations → 240
│  Reduction (Location)   │  Rare locations grouped as "other"
└────────┬────────────────┘
         │
         ▼
┌─────────────────────┐
│  Outlier Removal    │  3 domain-logic rules:
│                     │  → sqft/BHK ratio check
│                     │  → price_per_sqft per location
│                     │  → BHK price coherence
└────────┬────────────┘
         │  7,286 rows (final clean dataset)
         ▼
┌─────────────────────┐
│  Model Training     │  LinearRegression, Lasso,
│  & Comparison       │  DecisionTree, RandomForest
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Save Artifacts     │  model.pkl + columns.json
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Flask Web App      │  REST API + Beautiful UI
└─────────────────────┘
```

### Data Cleaning Details

| Column | Action | Reason |
|--------|--------|--------|
| `society` | Dropped | 41% missing, 2,688 unique values — pure noise |
| `availability` | Dropped | 81 unique dates, not useful for prediction |
| `area_type` | Dropped | Low predictive value |
| `balcony` | Dropped | High missing rate |
| Null rows | Removed | Tiny count (< 100 rows) — safe to drop |

### Outlier Removal (Domain Logic)

Three domain-specific rules removed 45% noise:

1. **Sqft/BHK ratio** — Remove if `total_sqft / bhk < 300` (physically impossible)
2. **Price/sqft per location** — Remove if > 1σ from that location's mean (data errors)
3. **BHK price coherence** — Remove if 3BHK costs less than same area's 2BHK average

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML** | scikit-learn 1.8 | Model training + evaluation |
| **Data** | pandas 3.0, numpy | Data processing + feature engineering |
| **Backend** | Flask 3.1 | REST API server |
| **Frontend** | HTML / CSS / JavaScript | Web UI — no framework needed |
| **Fonts** | Space Grotesk, Space Mono | Blueprint-style typography |
| **Model** | Random Forest (100 trees) | Best accuracy + generalization |

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (comes with Python)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/22f3002886/bangalore-house-price-predictor.git
cd bangalore-house-price-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the server**
```bash
cd server
python server.py
```

**4. Open in browser**
```
http://localhost:5000
```

### Retrain the Model (Optional)

Want to retrain from scratch on the original dataset?
```bash
# From the project root (not server/)
python train.py
```
Takes ~30 seconds. Saves new `bangalore_home_prices_model.pkl` automatically.

---

## 📡 API Reference

### `GET /api/locations`

Returns all 240 supported Bangalore localities.

```bash
curl http://localhost:5000/api/locations
```

**Response:**
```json
{
  "locations": [
    "Whitefield",
    "Indira Nagar",
    "Koramangala",
    "Rajaji Nagar",
    "Electronic City Phase II"
  ]
}
```

---

### `POST /api/predict`

Predicts house price from property details.

**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"location": "Whitefield", "total_sqft": 1200, "bhk": 2, "bath": 2}'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `location` | string | Yes | Bangalore locality name |
| `total_sqft` | float | Yes | Total area in square feet |
| `bhk` | int | Yes | Number of bedrooms |
| `bath` | int | Yes | Number of bathrooms |

**Success Response `200`:**
```json
{
  "estimated_price_lakh": 71.07,
  "estimated_price_inr": 7107000,
  "per_sqft": 5922
}
```

**Error Response `400`:**
```json
{
  "error": "200 sqft for 3 BHK is unrealistically small (need ~300+ sqft per room)."
}
```

### Validation Rules

| Scenario | HTTP | Error Message |
|----------|------|---------------|
| `sqft / bhk < 300` | 400 | Unrealistically small property |
| `sqft <= 0` | 400 | Values must be greater than zero |
| Missing fields | 400 | Please provide valid values |
| Unknown location | 200 | Uses "other" (average area price) |

---

## 🏘️ Example Predictions

| Location | Area | BHK | Bath | Predicted Price | Per Sqft |
|----------|------|-----|------|----------------|----------|
| Whitefield | 1,200 sqft | 2 | 2 | **₹71.07L** | ₹5,922 |
| Indira Nagar | 1,500 sqft | 3 | 2 | **₹177.92L** | ₹11,861 |
| Rajaji Nagar | 3,000 sqft | 4 | 4 | **₹474.88L** | ₹15,829 |
| Electronic City | 900 sqft | 2 | 1 | **₹32.17L** | ₹3,574 |
| Hebbal | 1,800 sqft | 3 | 3 | **₹132.22L** | ₹7,346 |
| Koramangala | 2,200 sqft | 4 | 3 | **₹278.94L** | ₹12,679 |

---

## 🔬 Model Evaluation

### All Models Compared

| Model | Test R² | MAE | Train R² | Overfit Gap | Verdict |
|-------|---------|-----|----------|-------------|---------|
| **Random Forest** | **0.888** | **₹14.85L** | 0.964 | 0.076 | ✅ Winner |
| Linear Regression | 0.855 | ₹17.42L | 0.851 | −0.004 | ✅ Very stable |
| Decision Tree | 0.844 | ₹17.19L | 0.910 | 0.066 | ⚠️ Minor overfit |
| Lasso (α=1.0) | 0.684 | ₹23.96L | — | — | ❌ Under-regularized |

### Experiments That Failed (And Why)

| Experiment | Result | Conclusion |
|-----------|--------|------------|
| Log-transform price target | LinearRegression R² → −3422 | Discarded |
| Aggressive RF regularization | Test R² dropped with every constraint | Kept defaults |
| Lasso default alpha=1.0 | Only 0.68 R² — over-penalized | Dropped |

> 💡 *The RF train/test gap of 0.076 looks like overfitting but isn't — every attempt to reduce the gap by constraining the model made test accuracy worse. Default params generalize best here.*

---

## 🎨 UI Design

Blueprint/valuation-instrument aesthetic with:

- **Color palette:** Deep navy `#0B1A2C` + brass `#E0A23C` + cyan accent
- **Typography:** Space Grotesk (display) + Space Mono (numbers) + Inter (body)
- **Signature element:** Animated brass count-up price reveal
- **Grid background:** Blueprint crosshatch pattern on result panel

### How to Use the App

1. **Type a locality** — autocomplete suggests from 240 areas
2. **Enter the area** — total property size in square feet
3. **Select BHK** — click a button from 1 to 6+
4. **Select bathrooms** — click a button from 1 to 6+
5. **Click "Estimate value"** — or just press Enter
6. **Watch the price** animate onto the screen with full breakdown

---

## 🤝 Contributing

Contributions are welcome! Some ideas to get started:

- 🚀 Deploy to Heroku, Railway, or Render
- 📈 Add price trend visualizations
- 🗺️ Add map view of Bangalore price zones
- 🤖 Add XGBoost for potentially higher accuracy
- 🧪 Write unit tests with pytest
- 📱 Build a mobile app with React Native

**How to contribute:**
```bash
# Fork → Clone → Branch → Code → PR
git checkout -b feature/your-idea
git commit -m "Add your feature"
git push origin feature/your-idea
# Open a Pull Request on GitHub
```

---

## 📋 Requirements

```
Flask==3.1.3
scikit-learn==1.8.0
pandas==3.0.2
numpy>=1.24.0
```

---

## 🗺️ Roadmap

- [x] Full data cleaning pipeline
- [x] Feature engineering
- [x] Domain-logic outlier removal
- [x] Multi-model training and comparison
- [x] Flask REST API
- [x] Beautiful responsive UI
- [x] Input validation
- [ ] Cloud deployment
- [ ] XGBoost integration
- [ ] Confidence intervals on predictions
- [ ] Historical price trends
- [ ] Docker support

---

## 📄 License

This project is open source under the [MIT License](LICENSE) — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- Dataset from Kaggle Bangalore House Prices
- Built with scikit-learn, Flask, pandas, numpy
- Fonts from Google Fonts

---

<div align="center">

**Made with ❤️ by [Abhishek Kumar](https://github.com/22f3002886)**

⭐ **Star this repo if it helped you!** ⭐

</div>
