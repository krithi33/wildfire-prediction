# 🔥 Wildfire Risk Prediction Using Geospatial ML

## Project Overview

This project predicts wildfire risk at a 1km grid resolution for 7-14 days in advance using:
- **Historical fire data** from NASA FIRMS (Fire Information for Resource Management System)
- **Weather data** from ERA5 or Google Earth Engine
- **Vegetation indices** from MODIS/Sentinel via Google Earth Engine
- **Topographic data** from SRTM elevation models
- **Machine learning** with gradient boosting and optional deep learning

**Goal**: Build an operational wildfire early warning system that balances precision and recall for rare fire events.

---

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://wildfire-prediction-cali.streamlit.app)

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUPRC** | **0.311** | 54× better than random guessing |
| **Recall @ 0.5** | **48%** | Catches nearly half of all fires |
| **Precision @ 0.5** | **33%** | 1 in 3 predictions are correct |

## 🛰️ Data & Methodology

### Multi-Modal Data Fusion

Satellite Imagery (MODIS) + Weather Reanalysis (ERA5) + Terrain (SRTM)
↓                        ↓                        ↓
Vegetation (NDVI)         Temperature, Wind       Elevation, Slope
↓
Feature Engineering (25 features)
• 7-day lags & rolling windows
• Cyclical time encoding
• Topographic interactions
↓
LightGBM Classifier
• Class weighting (99:1)
• Temporal validation
↓
7-Day Ahead Predictions

## Why Google Earth Engine?

**Google Earth Engine (GEE)** is a planetary-scale geospatial analysis platform that provides:

1. **Petabytes of satellite imagery** - MODIS, Landsat, Sentinel, etc.
2. **Pre-processed climate data** - Temperature, precipitation, humidity
3. **Cloud computing** - No need to download TBs of data locally
4. **Python/JavaScript API** - Easy integration with ML pipelines

**For this project, we use GEE for**:
- MODIS NDVI (vegetation greenness) - 250m resolution, 16-day composite
- MODIS Fire Product (FIRMS) - Active fire detections
- ERA5 weather reanalysis - Temperature, humidity, wind, precipitation
- SRTM elevation data - Slope, aspect, terrain features

**Alternative**: You can also use downloaded datasets, but GEE is faster for prototyping.

---

### Technology Stack

```python
Data: Google Earth Engine, MODIS, ERA5, SRTM
ML: LightGBM, Scikit-learn, Pandas, NumPy
Viz: Streamlit, Plotly, Folium, Matplotlib
Geo: GeoPandas, Shapely, Earth Engine API
```

## 🔑 Key Findings


### 1. Topography Dominates (47% importance)
- **Elevation**: Fire regimes vary by altitude
- **Slope**: Fires spread 2-3× faster uphill
- **Aspect**: South-facing slopes dry faster

### 2. Temporal Patterns Matter
- 7-day precipitation history crucial
- Temperature lags capture heat waves
- Rolling averages identify trends

### 3. Operational Flexibility
- **Threshold 0.2**: Evacuation mode (high recall)
- **Threshold 0.5**: Balanced (recommended)
- **Threshold 0.7**: High confidence only

## Project Structure

```
wildfire_prediction/
├── app.py                    
├── requirements.txt          
├── data/
│   └── predictions_for_dashboard.csv  # ← Your data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_multi_date_modeling.ipynb
├── docs/
│   ├── DASHBOARD_EXPLAINED.md
│   └── ...
└── README.md
```

---

## Key Technical Concepts (What You'll Learn)

### 1. **Geospatial Data Handling**
- Working with rasters (gridded satellite imagery)
- Coordinate reference systems (CRS) and projections
- Spatial joins and aggregations
- Handling multi-dimensional arrays (time × lat × lon × bands)

### 2. **Time-Series Features for Spatial Data**
- Lag features (yesterday's fire → today's risk)
- Rolling statistics (7-day average temperature)
- Cyclical encoding (day-of-year as sin/cos)
- Autoregressive patterns in grid cells

### 3. **Class Imbalance**
- Fires occur in <1% of grid cells → extreme imbalance
- Techniques: class weighting, undersampling, focal loss
- Metrics: AUPRC (not AUROC), precision at high recall

### 4. **Spatial Cross-Validation**
- Why random splits fail (spatial autocorrelation)
- Leave-region-out validation
- Temporal + spatial splits

### 5. **Feature Importance & Interpretability**
- SHAP values for tree models
- Partial dependence plots
- Feature interaction detection

---

## Installation & Setup

### Step 1: Install Dependencies

```bash
pip install --break-system-packages earthengine-api geopandas rasterio xarray pandas numpy scikit-learn lightgbm matplotlib seaborn folium jupyter
```

### Step 2: Authenticate Google Earth Engine

```bash
earthengine authenticate
```

This will open a browser window for you to log in with your Google account and authorize Earth Engine access.

### Step 3: Initialize Earth Engine in Python

```python
import ee
ee.Initialize()
```

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/wildfire-prediction
cd wildfire-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Authenticate with Google Earth Engine
import ee
ee.Authenticate()

# 4. Run notebooks in order or launch dashboard
streamlit run app.py
```

## 📈 Performance Analysis

<table>
<tr>
<td width="50%">

### Threshold Analysis

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.1 | 9% | 76% | Maximum safety |
| 0.2 | 14% | 68% | **Evacuations** ✅ |
| 0.3 | 19% | 58% | Early warnings |
| 0.5 | 33% | 48% | **Balanced** ⭐ |
| 0.7 | 45% | 31% | High confidence |
| 0.9 | 64% | 10% | Insurance only |

</td>
<td width="50%">

### Model Comparison

| Approach | AUPRC | Notes |
|----------|-------|-------|
| Random Baseline | 0.006 | Fire rate |
| Logistic Regression | 0.124 | Linear only |
| Random Forest | 0.267 | Good but slow |
| **LightGBM** | **0.311** | ✅ Best |
| XGBoost | 0.298 | Similar |

</td>
</tr>
</table>

## 🎓 Technical Deep Dive

### Handling Class Imbalance (99:1 ratio)

**Challenge**: Only 1% of samples are fires

**Solutions**:
1. ✅ Class weighting (`scale_pos_weight=99`)
2. ✅ AUPRC metric (not accuracy)
3. ✅ Threshold tuning for deployment
4. ✅ Temporal validation (no leakage)

### Temporal Validation Strategy

```python
# ❌ BAD: Random split (spatial autocorrelation leakage)
train_test_split(X, y, test_size=0.2)

# ✅ GOOD: Temporal split
train = data[data['date'] < '2020-08-25']
test = data[data['date'] >= '2020-08-25']
```

## Future Enhancements

- [ ] Scale to 1km resolution (400K cells vs current 4K)
- [ ] Add fire spread simulation (Rothermel model)
- [ ] Real-time daily updates (automated pipeline)
- [ ] Ensemble models (LightGBM + XGBoost + Neural Network)
- [ ] Multi-region support (Australia, Mediterranean, Amazon)
- [ ] Mobile app with push notifications

## Learning Approach

This project is designed to be **study material**. Each notebook and script includes:

✅ **Detailed comments** explaining what each function does  
✅ **Why decisions were made** (e.g., why use AUPRC instead of accuracy?)  
✅ **Common pitfalls** and how to avoid them  
✅ **Visualization** of intermediate steps  
✅ **References** to papers/documentation for deeper learning  

**Philosophy**: Don't just run the code — understand why it works.

---

## Dataset Details

### Fire Data: NASA FIRMS
- **Source**: https://firms.modaps.eosdis.nasa.gov/
- **Resolution**: 375m (VIIRS), 1km (MODIS)
- **Coverage**: 2000-present (MODIS), 2012-present (VIIRS)
- **What it contains**: Fire locations, brightness temperature, fire radiative power

### Weather: ERA5-Land
- **Source**: Google Earth Engine (`ECMWF/ERA5_LAND/HOURLY`)
- **Resolution**: 9km (can interpolate to 1km)
- **Variables**: Temperature, precipitation, humidity, wind speed/direction
- **Coverage**: 1950-present

### Vegetation: MODIS NDVI
- **Source**: Google Earth Engine (`MODIS/006/MOD13A1`)
- **Resolution**: 500m
- **Frequency**: 16-day composite
- **What it measures**: Vegetation greenness (proxy for fuel availability)

### Topography: SRTM
- **Source**: Google Earth Engine (`USGS/SRTMGL1_003`)
- **Resolution**: 30m
- **What it contains**: Elevation (we derive slope and aspect)

---

## Expected Outcomes

By the end of this project, you will have:

1. ✅ A trained model with **AUPRC > 0.3** (baseline is ~0.01 for random)
2. ✅ **Feature importance** showing temperature and NDVI as top predictors
3. ✅ **Spatial visualizations** of predicted risk vs actual fires
4. ✅ **Residual analysis** showing where/when model fails
5. ✅ A **deployable pipeline**

---

## Timeline

- **Week 1**: Data pipeline + EDA
- **Week 2**: Feature engineering + baseline model
- **Week 3**: Model tuning + spatial validation
- **Week 4**: Visualization + documentation


---

## References

- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [FIRMS Data](https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_products.html)
- [Handling Imbalanced Data](https://www.jmlr.org/papers/volume20/18-145/18-145.pdf)
- [Spatial Cross-Validation](https://doi.org/10.1016/j.ecolmodel.2012.02.014)

---

