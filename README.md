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

## Project Structure

```
wildfire_prediction/
├── data/               # Downloaded/processed data (gitignored)
│   ├── raw/           # Original downloads
│   ├── processed/     # Cleaned/engineered features
│   └── grid/          # Spatial grid definitions
├── notebooks/         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/               # Reusable Python modules
│   ├── data_pipeline.py      # GEE data extraction
│   ├── feature_engineering.py # Create features from raw data
│   ├── model.py              # Train/evaluate models
│   └── utils.py              # Helper functions
├── docs/              # Documentation and learning materials
│   ├── LEARNING_GUIDE.md     # Detailed explanations of each step
│   └── GEE_SETUP.md         # Google Earth Engine setup instructions
├── requirements.txt   # Python dependencies
└── README.md         # This file
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

### Phase 1: Data Collection (Week 1)

We'll start by:
1. Defining a study area (e.g., California)
2. Creating a 1km spatial grid
3. Extracting fire locations from FIRMS
4. Pulling weather and vegetation data from GEE

**Start here**: `notebooks/01_data_exploration.ipynb`

---

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
5. ✅ A **deployable pipeline** (optional: FastAPI service)

---

## Timeline

- **Week 1**: Data pipeline + EDA
- **Week 2**: Feature engineering + baseline model
- **Week 3**: Model tuning + spatial validation
- **Week 4**: Visualization + documentation

---

## Next Steps

1. Read `docs/GEE_SETUP.md` to configure Earth Engine
2. Open `notebooks/01_data_exploration.ipynb` to start
3. Follow `docs/LEARNING_GUIDE.md` for detailed explanations

---

## References

- [Earth Engine Python API](https://developers.google.com/earth-engine/guides/python_install)
- [FIRMS Data](https://firms.modaps.eosdis.nasa.gov/descriptions/FIRMS_products.html)
- [Handling Imbalanced Data](https://www.jmlr.org/papers/volume20/18-145/18-145.pdf)
- [Spatial Cross-Validation](https://doi.org/10.1016/j.ecolmodel.2012.02.014)

---

