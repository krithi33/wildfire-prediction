# Google Earth Engine Setup Guide

## What is Google Earth Engine?

Google Earth Engine (GEE) is a cloud-based platform for planetary-scale geospatial analysis. Instead of downloading terabytes of satellite imagery to your computer, you:

1. Write code that references datasets in GEE's catalog
2. Send that code to Google's servers
3. Computation happens in the cloud
4. Only results are sent back to you

**Analogy**: It's like SQL for satellite imagery — you query the data you need, and the heavy lifting happens server-side.

---

## Why Use GEE for This Project?

**Without GEE**: 
- Download 10+ years of MODIS data (~500 GB)
- Download ERA5 weather data (~200 GB)
- Process locally (hours/days on a laptop)

**With GEE**:
- Stream only the data you need
- Processing happens on Google's infrastructure
- Get results in minutes

---

## Installation Steps

### 1. Create a Google Account (if you don't have one)

Go to https://accounts.google.com/signup

### 2. Sign Up for Earth Engine

1. Go to https://signup.earthengine.google.com/
2. Select "Register a Noncommercial or Commercial Cloud project"
3. Choose "Unpaid usage" (free tier is generous for learning projects)
4. Fill in project details (e.g., "Wildfire Prediction ML Project")
5. Wait for approval (usually instant, sometimes takes 1-2 days)

### 3. Install the Python API

```bash
pip install --break-system-packages earthengine-api
```

**What this does**: Installs the `ee` (Earth Engine) Python library.

### 4. Authenticate

```bash
earthengine authenticate
```

**What happens**:
1. Opens browser window
2. You log in with your Google account
3. You grant Earth Engine permission
4. A token is saved locally (usually `~/.config/earthengine/credentials`)

**Note**: You only need to do this once per machine.

### 5. Test Your Setup

Open Python and run:

```python
import ee

# Initialize the Earth Engine library
ee.Initialize()

# Test query: Get elevation at a point
point = ee.Geometry.Point([-122.262, 37.872])  # Berkeley, CA
elevation = ee.Image('USGS/SRTMGL1_003').sample(point, 30).first().get('elevation').getInfo()

print(f"Elevation at Berkeley: {elevation} meters")
```

**Expected output**: `Elevation at Berkeley: 100 meters` (approximately)

If this runs without errors, you're ready to go! ✅

---

## Understanding GEE Concepts

### Image vs ImageCollection

- **Image**: A single raster (e.g., one MODIS scene from Jan 1, 2020)
- **ImageCollection**: A stack of images (e.g., all MODIS scenes from 2015-2024)

```python
# Single image
image = ee.Image('USGS/SRTMGL1_003')  # Elevation

# Collection of images
collection = ee.ImageCollection('MODIS/006/MOD13A1')  # NDVI time series
```

### Geometry

Defines a spatial region:

```python
# Point
point = ee.Geometry.Point([-120.5, 38.5])

# Bounding box
bbox = ee.Geometry.Rectangle([-124, 32, -114, 42])  # California

# Polygon
polygon = ee.Geometry.Polygon([[
    [-122, 37], [-122, 38], [-121, 38], [-121, 37]
]])
```

### Feature / FeatureCollection

Geometries with properties (like a GeoJSON):

```python
# Single feature
feature = ee.Feature(point, {'name': 'Fire Station 1'})

# Collection of features
features = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([-120, 38]), {'type': 'fire'}),
    ee.Feature(ee.Geometry.Point([-121, 39]), {'type': 'fire'})
])
```

---

## Common GEE Operations

### 1. Filter by Date

```python
# Get all MODIS NDVI images from 2020
ndvi_2020 = ee.ImageCollection('MODIS/006/MOD13A1') \
    .filterDate('2020-01-01', '2020-12-31')
```

### 2. Filter by Bounds

```python
# Get images overlapping California
california = ee.Geometry.Rectangle([-124, 32, -114, 42])
ndvi_ca = ee.ImageCollection('MODIS/006/MOD13A1') \
    .filterBounds(california)
```

### 3. Select a Band

MODIS images have multiple bands (red, blue, NIR, etc.). Select one:

```python
# NDVI is stored in the 'NDVI' band
ndvi_band = ee.ImageCollection('MODIS/006/MOD13A1').select('NDVI')
```

### 4. Reduce (Aggregate)

Compute statistics over time:

```python
# Mean NDVI across 2020
mean_ndvi = ee.ImageCollection('MODIS/006/MOD13A1') \
    .filterDate('2020-01-01', '2020-12-31') \
    .select('NDVI') \
    .mean()  # Average all images in collection
```

### 5. Extract Values at Points

```python
# Get temperature at fire locations
fires = ee.FeatureCollection([...])  # Your fire points
temp = ee.Image('ECMWF/ERA5_LAND/HOURLY').select('temperature_2m')

# Sample temperature at each fire point
fire_temps = temp.sampleRegions(
    collection=fires,
    scale=1000  # Resolution in meters
)
```

---

## Exporting Data from GEE

GEE processes data in the cloud, but you need to export results for local ML training.

### Option 1: Export to Google Drive

```python
task = ee.batch.Export.table.toDrive(
    collection=fire_temps,
    description='fire_temperatures',
    fileFormat='CSV'
)
task.start()
```

**What happens**:
1. Task runs asynchronously on GEE servers
2. Results are saved to your Google Drive
3. You download the CSV manually

### Option 2: Download Directly (Small Data Only)

```python
# Only works for small datasets (<5000 rows)
data = fire_temps.getInfo()  # Returns Python dict
import pandas as pd
df = pd.DataFrame(data['features'])
```

**Warning**: `.getInfo()` times out for large datasets. Use export for production.

---

## Key Datasets We'll Use

| Dataset | GEE ID | What It Contains |
|---------|--------|------------------|
| MODIS NDVI | `MODIS/006/MOD13A1` | Vegetation index (500m, 16-day) |
| MODIS Fires | `MODIS/006/MOD14A1` | Active fire detections (1km, daily) |
| ERA5 Weather | `ECMWF/ERA5_LAND/HOURLY` | Temp, humidity, wind, precip (9km, hourly) |
| SRTM Elevation | `USGS/SRTMGL1_003` | Terrain elevation (30m, static) |

---

## Debugging Tips

### Error: "User memory limit exceeded"

**Cause**: Trying to process too much data at once.

**Fix**: 
- Reduce date range
- Use `.limit(1000)` to process fewer images
- Increase `.scale` parameter (lower resolution = less data)

### Error: "Collection.getInfo() timeout"

**Cause**: Dataset too large to download directly.

**Fix**: Use `.export.table.toDrive()` instead.

### Error: "Image has no band 'NDVI'"

**Cause**: Band name typo or selecting wrong dataset.

**Fix**: Check dataset documentation at https://developers.google.com/earth-engine/datasets

---

## Next Steps

Once GEE is set up, proceed to:
1. `notebooks/01_data_exploration.ipynb` — Pull fire data and visualize
2. `src/data_pipeline.py` — Automate data extraction

---

## Additional Resources

- [Earth Engine Python API Docs](https://developers.google.com/earth-engine/guides/python_install)
- [Dataset Catalog](https://developers.google.com/earth-engine/datasets)
- [Code Examples](https://github.com/google/earthengine-api/tree/master/python/examples)
- [Forum](https://groups.google.com/g/google-earth-engine-developers) — Ask questions here

---

**Ready to pull planetary-scale data! 🌍**
