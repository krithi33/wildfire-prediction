# ========================================
# MULTI-DATE BATCH EXPORT CODE
# Add this to Phase 2 notebook (after single date extraction)
# ========================================

import ee
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Initialize Earth Engine
ee.Initialize()

# Load grid and static data
grid_gdf = gpd.read_parquet('../data/grid_10km.parquet')
CA_BBOX = [-124.5, 32.5, -114.0, 42.0]

print("✅ Setup complete")
print(f"Grid: {len(grid_gdf)} cells")

# ========================================
# STEP 1: Define Date Range
# ========================================

# Option A: One week (for testing)
start_date = '2020-08-18'
end_date = '2020-08-24'

# Option B: One month (recommended for portfolio)
# start_date = '2020-08-01'
# end_date = '2020-08-31'

# Option C: Full fire season (best results)
# start_date = '2020-06-01'
# end_date = '2020-10-31'

# Generate date range
date_range = pd.date_range(start_date, end_date, freq='D')

print(f"\n📅 Will process {len(date_range)} dates:")
print(f"   From: {date_range[0].strftime('%Y-%m-%d')}")
print(f"   To:   {date_range[-1].strftime('%Y-%m-%d')}")

# ========================================
# STEP 2: Create Extraction Functions
# ========================================

def extract_weather_for_date(date_str, grid_cells, bbox):
    """
    Extract weather features for a specific date.
    Returns: ee.FeatureCollection
    """
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(date_str, ee.Date(date_str).advance(1, 'day')) \
        .filterBounds(ee.Geometry.Rectangle(bbox))
    
    weather = era5.select([
        'temperature_2m',
        'dewpoint_temperature_2m',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m',
        'total_precipitation'
    ])
    
    # Daily aggregates
    temp_mean = weather.select('temperature_2m').mean()
    temp_max = weather.select('temperature_2m').max()
    dewpoint_mean = weather.select('dewpoint_temperature_2m').mean()
    
    u = weather.select('u_component_of_wind_10m').mean()
    v = weather.select('v_component_of_wind_10m').mean()
    wind_speed = u.pow(2).add(v.pow(2)).sqrt()
    
    precip = weather.select('total_precipitation').sum()
    
    weather_composite = ee.Image.cat([
        temp_mean.rename('temp_mean'),
        temp_max.rename('temp_max'),
        dewpoint_mean.rename('dewpoint'),
        wind_speed.rename('wind_speed'),
        precip.rename('precip')
    ])
    
    # Create points for sampling
    grid_points = [
        ee.Feature(
            ee.Geometry.Point([row['lon_center'], row['lat_center']]),
            {'cell_id': row['cell_id']}
        )
        for idx, row in grid_cells.iterrows()
    ]
    grid_fc = ee.FeatureCollection(grid_points)
    
    sampled = weather_composite.sampleRegions(
        collection=grid_fc,
        scale=9000,
        geometries=False
    )
    
    return sampled


def extract_ndvi_for_date(date_str, grid_cells, bbox):
    """
    Extract NDVI for a specific date.
    Returns: ee.FeatureCollection
    """
    ndvi_collection = ee.ImageCollection('MODIS/061/MOD13A1') \
        .filterDate(
            ee.Date(date_str).advance(-30, 'day'),
            date_str
        ) \
        .filterBounds(ee.Geometry.Rectangle(bbox)) \
        .select('NDVI')
    
    ndvi_image = ndvi_collection.sort('system:time_start', False).first()
    ndvi_scaled = ndvi_image.multiply(0.0001).rename('ndvi')
    
    grid_points = [
        ee.Feature(
            ee.Geometry.Point([row['lon_center'], row['lat_center']]),
            {'cell_id': row['cell_id']}
        )
        for idx, row in grid_cells.iterrows()
    ]
    grid_fc = ee.FeatureCollection(grid_points)
    
    sampled = ndvi_scaled.sampleRegions(
        collection=grid_fc,
        scale=500,
        geometries=False
    )
    
    return sampled

print("\n✅ Extraction functions defined")

# ========================================
# STEP 3: Submit Export Tasks (All Dates)
# ========================================

print("\n" + "="*60)
print("STARTING BATCH EXPORTS")
print("="*60)
print("\nThis will submit export tasks to Google Earth Engine.")
print("Tasks run asynchronously - you can close this notebook.")
print("Check progress at: https://code.earthengine.google.com/ → Tasks\n")

# Ask for confirmation
proceed = input("Submit exports for all dates? (yes/no): ")

if proceed.lower() != 'yes':
    print("❌ Cancelled. Change date range and try again.")
else:
    # Submit all export tasks
    tasks_submitted = []
    
    for date in tqdm(date_range, desc="Submitting exports"):
        date_str = date.strftime('%Y-%m-%d')
        date_filename = date.strftime('%Y_%m_%d')
        
        try:
            # Weather export
            weather_fc = extract_weather_for_date(date_str, grid_gdf, CA_BBOX)
            
            task_weather = ee.batch.Export.table.toDrive(
                collection=weather_fc,
                description=f'weather_{date_filename}',
                folder='wildfire_data_batch',
                fileFormat='CSV'
            )
            task_weather.start()
            tasks_submitted.append(('weather', date_str, task_weather))
            
            # NDVI export
            ndvi_fc = extract_ndvi_for_date(date_str, grid_gdf, CA_BBOX)
            
            task_ndvi = ee.batch.Export.table.toDrive(
                collection=ndvi_fc,
                description=f'ndvi_{date_filename}',
                folder='wildfire_data_batch',
                fileFormat='CSV'
            )
            task_ndvi.start()
            tasks_submitted.append(('ndvi', date_str, task_ndvi))
            
            # Small delay to avoid overwhelming API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n⚠️  Error submitting {date_str}: {e}")
    
    print(f"\n✅ All export tasks submitted!")
    print(f"   Total tasks: {len(tasks_submitted)}")
    print(f"   Weather exports: {len([t for t in tasks_submitted if t[0] == 'weather'])}")
    print(f"   NDVI exports: {len([t for t in tasks_submitted if t[0] == 'ndvi'])}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("\n1. ⏳ Wait for exports to complete (10 min - 2 hours)")
    print("   • Check status: https://code.earthengine.google.com/ → Tasks")
    print("   • All tasks should show 'Completed' status")
    
    print("\n2. 📥 Download files from Google Drive:")
    print("   • Go to: https://drive.google.com/")
    print("   • Open folder: wildfire_data_batch")
    print("   • Download all CSVs (or entire folder)")
    
    print("\n3. 💾 Save to project directory:")
    print("   • Create folder: data/multi_date/")
    print("   • Move all downloaded CSVs there")
    print("   • Files should be named: weather_2020_08_18.csv, ndvi_2020_08_18.csv, etc.")
    
    print("\n4. 🚀 Run Phase 3 notebook to process and model!")
    
    print("\n" + "="*60)

# ========================================
# OPTIONAL: Monitor Progress
# ========================================

monitor = input("\nMonitor export progress? (yes/no): ")

if monitor.lower() == 'yes' and len(tasks_submitted) > 0:
    print("\n📊 Monitoring export progress...")
    print("(Press Ctrl+C to stop monitoring)\n")
    
    try:
        while True:
            statuses = {}
            for task_type, date_str, task in tasks_submitted:
                status = task.status()['state']
                if status not in statuses:
                    statuses[status] = 0
                statuses[status] += 1
            
            print(f"Status: ", end='')
            for state, count in sorted(statuses.items()):
                print(f"{state}={count}  ", end='')
            print("", end='\r')
            
            # Check if all complete
            if 'RUNNING' not in statuses and 'READY' not in statuses:
                print("\n\n✅ All exports completed!")
                print("\nFinal status:")
                for state, count in sorted(statuses.items()):
                    print(f"   {state}: {count} tasks")
                break
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Monitoring stopped (tasks still running in background)")
        print("Check status at: https://code.earthengine.google.com/")

print("\n" + "="*60)
print("BATCH EXPORT COMPLETE!")
print("="*60)
