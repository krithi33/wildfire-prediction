# ========================================
# SIMPLE COPY-PASTE FOR PHASE 2 NOTEBOOK
# Add this as a NEW CELL at the end of Phase 2
# ========================================

"""
INSTRUCTIONS:
1. Open your Phase 2 notebook (02_feature_engineering.ipynb)
2. Scroll to the end
3. Add a new Markdown cell with title: "## Batch Export for Multiple Dates"
4. Add a new Code cell and paste the code below
5. Run it!
"""

# ========================================
# DATE RANGE SETUP
# ========================================

# Choose your date range (pick ONE):

# Option 1: One week (RECOMMENDED for first try)
start_date = '2020-08-18'
end_date = '2020-08-24'

# Option 2: One month (for portfolio)
# start_date = '2020-08-01'
# end_date = '2020-08-31'

# Option 3: Full fire season (best results, takes longer)
# start_date = '2020-06-01'
# end_date = '2020-10-31'

date_range = pd.date_range(start_date, end_date, freq='D')
print(f"Will export data for {len(date_range)} dates")

# ========================================
# SUBMIT EXPORTS
# ========================================

import time
from tqdm import tqdm

# You already have these functions from Phase 2, so we'll reuse them
# Just make sure these variables are defined:
# - grid_gdf (your grid)
# - CA_BBOX (bounding box)

tasks = []

for date in tqdm(date_range, desc="Submitting exports"):
    date_str = date.strftime('%Y-%m-%d')
    date_file = date.strftime('%Y_%m_%d')
    
    # WEATHER EXPORT
    weather_fc = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(date_str, ee.Date(date_str).advance(1, 'day')) \
        .filterBounds(ee.Geometry.Rectangle(CA_BBOX)) \
        .select(['temperature_2m', 'dewpoint_temperature_2m', 
                 'u_component_of_wind_10m', 'v_component_of_wind_10m', 
                 'total_precipitation'])
    
    weather_agg = ee.Image.cat([
        weather_fc.select('temperature_2m').mean().rename('temp_mean'),
        weather_fc.select('temperature_2m').max().rename('temp_max'),
        weather_fc.select('dewpoint_temperature_2m').mean().rename('dewpoint'),
        weather_fc.select('u_component_of_wind_10m').mean().pow(2)
            .add(weather_fc.select('v_component_of_wind_10m').mean().pow(2))
            .sqrt().rename('wind_speed'),
        weather_fc.select('total_precipitation').sum().rename('precip')
    ])
    
    grid_points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([row['lon_center'], row['lat_center']]), 
                   {'cell_id': row['cell_id']})
        for idx, row in grid_gdf.iterrows()
    ])
    
    weather_sampled = weather_agg.sampleRegions(
        collection=grid_points, scale=9000, geometries=False
    )
    
    task_w = ee.batch.Export.table.toDrive(
        collection=weather_sampled,
        description=f'weather_{date_file}',
        folder='wildfire_data_batch',
        fileFormat='CSV'
    )
    task_w.start()
    tasks.append(task_w)
    
    # NDVI EXPORT
    ndvi = ee.ImageCollection('MODIS/061/MOD13A1') \
        .filterDate(ee.Date(date_str).advance(-30, 'day'), date_str) \
        .filterBounds(ee.Geometry.Rectangle(CA_BBOX)) \
        .select('NDVI') \
        .sort('system:time_start', False) \
        .first() \
        .multiply(0.0001).rename('ndvi')
    
    ndvi_sampled = ndvi.sampleRegions(
        collection=grid_points, scale=500, geometries=False
    )
    
    task_n = ee.batch.Export.table.toDrive(
        collection=ndvi_sampled,
        description=f'ndvi_{date_file}',
        folder='wildfire_data_batch',
        fileFormat='CSV'
    )
    task_n.start()
    tasks.append(task_n)
    
    time.sleep(0.3)  # Small delay

print(f"\n✅ Submitted {len(tasks)} export tasks!")
print(f"\n📍 Next steps:")
print(f"1. Check https://code.earthengine.google.com/ → Tasks")
print(f"2. Wait for completion (10 min - 2 hours)")
print(f"3. Download from Google Drive → wildfire_data_batch folder")
print(f"4. Save to: data/multi_date/")
print(f"5. Run Phase 3 notebook!")

# ========================================
# THAT'S IT! 
# The exports are now running in the background.
# ========================================
