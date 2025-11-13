import xarray as xr
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# List of cities and their coordinates (city center lat/lon)
cities = pd.DataFrame([
    {'city': 'Ahmedabad', 'latitude': 23.0225, 'longitude': 72.5714},
    {'city': 'Aizawl', 'latitude': 23.7271, 'longitude': 92.7176},
    {'city': 'Amaravati', 'latitude': 16.5417, 'longitude': 80.5150},
    {'city': 'Amritsar', 'latitude': 31.6340, 'longitude': 74.8723},
    {'city': 'Bengaluru', 'latitude': 12.9716, 'longitude': 77.5946},
    {'city': 'Bhopal', 'latitude': 23.2599, 'longitude': 77.4126},
    {'city': 'Brajrajnagar', 'latitude': 21.8167, 'longitude': 83.9167},
    {'city': 'Chandigarh', 'latitude': 30.7333, 'longitude': 76.7794},
    {'city': 'Chennai', 'latitude': 13.0827, 'longitude': 80.2707},
    {'city': 'Coimbatore', 'latitude': 11.0168, 'longitude': 76.9558},
    {'city': 'Delhi', 'latitude': 28.7041, 'longitude': 77.1025},
    {'city': 'Ernakulam', 'latitude': 9.9816, 'longitude': 76.2999},
    {'city': 'Gurugram', 'latitude': 28.4595, 'longitude': 77.0266},
    {'city': 'Guwahati', 'latitude': 26.1445, 'longitude': 91.7362},
    {'city': 'Hyderabad', 'latitude': 17.3850, 'longitude': 78.4867},
    {'city': 'Jaipur', 'latitude': 26.9124, 'longitude': 75.7873},
    {'city': 'Jorapokhar', 'latitude': 23.7833, 'longitude': 86.4167},
    {'city': 'Kochi', 'latitude': 9.9312, 'longitude': 76.2673},
    {'city': 'Kolkata', 'latitude': 22.5726, 'longitude': 88.3639},
    {'city': 'Lucknow', 'latitude': 26.8467, 'longitude': 80.9462},
    {'city': 'Mumbai', 'latitude': 19.0760, 'longitude': 72.8777},
    {'city': 'Patna', 'latitude': 25.5941, 'longitude': 85.1376},
    {'city': 'Shillong', 'latitude': 25.5788, 'longitude': 91.8933},
    {'city': 'Talcher', 'latitude': 20.9500, 'longitude': 85.2333},
    {'city': 'Thiruvananthapuram', 'latitude': 8.5241, 'longitude': 76.9366},
    {'city': 'Visakhapatnam', 'latitude': 17.6868, 'longitude': 83.2185},
])

MAX_KM = 40  # radius from city centers

def to_xy(lat, lon):
    """Convert lat/lon to planar x/y (in km) using equirectangular approximation."""
    R = 6371  # Earth radius in km
    x = np.radians(lon) * R * np.cos(np.radians(lat))
    y = np.radians(lat) * R
    return np.column_stack((x, y))


def filter_nc_by_city(nc_file, out_nc, out_csv, max_km=MAX_KM):
    # Load NetCDF into an xarray Dataset
    ds = xr.open_dataset(nc_file)

    # Get coordinate values
    lons = ds['lon'].values
    lats = ds['lat'].values

    # Create 2D meshgrid of all lon/lat pairs
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Convert city and grid coordinates into x/y for distance calculation
    city_xy = to_xy(cities['latitude'].values, cities['longitude'].values)
    data_xy = to_xy(lat_grid.ravel(), lon_grid.ravel())

    # Build KDTree for efficient nearest-city lookup
    city_tree = cKDTree(city_xy)

    # Query nearest city for each grid point
    dist, idx = city_tree.query(data_xy, k=1)

    # Assign city names or NaN if beyond radius
    city_names = np.where(dist <= max_km, cities.iloc[idx]['city'].values, np.nan)

    # Mask points within the city radius
    mask = ~pd.isna(city_names)

    # Convert mask and city names into DataArrays with correct coordinates
    mask_da = xr.DataArray(
        mask.reshape(lat_grid.shape),
        coords={'lat': lats, 'lon': lons},
        dims=('lat', 'lon')
    )

    city_da = xr.DataArray(
        city_names.reshape(lat_grid.shape),
        coords={'lat': lats, 'lon': lons},
        dims=('lat', 'lon')
    )

    # Apply mask to dataset
    ds_filtered = ds.where(mask_da, drop=True)

    # Add city names to dataset
    ds_filtered['city'] = city_da.where(mask_da)
    # check dtype and values that will be written
    print("city dtype:", ds_filtered['city'].dtype)
    print("city numpy dtype:", ds_filtered['city'].values.dtype)
    print("sample city values:", pd.Series(ds_filtered['city'].values.ravel()).dropna().unique()[:10])
    # check disk and permissions (Windows)
    import shutil, os
    print("free bytes:", shutil.disk_usage('.').free)
    print("can write out file:", os.access(out_nc, os.W_OK) or not os.path.exists(out_nc))
    
    # Save filtered dataset to NetCDF
    ds_filtered.to_netcdf(out_nc)

    # Convert to DataFrame & save CSV
    df = ds_filtered.to_dataframe().reset_index()
    df = df.dropna(subset=['city'])
    df.to_csv(out_csv, index=False)

    print(f"✅ Saved filtered NetCDF: {out_nc}")
    print(f"✅ Saved filtered CSV: {out_csv}")


# Example usage
filter_nc_by_city('india_subset_uv_sw.nc', 'filtered_uv_sw.nc', 'filtered_uv_sw.csv')