import xarray as xr
import pandas as pd


# Load multiple NetCDF files
ds = xr.open_mfdataset(
    "met_uv_sw/*.nc",
    engine="h5netcdf",
    combine="by_coords",
    parallel=False,
    join="override"
)
print(ds.lat.values[:10])     # first few lats
print(ds.lat.values[-10:])    # last few lats

# Define India's bounding box
lat_min, lat_max = 6.0, 37.1
lon_min, lon_max = 68.7, 97.25

# Check the order of lat/lon the dataset uses
india = ds.sel(lat=slice(lat_min, lat_max),
               lon=slice(lon_min, lon_max))

print("Subset dimensions:", india.sizes)
print("Lat range:", india.lat.values.min(), "→", india.lat.values.max())
print("Lon range:", india.lon.values.min(), "→", india.lon.values.max())


# Compute in-memory (avoid Dask write conflicts)
india = india.compute()


# Clean coordinate dtypes & attributes
# Ensure lat/lon are floats
india = india.assign_coords(
    lat=india.lat.astype(float),
    lon=india.lon.astype(float)
)

# Strip any problematic attributes
for v in india.variables:
    india[v].attrs = {}
india.attrs = {}


# Write a clean NetCDF subset
india.to_netcdf("india_subset_uv_sw.nc", engine="h5netcdf")


print("✅ Saved india_subset_uv_sw.nc successfully!")
