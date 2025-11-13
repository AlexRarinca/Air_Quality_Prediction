import pandas as pd

# === CONFIGURATION ===
csv_in = "filtered_uv_sw.csv"          # input hourly CSV
csv_out = "city_daily_means_uv_sw.csv"     # output daily-averaged CSV
EXPECTED_HOURS = 24                  # set to None to skip filtering incomplete days

# === LOAD ===
print(f"Loading {csv_in} ...")
df = pd.read_csv(csv_in, parse_dates=["time"])
df.columns = df.columns.str.strip()  # clean any stray spaces in headers

# Ensure required columns exist
required = {"time", "city"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Determine which numeric variables to average
# Explicitly prefer these if present, otherwise average all numeric (excluding lat/lon)
preferred_vars = [c for c in ["PSurf", "Rainf"] if c in df.columns]
numeric_cols = df.select_dtypes("number").columns.tolist()
value_cols = preferred_vars if preferred_vars else [c for c in numeric_cols if c not in ("lat", "lon")]

if not value_cols:
    raise ValueError("No numeric value columns found to average.")

print(f"Will average: {value_cols}")

# Add date column (midnight-aligned day)
df["date"] = df["time"].dt.normalize()

# === DAILY AVERAGES PER CITY ===
agg_spec = {c: "mean" for c in value_cols}

# Optional: include a representative lat/lon per city-day
# Change to "first" if you prefer not to average coordinates
if "lat" in df.columns:
    agg_spec["lat"] = "mean"
if "lon" in df.columns:
    agg_spec["lon"] = "mean"

daily = (
    df.groupby(["city", "date"], as_index=False)
      .agg(agg_spec)
      .sort_values(["city", "date"])
)

# Add how many rows (hours) contributed per city-day
counts = df.groupby(["city", "date"]).size().rename("n_obs").reset_index()
daily = daily.merge(counts, on=["city", "date"], how="left")

# Optional: keep only complete days
if isinstance(EXPECTED_HOURS, int):
    before = len(daily)
    daily = daily[daily["n_obs"] >= EXPECTED_HOURS].copy()
    print(f"Filtered incomplete days: kept {len(daily)}/{before} city-day rows with >= {EXPECTED_HOURS} observations.")

# === SAVE ===
daily.to_csv(csv_out, index=False, float_format="%.6f")
print(f"Wrote {len(daily)} city-day rows to {csv_out}")