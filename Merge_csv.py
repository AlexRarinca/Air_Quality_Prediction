import pandas as pd

st = pd.read_csv('stations.csv', low_memory=False)
st_hour = pd.read_csv('station_hour.csv', low_memory=False)

for df in (st, st_hour):
    if 'StationId' in df.columns:
        df['StationId'] = df['StationId'].astype(str).str.strip()
    if 'City' in df.columns:
        df['City'] = df['City'].astype(str).str.strip()

st_hour['datetime'] = pd.to_datetime(st_hour['Datetime'], errors='coerce')
st_hour = st_hour.drop(columns=['Datetime'])
st_hour = st_hour.drop(columns=[c for c in ['Benzene','Toluene','Xylene'] if c in st_hour.columns], errors='ignore')

panel = st_hour.merge(
    st[['StationId','StationName','City','State','Status']],
    on='StationId', how='left', validate='many_to_one'
)

polls = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI']
for c in polls:
    if c in panel.columns:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')

if {'StationId','datetime'}.issubset(panel.columns):
    panel = panel.drop_duplicates(subset=['StationId','datetime'])

front = ['datetime','StationId','StationName','City','State','Status']
meas  = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','AQI','AQI_Bucket']
ordered = [c for c in front if c in panel.columns] + [c for c in meas if c in panel.columns] + [c for c in panel.columns if c not in set(front+meas)]
panel = panel[ordered]

panel.to_parquet('India_complete.parquet', index=False, compression='zstd')
panel.to_csv('India_complete.csv', index=False)