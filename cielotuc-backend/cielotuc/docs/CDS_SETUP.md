# CDS API Setup for ERA5 Data

CIELO·TUC uses ERA5 reanalysis data to improve model accuracy, especially for CAPE (Convective Available Potential Energy) and pressure-level wind data critical for Zonda detection.

## What is ERA5?

ERA5 is the fifth generation atmospheric reanalysis from ECMWF (European Centre for Medium-Range Weather Forecasts). It provides:

- **Surface variables**: temperature, pressure, wind, precipitation, CAPE, cloud cover, radiation
- **Pressure-level data**: temperature, wind, humidity at 850/700/500/300 hPa — essential for Zonda detection
- **Hourly resolution** from 1940 to present
- **0.25° grid resolution** (~25km)

ERA5 is **free** but requires registration.

## Step 1: Create CDS Account

1. Go to https://cds.climate.copernicus.eu
2. Click **"Log in"** → **"Create an account"**
3. Fill in your details (name, email, organization)
4. Verify your email

## Step 2: Create API Key

1. After logging in, go to https://cds.climate.copernicus.eu/how-to-api
2. You'll see your **UID** (User ID) and **API Key**
3. Copy both values

## Step 3: Configure CDS API

Create the file `~/.cdsapirc` (in your home directory) with:

```json
{
    "url": "https://cds.climate.copernicus.eu/api",
    "key": "YOUR_UID:YOUR_API_KEY"
}
```

Replace `YOUR_UID` and `YOUR_API_KEY` with your actual values.

### Windows Location

The file should be at:
```
C:\Users\<YourUsername>\.cdsapirc
```

### Alternative: Environment Variables

Instead of `.cdsapirc`, you can set:
```bash
set CDS_URL=https://cds.climate.copernicus.eu/api
set CDS_KEY=YOUR_UID:YOUR_API_KEY
```

## Step 4: Accept License

Before downloading ERA5 data, you must accept the ERA5 license:

1. Go to https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels
2. Click **"Download"** — you'll be prompted to accept the terms
3. Accept the license for both:
   - `reanalysis-era5-single-levels` (surface data)
   - `reanalysis-era5-pressure-levels` (pressure level data)

## Step 5: Test

```bash
cd cielotuc-backend/cielotuc
& "C:\tmp\cielotuc_venv\Scripts\python.exe" -c "
import cdsapi
c = cdsapi.Client(quiet=True)
print('CDS API configured successfully')
"
```

If it says "CDS API configured successfully", you're ready.

## Step 6: Train with ERA5

```bash
# Full training with ERA5 (downloads ~500MB of data, takes 10-30 minutes)
& "C:\tmp\cielotuc_venv\Scripts\python.exe" scripts/train_v1.py --era5 --max-epochs 100

# Quick test (1 fold, fewer epochs)
& "C:\tmp\cielotuc_venv\Scripts\python.exe" scripts/train_v1.py --era5 --folds 1 --max-epochs 20
```

## What ERA5 Improves

| Feature | NASA POWER | ERA5 |
|---------|-----------|------|
| CAPE | Bolton proxy (estimated) | Real CAPE from model |
| Pressure levels | Not available | 850/700/500 hPa temp & wind |
| Zonda detection | Proxy-based | Pressure-level wind shear |
| Cloud cover | Basic estimate | Model-assimilated |
| Soil temperature | Not available | Level 1 (0-7cm) |

## Data Size

- **Surface data**: ~150MB per decade (compressed NetCDF)
- **Pressure levels**: ~200MB per decade (compressed NetCDF)
- Total first download: ~350MB
- Subsequent runs use cached data (`models/era5_cache/`)

## Troubleshooting

### "ModuleNotFoundError: No module named 'cdsapi'"
```bash
& "C:\tmp\cielotuc_venv\Scripts\pip.exe" install cdsapi xarray netcdf4
```

### "HTTP 401 Unauthorized"
- Check your `.cdsapirc` UID and key are correct
- Make sure you've accepted the ERA5 license on the CDS website

### "HTTP 404 Not Found"
- You may need to accept the license first (Step 4)
- Try accessing the dataset page on the CDS website

### Download is slow
- ERA5 downloads can take 10-30 minutes for 10 years of hourly data
- Data is cached in `models/era5_cache/` — subsequent runs are instant
- You can cancel and resume (cached years won't re-download)

## No CDS Account?

CIELO·TUC works without ERA5. The `--era5` flag is optional:

```bash
# Training without ERA5 (uses NASA POWER only)
& "C:\tmp\cielotuc_venv\Scripts\python.exe" scripts/train_v1.py
```

NASA POWER provides sufficient data for the base model. ERA5 improves accuracy for extreme weather events (Zonda, storms) by providing better atmospheric profile data.
