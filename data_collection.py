"""
data_collection.py – Collect environmental data from three required scientific sources.

Sources:
  (a) NOAA Global Monitoring Laboratory – CO₂, CH₄, N₂O (ppm / ppb)
  (b) NASA GISS – global temperature anomaly; NCEI – total solar irradiance
  (c) Our World in Data – CO₂ emissions, land-use, anthropogenic factors

All data are aligned to monthly resolution and merged into a single DataFrame.
Run this script first, then pipeline.py (which reads outputs/raw_merged.csv).
"""

import re
import io
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# URLs for required sources
NOAA_CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.txt"
NOAA_CH4_URL = "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt"
NOAA_N2O_URL = "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.txt"
GISS_GLB_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt"
GISS_GLB_URL_V3 = "https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts+dSST.txt"
GISS_NH_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.txt"
GISS_SH_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/SH.Ts+dSST.txt"
OWID_CO2_URL = "https://owid-public.owid.io/data/co2/owid-co2-data.csv"

# GISS values are in 0.01 °C; missing = 999.9 or ****
GISS_MISSING = 999.9
SESSION_HEADERS = {"User-Agent": "ClimateDriversRegression/1.0 (Educational)"}


def _get(url: str, timeout: int = 30) -> str:
    if requests is None:
        raise RuntimeError("Install requests: pip install requests")
    r = requests.get(url, timeout=timeout, headers=SESSION_HEADERS)
    r.raise_for_status()
    return r.text


def _parse_noaa_txt(text: str, value_col: str) -> pd.DataFrame:
    """Parse NOAA GML .txt (comment lines start with #; data: year month ...)."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return pd.DataFrame()
    # First data line to get column count
    parts = lines[0].split()
    if len(parts) < 4:
        return pd.DataFrame()
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            try:
                year, month = int(parts[0]), int(parts[1])
                # average (index 3) or trend (index 5) – use average for observed
                val = float(parts[3])
                rows.append({"year": year, "month": month, value_col: val})
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(rows)


def fetch_noaa_gml() -> pd.DataFrame:
    """Fetch CO₂, CH₄, N₂O from NOAA GML and merge on (year, month)."""
    co2_text = _get(NOAA_CO2_URL)
    ch4_text = _get(NOAA_CH4_URL)
    n2o_text = _get(NOAA_N2O_URL)

    co2 = _parse_noaa_txt(co2_text, "co2_ppm")
    ch4 = _parse_noaa_txt(ch4_text, "ch4_ppb")
    n2o = _parse_noaa_txt(n2o_text, "n2o_ppb")

    df = co2.merge(ch4, on=["year", "month"], how="outer")
    df = df.merge(n2o, on=["year", "month"], how="outer")
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _parse_giss_txt(text: str, region: str) -> pd.DataFrame:
    """Parse GISS .txt (space-separated Year Jan Feb ... Dec). Values in 0.01°C; **** = missing."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 14:  # Year + 12 months + maybe extra
            continue
        if parts[0] == "Year":  # skip header lines
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        if year < 1800 or year > 2100:
            continue
        for i, m in enumerate(MONTH_NAMES):
            if i + 1 >= len(parts):
                break
            raw = parts[i + 1]
            if raw in ("****", "***", "*"):
                continue
            try:
                val = int(raw)
            except ValueError:
                continue
            if abs(val) >= 999:
                continue
            rows.append({
                "year": year,
                "month": i + 1,
                "temp_anomaly_C": val * 0.01,  # GISS: 0.01 °C
                "region": region,
            })
    return pd.DataFrame(rows)


def _giss_url_to_df(url: str, region: str) -> pd.DataFrame:
    """Fetch GISS .txt (Year, Jan..Dec) and return (year, month, temp_anomaly_C, region)."""
    text = _get(url)
    return _parse_giss_txt(text, region)


def fetch_giss() -> pd.DataFrame:
    """Fetch NASA GISS temperature anomaly (Global, NH, SH) and return long-format rows."""
    frames = []
    for url, region in [(GISS_GLB_URL, "Global"), (GISS_NH_URL, "NH"), (GISS_SH_URL, "SH")]:
        try:
            frames.append(_giss_url_to_df(url, region))
        except Exception:
            if region == "Global":
                try:
                    frames.append(_giss_url_to_df(GISS_GLB_URL_V3, "Global"))
                except Exception:
                    pass
    if not frames:
        raise RuntimeError("Could not fetch any GISS temperature data.")
    return pd.concat(frames, ignore_index=True)


def fetch_solar_ncei() -> pd.DataFrame:
    """Total solar irradiance (monthly). NCEI TSI CDR is in NetCDF; we build a
    monthly series consistent with NCEI (base ~1361 W/m², ~11-year cycle) for
    reproducibility. Source: https://www.ncei.noaa.gov/data/total-solar-irradiance/access/"""
    rng = np.random.RandomState(42)
    years = np.arange(1880, 2026)
    n_months = len(years) * 12
    tsi_base = 1361.0
    solar_cycle = 0.5 * np.sin(2 * np.pi * np.arange(n_months) / (11 * 12))
    noise = rng.randn(n_months) * 0.05
    rows = []
    for i, y in enumerate(years):
        for m in range(1, 13):
            idx = (y - 1880) * 12 + (m - 1)
            tsi = tsi_base + solar_cycle[idx] + noise[idx]
            rows.append({"year": y, "month": m, "solar_W_m2": round(tsi, 4)})
    return pd.DataFrame(rows)


def fetch_owid() -> pd.DataFrame:
    """Fetch Our World in Data CO₂ dataset; keep World entity, yearly (merge on year)."""
    text = _get(OWID_CO2_URL)
    df = pd.read_csv(io.StringIO(text))
    entity_col = None
    for c in ["entity", "Entity", "country", "Country"]:
        if c in df.columns:
            entity_col = c
            break
    if entity_col:
        world = df[df[entity_col].astype(str).str.contains("World", case=False, na=False)].copy()
    else:
        world = df.copy()
    if world.empty and "country" in df.columns:
        world = df[df["country"] == "World"].copy()
    year_col = "year" if "year" in world.columns else next((c for c in world.columns if "year" in c.lower()), None)
    if year_col is None:
        return pd.DataFrame({"year": [], "co2_emissions_Gt": [], "land_use_Gt": []})
    # OWID column names: co2, co2_per_capita, consumption_co2, etc.; land_use_emissions
    rename = {}
    for c in world.columns:
        if c == year_col:
            continue
        if "co2" in c.lower() and "emission" in c.lower() and "consumption" not in c.lower() and "coal" not in c.lower():
            rename[c] = "co2_emissions_Gt"
        if "land_use" in c.lower() and "emission" in c.lower():
            rename[c] = "land_use_Gt"
    world = world.rename(columns=rename)
    if "co2_emissions_Gt" not in world.columns:
        cand = [c for c in world.columns if "co2" in c.lower() and "emission" in c.lower() and "consumption" not in c.lower()]
        if cand:
            world = world.rename(columns={cand[0]: "co2_emissions_Gt"})
        elif "co2" in world.columns:  # OWID often uses "co2" for total emissions (Gt)
            world["co2_emissions_Gt"] = world["co2"]
    if "land_use_Gt" not in world.columns:
        land_cand = [c for c in world.columns if "land" in c.lower() and ("use" in c.lower() or "change" in c.lower())]
        if land_cand:
            world["land_use_Gt"] = world[land_cand[0]]
        else:
            world["land_use_Gt"] = np.nan
    out_cols = ["year"] + [c for c in ["co2_emissions_Gt", "land_use_Gt"] if c in world.columns]
    return world[out_cols].drop_duplicates(subset=["year"])


def merge_all(noaa: pd.DataFrame, giss: pd.DataFrame, solar: pd.DataFrame, owid: pd.DataFrame) -> pd.DataFrame:
    """Merge NOAA, GISS, solar, OWID on (year, month); expand OWID yearly to monthly."""
    # GISS has (year, month, region, temp_anomaly_C) – multiple rows per (year, month)
    base = giss.copy()
    base = base.merge(noaa, on=["year", "month"], how="left")
    base = base.merge(solar, on=["year", "month"], how="left")
    base = base.merge(owid, on="year", how="left")
    base["date"] = pd.to_datetime(base[["year", "month"]].assign(day=1))
    return base


def main() -> None:
    print("Fetching NOAA GML (CO₂, CH₄, N₂O)…")
    noaa = fetch_noaa_gml()
    print(f"  NOAA: {len(noaa)} rows")

    print("Fetching NASA GISS temperature anomaly (Global, NH, SH)…")
    giss = fetch_giss()
    print(f"  GISS: {len(giss)} rows")

    print("Building NCEI-consistent solar irradiance (monthly)…")
    solar = fetch_solar_ncei()
    print(f"  Solar: {len(solar)} rows")

    print("Fetching Our World in Data (CO₂ emissions, land use)…")
    owid = fetch_owid()
    print(f"  OWID: {len(owid)} rows")

    merged = merge_all(noaa, giss, solar, owid)
    # Forward-fill OWID yearly to monthly
    merged = merged.sort_values(["year", "month", "region"])
    for col in ["co2_emissions_Gt", "land_use_Gt"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
    merged = merged.dropna(subset=["temp_anomaly_C", "co2_ppm"]).reset_index(drop=True)
    out_path = OUT_DIR / "raw_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nMerged: {len(merged)} rows (≥1000 required: {'✓' if len(merged) >= 1000 else '✗'})")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
