import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Interactive Temperature Explorer", layout="wide")
st.title("📈 Interactive Temperature Explorer")

@st.cache_data(show_spinner=False)
def load_csv_clean(file_or_path, parse_dates=None):
    encodings_to_try = ["cp1252", "latin1", "utf-8-sig", "utf-8"]
    last_err = None
    df_local = None
    for enc in encodings_to_try:
        try:
            df_local = pd.read_csv(file_or_path, encoding=enc, parse_dates=parse_dates)
            break
        except Exception as e:
            last_err = e
    if df_local is None:
        df_local = pd.read_csv(file_or_path, encoding="cp1252", parse_dates=parse_dates, encoding_errors="replace")
        st.warning(f"Loaded with encoding='cp1252' and encoding_errors='replace' due to: {last_err}")
    df_local.columns = df_local.columns.astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    for col in df_local.select_dtypes(include="object").columns:
        df_local[col] = df_local[col].str.replace("\xa0", " ", regex=False).str.strip()
    return df_local

TIME_ORDER = ["07:00", "11:00", "14:00", "16:00", "18:00", "21:00"]
TIME_PAT = re.compile(r"(?:(\d{2})(\d{2}))|(\d{2})")

def to_time_label(colname: str) -> str:
    m = TIME_PAT.search(colname)
    if not m:
        return colname
    if m.group(1) and m.group(2):
        return f"{m.group(1)}:{m.group(2)}"
    if m.group(3):
        return f"{m.group(3)}:00"
    return colname

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    if "Station" not in df.columns and "Staion" in df.columns:
        df = df.rename(columns={"Staion": "Station"})
    elif "Station" in df.columns and "Staion" in df.columns:
        df = df.drop(columns=["Staion"])
    required_id_cols = ["Date", "Station", "Common Location", "Location"]
    missing = [c for c in required_id_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}.")
    if not np.issubdtype(df["Date"].dtype, np.datetime64):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bms_cols = [c for c in ["0700_BMS", "1100_BMS", "1400_BMS", "1600_BMS", "1800_BMS", "2100_BMS"] if c in df.columns]
    manual_cols = [c for c in ["Manual07", "Manual11", "Manual14", "Manual16", "Manual18", "Manual21"] if c in df.columns]
    if not bms_cols and not manual_cols:
        raise KeyError("None of the expected reading columns were found.")
    id_vars = ["Date", "Station", "Common Location", "Location"]
    frames = []
    if manual_cols:
        m = df.melt(id_vars=id_vars, value_vars=manual_cols, var_name="Time", value_name="Value")
        m["Source"] = "Manual"
        frames.append(m)
    if bms_cols:
        b = df.melt(id_vars=id_vars, value_vars=bms_cols, var_name="Time", value_name="Value")
        b["Source"] = "BMS"
        frames.append(b)
    out = pd.concat(frames, ignore_index=True)
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna(subset=["Value", "Date"])
    out = out[(out["Value"] >= 15) & (out["Value"] <= 50)]
    out["TimeLabel"] = out["Time"].apply(to_time_label)
    out["TimeLabel"] = pd.Categorical(out["TimeLabel"], categories=TIME_ORDER, ordered=True)
    out = out[(out["Station"].astype(str).str.strip() != "") & (out["Location"].astype(str).str.strip() != "")]
    return out

# Hardcoded path to the backend CSV file
backend_csv_path = "data.csv"  # Make sure this matches the name of your file in the repo

# Load data from the CSV file
df_raw = load_csv_clean(backend_csv_path, parse_dates=["Date"])

# Transform the data
df_long = to_long(df_raw)

# Streamlit Sidebar Filters
st.sidebar.header("Filters")
min_date = pd.to_datetime(df_long["Date"].min())
max_date = pd.to_datetime(df_long["Date"].max())
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date.date(), max_date.date()))
stations = sorted(df_long["Station"].dropna().astype(str).unique().tolist())

sel_station = st.sidebar.selectbox("Station", stations, index=0)

avail_locs = sorted(df_long[df_long["Station"].astype(str) == sel_station]["Location"].dropna().astype(str).unique().tolist())
sel_locs = st.sidebar.multiselect("Location", avail_locs, default=avail_locs)
sources = sorted(df_long["Source"].unique().tolist())
sel_sources = st.sidebar.multiselect("Source", sources, default=sources)
times = [t for t in TIME_ORDER if t in df_long["TimeLabel"].astype(str).unique().tolist()]
sel_times = st.sidebar.multiselect("Time", times, default=times)
facet_by_location = st.sidebar.toggle("Facet by Location", value=True)

mask = (
    (df_long["Date"].dt.date >= start_date)
    & (df_long["Date"].dt.date <= end_date)
    & (df_long["Station"] == sel_station)
    & (df_long["Location"].astype(str).isin(sel_locs))
    & (df_long["Source"].isin(sel_sources))
    & (df_long["TimeLabel"].astype(str).isin(sel_times))
)

filtered = df_long.loc[mask].copy()

if filtered.empty:
    st.info("No data with current filters.")
    st.stop()

# Plot the data
facet_col = "Location" if facet_by_location else None
fig = px.scatter(
    filtered,
    x="Date",
    y="Value",
    color="Source",
    symbol="TimeLabel",
    facet_col=facet_col,
    facet_col_wrap=4 if facet_by_location else None,
    hover_data={"Station": True, "Common Location": True, "Location": True, "TimeLabel": True, "Source": True, "Value": ":.1f", "Date": "|%Y-%m-%d"},
)

if facet_by_location:
    loc_order = sorted(list(dict.fromkeys(filtered["Location"].astype(str))))
    columns = 4
    import math
    for idx, loc in enumerate(loc_order, start=1):
        ref = 24 if ("Platform" in str(loc)) else 28
        row = 1 + (idx - 1) // columns
        col = 1 + (idx - 1) % columns
        fig.add_hline(y=ref, line_width=2, line_dash="solid", line_color="red", row=row, col=col)
else:
    unique_refs = sorted({24 if ("Platform" in str(l)) else 28 for l in sel_locs})
    for ref in unique_refs:
        fig.add_hline(y=ref, line_width=2, line_dash="solid", line_color="red")

fig.update_layout(
    height=700,
    margin=dict(l=40, r=20, t=40, b=40),
    legend_title_text="Source / Time",
    yaxis_title="Temperature (°C)",
)

st.plotly_chart(fig, use_container_width=True)

# Provide the option to download the filtered data
@st.cache_data(show_spinner=False)
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
st.download_button(label="⬇️ Download filtered CSV", data=_to_csv_bytes(filtered), file_name="filtered_temperatures.csv", mime="text/csv")
