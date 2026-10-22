import io
import re
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from PIL import Image
import cv2

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

st.set_page_config(page_title="IFX | Defect Analytics", page_icon="▦", layout="wide")

st.markdown("""
<style>
.stApp { background: #0E1117; color: #E6EAF2; }
[data-testid="stSidebar"] { background: #151A23; }
.block-container { padding-top: 1.3rem; max-width: 1800px; padding-bottom: 3rem; }
h1, h2, h3 { color: #F3F6FC; }
div[data-testid="stMetric"] { background:#171D28; border:1px solid #2A3342; padding:16px 18px; border-radius:12px; }
div[data-testid="stMetricLabel"] { color:#AAB5C5; }
div[data-testid="stMetricValue"] { color:#FFFFFF; }
div[data-testid="stDataFrame"] { border:1px solid #283140; border-radius:8px; }
.small-note { color:#9BA7B8; font-size:0.86rem; }
.footer-signature {
    position: fixed;
    bottom: 8px;
    right: 16px;
    color: #2A3342;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    z-index: 999;
    user-select: none;
    pointer-events: none;
}
</style>
<div class="footer-signature">Designed by Danial Hensemmm</div>
""", unsafe_allow_html=True)

st.title("MULTI-DEVICE STRIPMAP ANALYTICS")
st.caption("ADVANCED MONITORING FOR VEGA, SPARROW, SQUID & SOLARIS  •  LOCAL WORKSPACE")

DEVICE_SPECS = {
    "VEGA": {"max_x": 64, "max_y": 15, "total_units": 960, "cols_per_panel": 16},
    "SPARROW": {"max_x": 96, "max_y": 20, "total_units": 1920, "cols_per_panel": 24},
    "SQUID": {"max_x": 72, "max_y": 17, "total_units": 1224, "cols_per_panel": 18},
    "SOLARIS": {"max_x": 76, "max_y": 17, "total_units": 1292, "cols_per_panel": 19},
}

ALIASES = {
    "lot": "MESLotID", "meslotid": "MESLotID", "lotid": "MESLotID", "lot_id": "MESLotID",
    "x": "X", "xcoord": "X", "x_coordinate": "X", "xposition": "X", "stripx": "X",
    "y": "Y", "ycoord": "Y", "y_coordinate": "Y", "yposition": "Y", "stripy": "Y",
    "device": "Device", "devicecategory": "Device", "product": "Device", "model": "Device",
    "usmdevice": "Device",
    "timestamp": "Timestamp", "datetime": "Timestamp", "date": "Timestamp",
    "productiondate": "Timestamp", "prod_date": "Timestamp", "proddate": "Timestamp",
    "rejectcode": "RejectCode", "reject_code": "RejectCode", "defectcode": "RejectCode", "failurecode": "RejectCode",
    "machinerejectcode": "MachineRejectCode", "machine_reject_code": "MachineRejectCode",
    "userrejectcode": "UserRejectCode", "user_reject_code": "UserRejectCode",
    "ifxlotno": "IFXLotNo", "stripid": "StripID",
}

REJECT_CODE_MAPPING = {
    2: "TA", 6: "ND", 10: "EC", 15: "NSOP", 16: "MB", 22: "NSOL", 26: "MBP", 30: "OT",
    33: "CS", 36: "NL", 38: "LP", 40: "LR", 68: "MSO", 78: "LT", 79: "NC", 102: "ISS"
}

def clean_columns(df):
    df = df.copy()
    renamed = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9_]", "", str(c).strip().lower().replace(" ", "_"))
        renamed[c] = ALIASES.get(key, str(c).strip())
    df = df.rename(columns=renamed)
    return df.loc[:, ~df.columns.duplicated()]

def normalize_device(value):
    s = str(value).strip().upper()
    if "SPARROW" in s or "U9927" in s:
        return "SPARROW"
    if "VEGA" in s or "U9921B" in s:
        return "VEGA"
    if "SOLARIS" in s:
        return "SOLARIS"
    if "SQUID" in s:
        return "SQUID"
    return "VEGA"

def unit_number(x, y, device_name="VEGA"):
    spec = DEVICE_SPECS.get(device_name, DEVICE_SPECS["VEGA"])
    max_x, max_y = spec["max_x"], spec["max_y"]
    x, y = int(x), int(y)
    
    if x < 1 or x > max_x or y < 1 or y > max_y:
        return 0
        
    x_offset = max_x - x
    pair_index = x_offset // 2
    is_ascending = (x_offset % 2) == 1
    total = max_x * max_y
    top_unit = total - (pair_index * 2 * max_y) - ((2 * max_y - 1) if is_ascending else 0)
    return top_unit + ((y - 1) if is_ascending else -(y - 1))

def parse_production_dates(values):
    parsed = pd.to_datetime(values, errors="coerce")
    numeric = pd.to_numeric(values, errors="coerce")
    serial_mask = numeric.between(20000, 80000)
    if serial_mask.any():
        parsed = pd.Series(parsed, index=values.index)
        parsed.loc[serial_mask] = pd.to_datetime(
            numeric.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    return pd.to_datetime(parsed, errors="coerce")

def prepare_data(df):
    df = clean_columns(df)
    if "Device" in df.columns and "DeviceID" not in df.columns:
        df["DeviceID"] = df["Device"].astype(str)
    if "X" in df.columns and "SourceX" not in df.columns:
        df["SourceX"] = df["X"]
    if "Y" in df.columns and "SourceY" not in df.columns:
        df["SourceY"] = df["Y"]
    required_defaults = [
        ("MESLotID", "UNASSIGNED"),
        ("X", np.nan),
        ("Y", np.nan),
        ("Device", "VEGA"),
        ("Timestamp", pd.NaT),
    ]
    for col, default in required_defaults:
        if col not in df.columns:
            df[col] = default
            
    if "MachineRejectCode" not in df.columns:
        df["MachineRejectCode"] = "Unspecified"
    if "UserRejectCode" not in df.columns:
        df["UserRejectCode"] = "Unspecified"
    if "RejectCode" not in df.columns:
        df["RejectCode"] = df["UserRejectCode"]

    df["Device_Category"] = df["DeviceID"].map(normalize_device)
    df["Device"] = df["Device_Category"]
    df["X"] = pd.to_numeric(df["X"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["SourceX"] = pd.to_numeric(df["SourceX"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["SourceY"] = pd.to_numeric(df["SourceY"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df["MESLotID"] = df["MESLotID"].fillna("UNASSIGNED").astype(str)
    df["Timestamp"] = parse_production_dates(df["Timestamp"])
    return df

def read_upload(upload):
    upload.seek(0)
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload, sheet_name=0, engine="openpyxl")

def sample_data():
    rng = np.random.default_rng(24)
    rows = []
    lots = ["LOT-VEGA-2401", "LOT-SPR-1108", "LOT-SQD-9901", "LOT-SOL-3321"]
    devices = ["VEGA", "SPARROW", "SQUID", "SOLARIS"]
    codes = [2, 10, 33, 38, 102]
    for i in range(500):
        idx = int(rng.integers(0, 4))
        dev_name = devices[idx]
        spec = DEVICE_SPECS[dev_name]
        x = int(rng.integers(1, spec["max_x"] + 1))
        y = int(rng.integers(1, spec["max_y"] + 1))
        c = int(rng.choice(codes))
        rows.append({
            "Timestamp": pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=int(rng.integers(0, 14*24*60))),
            "Device": dev_name, "DeviceID": dev_name, "MESLotID": lots[idx], "X": x, "Y": y,
            "MachineRejectCode": c, "UserRejectCode": c,
            "RejectCode": REJECT_CODE_MAPPING.get(c, str(c))
        })
    return prepare_data(pd.DataFrame(rows))

if "raw_data" not in st.session_state:
    st.session_state.raw_data = sample_data()

with st.sidebar:
    st.header("Data Control")
    enable_auto_refresh = st.checkbox("Enable Auto-Refresh (30s)", value=False)
    if enable_auto_refresh and st_autorefresh:
        st_autorefresh(interval=30000, key="datarefresh")

    uploaded = st.file_uploader("Upload defect logs", type=["csv", "xlsx"])
    if uploaded is not None:
        upload_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("loaded_upload_key") != upload_key:
            try:
                with st.spinner("Reading the raw data worksheet..."):
                    incoming = prepare_data(read_upload(uploaded))
                st.session_state.raw_data = incoming
                st.session_state.loaded_upload_key = upload_key
                st.session_state.pop("defect_editor", None)
                st.success(f"Loaded {len(incoming):,} rows from {uploaded.name}.")
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")
    st.markdown("---")
    st.subheader("Paste / Edit Records")

raw = st.session_state.raw_data.copy()
with st.expander("Open data editor", expanded=False):
    edited = st.data_editor(
        raw, num_rows="dynamic", use_container_width=True, height=300,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
            "X": st.column_config.NumberColumn("X coordinate", min_value=1, step=1),
            "Y": st.column_config.NumberColumn("Y coordinate", min_value=1, step=1),
        },
        key="defect_editor"
    )
    c1, c2 = st.columns([1, 5])
    if c1.button("Apply edits", type="primary"):
        st.session_state.raw_data = prepare_data(edited)
        st.success("Edits applied to the active dataset.")
        st.rerun()
    c2.download_button("Download current data as CSV", raw.to_csv(index=False).encode("utf-8"), "defect_logs.csv", "text/csv")

raw = prepare_data(st.session_state.raw_data)
if raw.empty:
    st.info("Upload or enter some defect records to start.")
    st.stop()

# ==========================================
# SETUP TABS (DASHBOARD, MAPPING PAINTER, UNIT FINDER)
# ==========================================
tab_dashboard, tab_painter, tab_finder = st.tabs(["📊 Dashboard", "🎨 Mapping Painter", "🔍 Unit Finder"])

# ------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------
with tab_dashboard:
    st.subheader("Detailed Analysis Filters")

    f1, f2, f3, f4, f5 = st.columns(5)

    category_options = ["VEGA", "SPARROW", "SQUID", "SOLARIS"]
    selected_category = f1.selectbox("Device Category", category_options, index=0)

    spec = DEVICE_SPECS[selected_category]
    max_x, max_y = spec["max_x"], spec["max_y"]

    raw_device_options = ["ALL"] + sorted(raw[raw["Device_Category"] == selected_category]["DeviceID"].dropna().unique().tolist())
    selected_device_model = f2.selectbox("Device Model / Variant", raw_device_options, index=0)

    lot_options = sorted(raw[raw["Device_Category"] == selected_category]["MESLotID"].dropna().unique().tolist())
    selected_lots = f3.multiselect("Production Lots", lot_options, default=lot_options)

    available_machine_codes = sorted(raw["MachineRejectCode"].dropna().unique().tolist())
    machine_code_labels = {c: f"{c} ({REJECT_CODE_MAPPING.get(int(c), 'UNKNOWN')})" if str(c).isdigit() and int(c) in REJECT_CODE_MAPPING else str(c) for c in available_machine_codes}
    selected_machine_codes = f4.multiselect(
        "Machine Reject Codes", 
        options=available_machine_codes, 
        format_func=lambda x: machine_code_labels.get(x, str(x)),
        default=available_machine_codes
    )

    available_user_codes = sorted(raw["UserRejectCode"].dropna().unique().tolist())
    user_code_labels = {c: f"{c} ({REJECT_CODE_MAPPING.get(int(c), 'UNKNOWN')})" if str(c).isdigit() and int(c) in REJECT_CODE_MAPPING else str(c) for c in available_user_codes}
    selected_user_codes = f5.multiselect(
        "User Reject Codes", 
        options=available_user_codes, 
        format_func=lambda x: user_code_labels.get(x, str(x)),
        default=available_user_codes
    )

    valid_dates = raw["Timestamp"].dropna()
    if len(valid_dates):
        min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
    else:
        min_date, max_date = date.today(), date.today()
    date_range = st.date_input("Production Date Range", value=(min_date, max_date))

    filtered = raw[raw["Device_Category"] == selected_category].copy()
    if selected_device_model != "ALL":
        filtered = filtered[filtered["DeviceID"] == selected_device_model]

    filtered = filtered[filtered["MESLotID"].isin(selected_lots)]
    filtered = filtered[filtered["MachineRejectCode"].isin(selected_machine_codes)]
    filtered = filtered[filtered["UserRejectCode"].isin(selected_user_codes)]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        if filtered["Timestamp"].notna().any():
            filtered = filtered[
                filtered["Timestamp"].isna() |
                ((filtered["Timestamp"].dt.date >= start_date) & (filtered["Timestamp"].dt.date <= end_date))
            ]

    filtered["X_internal"] = filtered["X"].abs()
    filtered["Y_internal"] = filtered["Y"]
    valid = filtered[
        filtered["X_internal"].between(1, max_x) &
        filtered["Y_internal"].between(1, max_y)
    ].copy()

    capacity = max_x * max_y
    unique_affected = valid[["DeviceID", "MESLotID", "X_internal", "Y_internal"]].drop_duplicates().shape[0]
    lot_device_count = valid[["DeviceID", "MESLotID"]].drop_duplicates().shape[0]
    total_max_units = max(1, lot_device_count * capacity)
    yield_pct = max(0.0, 100.0 - (unique_affected / total_max_units * 100.0))
    reject_count = len(filtered)
    lots_evaluated = filtered["MESLotID"].nunique()
    ratio_pct = (unique_affected / total_max_units * 100.0) if total_max_units else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rejects", f"{reject_count:,}")
    k2.metric("Lots Evaluated", f"{lots_evaluated:,}")
    k3.metric("Affected Unit Coordinates", f"{unique_affected:,}", f"{ratio_pct:.2f}% of lot capacity")
    k4.metric("Estimated Yield", f"{yield_pct:.2f}%")

    # Severity Analytics (Updated Categories)
    st.markdown("---")
    st.subheader(f"Defect Severity Analytics ({selected_category})")

    if not valid.empty:
        coordinate_rejects = valid.groupby(["X_internal", "Y_internal"]).size().to_dict()
        active_lots = valid[["DeviceID", "MESLotID"]].drop_duplicates().values
        full_grid_records = []
        for dev, lot in active_lots:
            for x in range(1, max_x + 1):
                for y in range(1, max_y + 1):
                    cnt = coordinate_rejects.get((x, y), 0)
                    full_grid_records.append({"DeviceID": dev, "MESLotID": lot, "X_internal": x, "Y_internal": y, "Count": cnt})
        
        full_grid_df = pd.DataFrame(full_grid_records)
        
        def get_severity(c):
            if c == 0: return "Zero"
            elif 1 <= c <= 10: return "Clean"
            elif 11 <= c <= 20: return "Moderate"
            elif 21 <= c <= 30: return "High"
            else: return "Critical"
            
        full_grid_df["Severity"] = full_grid_df["Count"].apply(get_severity)
        
        col_pie, col_trend = st.columns(2)
        with col_pie:
            st.markdown("#### Defect Proportion by Category")
            sev_counts = full_grid_df["Severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Total"]
            fig_pie = px.pie(
                sev_counts, names="Severity", values="Total", hole=0.4,
                color="Severity", 
                color_discrete_map={
                    "Zero": "#008000",
                    "Clean": "#FFC0CB", 
                    "Moderate": "#FFA500", 
                    "High": "#FF4500", 
                    "Critical": "#FF0000"
                }
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6EAF2", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_trend:
            st.markdown("#### Defect Trend Over Time")
            if valid["Timestamp"].notna().any():
                trend_df = valid.dropna(subset=["Timestamp"]).copy()
                trend_df["Date"] = trend_df["Timestamp"].dt.date
                trend_grouped = trend_df.groupby(["Date"]).size().reset_index(name="Rejects")
                fig_trend = px.line(trend_grouped, x="Date", y="Rejects", markers=True)
                fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E6EAF2", margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("Timestamp data not available for trend chart.")

    # Grid Map Generator Function
    st.markdown("---")
    st.subheader(f"Individual Unit Map ({selected_category}: {spec['total_units']} Units)")
    st.caption(f"Each square represents one physical unit. X axis: 1 to {max_x} (right to left), Y axis: 1 to {max_y} (top to bottom).")

    coordinate_rejects = valid.groupby(["X_internal", "Y_internal"]).size().to_dict()

    cols_per_p = spec["cols_per_panel"]
    panel_definitions = [
        ("Panel 4", list(range(max_x, max_x - cols_per_p, -1))),
        ("Panel 3", list(range(max_x - cols_per_p, max_x - 2*cols_per_p, -1))),
        ("Panel 2", list(range(max_x - 2*cols_per_p, max_x - 3*cols_per_p, -1))),
        ("Panel 1", list(range(max_x - 3*cols_per_p, 0, -1))),
    ]

    def get_tile_color(count):
        if count == 0:
            return "#008000"
        elif 1 <= count <= 10:
            return "#FFC0CB"
        elif 11 <= count <= 20:
            return "#FFA500"
        elif 21 <= count <= 30:
            return "#FF4500"
        else:
            return "#FF0000"

    panel_html_parts = []
    for panel_name, x_values in panel_definitions:
        tile_html = []
        x_header_html = "".join([f'<div class="axis-x-label">{x}</div>' for x in x_values])
        
        for y in range(1, max_y + 1):
            for x in x_values:
                unit_id = unit_number(x, y, selected_category)
                reject_count = int(coordinate_rejects.get((x, y), 0))
                tile_color = get_tile_color(reject_count)
                tooltip = f"X: {x}&#10;Y: {y}&#10;Unit: {unit_id}&#10;Rejects: {reject_count}"
                tile_html.append(
                    f'<div class="unit" style="background:{tile_color}" '
                    f'title="{tooltip}" aria-label="X {x}, Y {y}, Unit {unit_id}, Rejects {reject_count}">'
                    f'{unit_id}</div>'
                )
        
        x_footer_html = "".join([f'<div class="axis-x-label">{x}</div>' for x in x_values])
        
        panel_html_parts.append(
            f'<section class="panel"><div class="panel-title">{panel_name}</div>'
            f'<div class="axis-x-row" style="grid-template-columns: repeat({len(x_values)}, 22px);">{x_header_html}</div>'
            f'<div class="panel-grid" style="grid-template-columns: repeat({len(x_values)}, 22px);">{"".join(tile_html)}</div>'
            f'<div class="axis-x-row" style="grid-template-columns: repeat({len(x_values)}, 22px);">{x_footer_html}</div></section>'
        )

    y_axis_labels_html = "".join([f'<div class="axis-y-label">{y}</div>' for y in range(1, max_y + 1)])
    y_axis_column_html = f'<section class="y-panel"><div class="panel-title">Y</div><div class="axis-x-row">&nbsp;</div><div class="y-axis-grid">{y_axis_labels_html}</div><div class="axis-x-row">&nbsp;</div></section>'

    unit_map_html = f'''<!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: #0E1117; color: #E6EAF2; font-family: Arial, sans-serif; }}
    .matrix-scroll {{ width: 100%; overflow-x: auto; overflow-y: hidden; padding: 6px 4px 18px 4px; }}
    .panels {{ display: flex; align-items: flex-start; gap: 10px; width: max-content; padding-right: 15px; }}
    .panel {{ flex: 0 0 auto; }}
    .y-panel {{ flex: 0 0 auto; margin-left: 2px; }}
    .panel-title {{ text-align: center; color: #DCE5F2; font-size: 12px; font-weight: 600; padding: 0 0 5px 0; }}
    .axis-x-row {{ display: grid; gap: 2px; margin: 4px 0; }}
    .axis-x-label {{ width: 22px; text-align: center; color: #AAB5C5; font-size: 8px; font-weight: 600; line-height: 1; }}
    .panel-grid {{ display: grid; grid-auto-rows: 22px; gap: 2px; }}
    .y-axis-grid {{ display: grid; grid-template-columns: 20px; grid-auto-rows: 22px; gap: 2px; }}
    .axis-y-label {{ height: 22px; display: flex; align-items: center; justify-content: center; color: #AAB5C5; font-size: 9px; font-weight: 600; }}
    .unit {{ width: 22px; height: 22px; border: 1px solid rgba(255,255,255,.15); border-radius: 2px; display: flex; align-items: center; justify-content: center; color: #101318; font-size: 7px; line-height: 1; font-weight: 600; cursor: help; user-select: none; }}
    .unit:hover {{ outline: 2px solid #FFFFFF; outline-offset: 1px; transform: scale(1.10); position: relative; z-index: 5; }}
    .legend-note {{ color: #9BA7B8; font-size: 11px; margin: 8px 3px 0 3px; }}
    </style></head>
    <body><div class="matrix-scroll"><div class="panels">{''.join(panel_html_parts)}{y_axis_column_html}</div></div>
    <div class="legend-note">Hover over a unit to view X, Y, unit number, and reject count.</div>
    </body></html>'''
    components.html(unit_map_html, height=540, scrolling=False)

    legend_cols = st.columns(5)
    legends_data = [
        ("0 · Zero", "#008000"),
        ("1–10 · Clean", "#FFC0CB"),
        ("11–20 · Moderate", "#FFA500"),
        ("21–30 · High", "#FF4500"),
        (">30 · Critical", "#FF0000")
    ]
    for col, (label, color) in zip(legend_cols, legends_data):
        col.markdown(f'<div style="background:#171D28;border:1px solid #2A3342;border-radius:8px;padding:9px 12px;margin-top:8px;"><span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;border-radius:2px;"></span>{label}</div>', unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Top 10 Critical Hotspots")
        if valid.empty:
            st.info("No valid X/Y coordinates in the current selection.")
        else:
            hotspots = valid.groupby(["X_internal", "Y_internal"]).size().reset_index(name="Reject count")
            hotspots["X"] = hotspots["X_internal"].astype(int)
            hotspots["Y"] = hotspots["Y_internal"].astype(int)
            hotspots["Unit No."] = hotspots.apply(lambda r: unit_number(r["X"], r["Y"], selected_category), axis=1)
            hotspots = hotspots.sort_values(["Reject count", "X", "Y"], ascending=[False, False, True]).head(10)
            hotspots.insert(0, "Rank", range(1, len(hotspots)+1))
            st.dataframe(hotspots[["Rank", "Unit No.", "X", "Y", "Reject count"]], hide_index=True, use_container_width=True)
    with right:
        st.subheader("Recent Raw Data Stream")
        recent = filtered.sort_values("Timestamp", ascending=False, na_position="last").head(25).copy()
        recent["UnitNo"] = recent.apply(
            lambda r: unit_number(int(r["X"]), int(r["Y"]), selected_category)
            if pd.notna(r["X"]) and pd.notna(r["Y"]) and 1 <= r["X"] <= max_x and 1 <= r["Y"] <= max_y else np.nan,
            axis=1,
        )
        preferred = ["Timestamp", "Device", "DeviceID", "MESLotID", "IFXLotNo", "StripID", "SourceX", "SourceY", "X", "Y", "UnitNo", "MachineRejectCode", "UserRejectCode", "RejectCode"]
        display_recent = recent[[c for c in preferred if c in recent.columns]].rename(columns={
            "SourceX": "Raw X", "SourceY": "Raw Y", "X": "Normalized X", "Y": "Normalized Y", "UnitNo": "Unit No."
        })
        st.dataframe(display_recent, hide_index=True, use_container_width=True, height=360)
        st.download_button("Export filtered records (CSV)", filtered.to_csv(index=False).encode("utf-8"), f"filtered_{selected_category}_records.csv", "text/csv")

# ------------------------------------------
# TAB 2: MAPPING PAINTER
# ------------------------------------------
with tab_painter:
    st.subheader("🎨 Mapping Painter & Reference Viewer")
    painter_device = st.selectbox("Select Device Category for Painting", ["VEGA", "SPARROW", "SQUID", "SOLARIS"], index=0, key="painter_device_select")
    
    p_spec = DEVICE_SPECS[painter_device]
    p_max_x, p_max_y = p_spec["max_x"], p_spec["max_y"]
    p_cols_per_panel = p_spec["cols_per_panel"]

    st.caption(f"Device: {painter_device} ({p_spec['total_units']} Units). Click on any unit tile to toggle red (Reject).")

    if "selected_painter_units" not in st.session_state:
        st.session_state.selected_painter_units = set()

    query_params = st.query_params
    if "paint_unit" in query_params:
        try:
            u_clicked = int(query_params["paint_unit"])
            if u_clicked in st.session_state.selected_painter_units:
                st.session_state.selected_painter_units.remove(u_clicked)
            else:
                st.session_state.selected_painter_units.add(u_clicked)
        except ValueError:
            pass
        st.query_params.clear()
        st.rerun()

    if st.button("🗑️ Reset All Marked Units"):
        st.session_state.selected_painter_units = set()
        st.rerun()

    p_panel_definitions = [
        ("Panel 4", list(range(p_max_x, p_max_x - p_cols_per_panel, -1))),
        ("Panel 3", list(range(p_max_x - p_cols_per_panel, p_max_x - 2*p_cols_per_panel, -1))),
        ("Panel 2", list(range(p_max_x - 2*p_cols_per_panel, p_max_x - 3*p_cols_per_panel, -1))),
        ("Panel 1", list(range(p_max_x - 3*p_cols_per_panel, 0, -1))),
    ]

    painter_html_panels = []
    for panel_name, x_values in p_panel_definitions:
        tile_html = []
        x_header_html = "".join([f'<div class="axis-x-label">{x}</div>' for x in x_values])
        
        for y in range(1, p_max_y + 1):
            for x in x_values:
                u_id = unit_number(x, y, painter_device)
                is_marked = u_id in st.session_state.selected_painter_units
                bg_color = "#FF0000" if is_marked else "#008000"
                border_style = "2px solid #FFFFFF" if is_marked else "1px solid rgba(255,255,255,.15)"
                
                tooltip = f"Unit: {u_id} (X: {x}, Y: {y}) - Click to toggle"
                tile_html.append(
                    f'<div class="unit" style="background:{bg_color}; border:{border_style};" '
                    f'title="{tooltip}" onclick="try {{ window.parent.history.pushState({{}}, \'\', \'?paint_unit={u_id}\'); window.parent.dispatchEvent(new Event(\'popstate\')); }} catch(e) {{ window.parent.location.href = window.parent.location.pathname + \'?paint_unit={u_id}\'; }}">{u_id}</div>'
                )
        
        x_footer_html = "".join([f'<div class="axis-x-label">{x}</div>' for x in x_values])
        
        painter_html_panels.append(
            f'<section class="panel"><div class="panel-title">{panel_name}</div>'
            f'<div class="axis-x-row" style="grid-template-columns: repeat({len(x_values)}, 22px);">{x_header_html}</div>'
            f'<div class="panel-grid" style="grid-template-columns: repeat({len(x_values)}, 22px);">{"".join(tile_html)}</div>'
            f'<div class="axis-x-row" style="grid-template-columns: repeat({len(x_values)}, 22px);">{x_footer_html}</div></section>'
        )

    painter_y_axis_labels_html = "".join([f'<div class="axis-y-label">{y}</div>' for y in range(1, p_max_y + 1)])
    painter_y_axis_html = f'<section class="y-panel"><div class="panel-title">Y</div><div class="axis-x-row">&nbsp;</div><div class="y-axis-grid">{painter_y_axis_labels_html}</div><div class="axis-x-row">&nbsp;</div></section>'

    painter_map_html = f'''<!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: #0E1117; color: #E6EAF2; font-family: Arial, sans-serif; }}
    .matrix-scroll {{ width: 100%; overflow-x: auto; overflow-y: hidden; padding: 6px 4px 18px 4px; }}
    .panels {{ display: flex; align-items: flex-start; gap: 10px; width: max-content; padding-right: 15px; }}
    .panel {{ flex: 0 0 auto; }}
    .y-panel {{ flex: 0 0 auto; margin-left: 2px; }}
    .panel-title {{ text-align: center; color: #DCE5F2; font-size: 12px; font-weight: 600; padding: 0 0 5px 0; }}
    .axis-x-row {{ display: grid; gap: 2px; margin: 4px 0; }}
    .axis-x-label {{ width: 22px; text-align: center; color: #AAB5C5; font-size: 8px; font-weight: 600; line-height: 1; }}
    .panel-grid {{ display: grid; grid-auto-rows: 22px; gap: 2px; }}
    .y-axis-grid {{ display: grid; grid-template-columns: 20px; grid-auto-rows: 22px; gap: 2px; }}
    .axis-y-label {{ height: 22px; display: flex; align-items: center; justify-content: center; color: #AAB5C5; font-size: 9px; font-weight: 600; }}
    .unit {{ width: 22px; height: 22px; border-radius: 2px; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-size: 7px; line-height: 1; font-weight: 600; cursor: pointer; user-select: none; }}
    .unit:hover {{ outline: 2px solid #FFFFFF; outline-offset: 1px; transform: scale(1.10); position: relative; z-index: 5; }}
    </style></head>
    <body><div class="matrix-scroll"><div class="panels">{''.join(painter_html_panels)}{painter_y_axis_html}</div></div>
    </body></html>'''
    
    components.html(painter_map_html, height=540, scrolling=False)

    # Reference Viewer
    st.markdown("---")
    st.subheader("📋 StripMap Reference Image Viewer")
    st.caption("Upload your reference stripmap image here for direct visual comparison.")
    
    ref_upload = st.file_uploader("Upload Reference StripMap Image", type=["png", "jpg", "jpeg"], key="ref_uploader")
    if ref_upload is not None:
        ref_image = Image.open(ref_upload)
        st.image(ref_image, caption="Reference StripMap Screenshot", use_container_width=True)

    # Marked Rejects Table
    st.markdown("---")
    selected_count = len(st.session_state.selected_painter_units)
    st.markdown(f"### 🎯 Marked Reject Units: `{selected_count}` / {p_spec['total_units']}")

    if selected_count > 0:
        rows = []
        for u in sorted(st.session_state.selected_painter_units):
            found_x, found_y = None, None
            for x in range(1, p_max_x + 1):
                for y in range(1, p_max_y + 1):
                    if unit_number(x, y, painter_device) == u:
                        found_x, found_y = x, y
                        break
                if found_x:
                    break
            rows.append({
                "Device": painter_device,
                "Unit No.": u,
                "X": found_x,
                "Y": found_y,
                "Status": "REJECT (User Marked)"
            })
        
        marked_df = pd.DataFrame(rows)
        st.dataframe(marked_df, use_container_width=True, hide_index=True)
        
        st.download_button(
            "📥 Download Marked Units Table (CSV)",
            marked_df.to_csv(index=False).encode("utf-8"),
            f"marked_{painter_device}_rejects.csv",
            "text/csv",
            type="primary"
        )
    else:
        st.info("Click on any unit tile on the grid map above to mark it as reject (turns red).")

# ------------------------------------------
# TAB 3: UNIT FINDER
# ------------------------------------------
with tab_finder:
    st.subheader("🔍 Unit Finder & Coordinate Lookup")
    st.caption("Look up unit numbers from X & Y coordinates, or find X & Y coordinates from a unit number for specific device specifications.")

    # Strictly specific device selection - NO 'ALL' option
    finder_device = st.selectbox(
        "Select Device Category for Lookup",
        ["VEGA", "SPARROW", "SQUID", "SOLARIS"],
        index=0,
        key="finder_device_filter"
    )

    f_spec = DEVICE_SPECS[finder_device]
    f_max_x, f_max_y, f_total = f_spec["max_x"], f_spec["max_y"], f_spec["total_units"]

    st.markdown(f"**Selected Device**: `{finder_device}` — Total Units: `{f_total}` | Max X: `{f_max_x}` | Max Y: `{f_max_y}`")
    st.markdown("---")

    f_col1, f_col2 = st.columns(2, gap="large")

    with f_col1:
        st.markdown("### 📍 Lookup by X & Y Coordinates")
        st.markdown(f"Enter coordinates $(1 \le X \le {f_max_x}, 1 \le Y \le {f_max_y})$ to get the corresponding Unit Number.")
        
        fx_in, fy_in = st.columns(2)
        with fx_in:
            search_x = st.number_input("X Coordinate", min_value=1, max_value=f_max_x, value=1, step=1, key="search_x_input")
        with fy_in:
            search_y = st.number_input("Y Coordinate", min_value=1, max_value=f_max_y, value=1, step=1, key="search_y_input")
        
        calculated_unit = unit_number(search_x, search_y, finder_device)
        
        st.markdown(
            f"""
            <div style="background:#171D28; border:2px solid #2A3342; border-radius:12px; padding:24px; text-align:center; margin-top:16px;">
                <div style="color:#AAB5C5; font-size:0.95rem; font-weight:600; text-transform:uppercase; margin-bottom:8px;">Resulting Unit Number</div>
                <div style="color:#00FFA6; font-size:2.8rem; font-weight:800;">{calculated_unit}</div>
                <div style="color:#9BA7B8; font-size:0.85rem; margin-top:8px;">Device: <b>{finder_device}</b> | X: {search_x} | Y: {search_y}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with f_col2:
        st.markdown("### 🔢 Lookup by Unit Number")
        st.markdown(f"Enter a Unit Number $(1 \le \text{{Unit}} \le {f_total})$ to find its X & Y coordinates.")
        
        search_unit = st.number_input("Unit Number", min_value=1, max_value=f_total, value=1, step=1, key="search_unit_input")
        
        # Reverse lookup for X and Y
        found_xy = None
        for x in range(1, f_max_x + 1):
            for y in range(1, f_max_y + 1):
                if unit_number(x, y, finder_device) == search_unit:
                    found_xy = (x, y)
                    break
            if found_xy:
                break
        
        if found_xy:
            rx, ry = found_xy
            st.markdown(
                f"""
                <div style="background:#171D28; border:2px solid #2A3342; border-radius:12px; padding:24px; text-align:center; margin-top:16px;">
                    <div style="color:#AAB5C5; font-size:0.95rem; font-weight:600; text-transform:uppercase; margin-bottom:8px;">Resulting Coordinates</div>
                    <div style="color:#3B82F6; font-size:2.4rem; font-weight:800;">X: {rx} &nbsp;|&nbsp; Y: {ry}</div>
                    <div style="color:#9BA7B8; font-size:0.85rem; margin-top:8px;">Device: <b>{finder_device}</b> | Unit ID: {search_unit}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Unit number not found in grid layout.")