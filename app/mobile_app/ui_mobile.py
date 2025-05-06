import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import folium
import branca.colormap as cm
from streamlit_folium import st_folium

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets" / "data"

def render_mobile():
    st.set_page_config(page_title="Crop Health – Mobile", layout="wide")

    # ── Mobile viewport meta + minimal padding ────────────────────────
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5.0, user-scalable=yes">
        <style>
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 0.5rem !important;
                padding-right: 0.5rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📱 Crop Health (Mobile)")
    st.markdown("---")

    # ── User inputs ────────────────────────────────────────────────────
    indice = st.text_input("Vegetation Index", value="NDVI")
    anio   = st.text_input("Year",            value="2024")
    qgis_path = ASSETS_DIR / f"INFORME_{indice}_QGIS_{anio}.xlsx"
    st.markdown("---")

    if not qgis_path.exists():
        st.warning(f"QGIS file not found at:\n`{qgis_path}`")
        qgis_path = st.file_uploader("Upload QGIS Excel", type=["xlsx", "xls"])
    if not qgis_path:
        st.stop()

    try:
        xls = pd.ExcelFile(qgis_path)
    except Exception as e:
        st.error(f"Error opening Excel: {e}")
        st.stop()

    sheet = st.selectbox("Select NDVI sheet", xls.sheet_names)
    try:
        df = pd.read_excel(qgis_path, sheet_name=sheet)
    except Exception as e:
        st.error(f"Error reading sheet: {e}")
        st.stop()

    # ── Clean + filter ─────────────────────────────────────────────────
    df["long-xm"] = pd.to_numeric(df["long-xm"], errors="coerce")
    df["long-ym"] = pd.to_numeric(df["long-ym"], errors="coerce")
    df["NDVI"]    = pd.to_numeric(df["NDVI"],    errors="coerce")
    df = df.dropna(subset=["long-xm", "long-ym", "NDVI"]).reset_index(drop=True)
    if df.empty:
        st.error("No valid coordinate/NDVI data to map.")
        return

    # ── Build Folium map centered on data ─────────────────────────────
    center_lat = df["long-ym"].mean()
    center_lon = df["long-xm"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)

    # ── Color scale legend ────────────────────────────────────────────
    colormap = cm.LinearColormap([
        "blue", "green", "yellow", "red"
    ], vmin=-1, vmax=1, caption="NDVI")
    colormap.add_to(m)

    # ── Add circle markers without popups/tooltips ─────────────────────
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row["long-ym"], row["long-xm"]],
            radius=6,
            color=colormap(row["NDVI"]),
            fill=True,
            fill_color=colormap(row["NDVI"]),
            fill_opacity=0.85,
        ).add_to(m)

    # ── Render the map and capture last click coords ───────────────────
    map_data = st_folium(
        m,
        width="100%", height=500,
        returned_objects=["last_clicked"]
    )
    click = map_data.get("last_clicked")

    # ── Show fields below when a point is clicked ─────────────────────
    if click:
        lat, lon = click.get("lat"), click.get("lng")
        # find nearest point
        dist2 = (df["long-ym"] - lat)**2 + (df["long-xm"] - lon)**2
        idx = dist2.idxmin()
        point = df.loc[idx]

        st.markdown(f"### Punto {idx + 1}")
        st.write(f"**Latitud:** {point['long-ym']:.6f}  •  **Longitud:** {point['long-xm']:.6f}")
        st.text_input("NDVI Value", value=f"{point['NDVI']:.4f}", disabled=True, key="ndvi_val")
        st.text_input("Riesgo Actual", value=str(point['Riesgo']), disabled=True, key="riesgo_act")
        st.selectbox("Riesgo", options=list(range(7)), key="riesgo_new")
    else:
        st.info("Touch a point on the map to view its data below.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(Y-%m-%d %H:%M:%S) [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_mobile()
