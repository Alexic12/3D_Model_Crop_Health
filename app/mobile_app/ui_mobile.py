"""
ui_mobile.py – mobile‑first Streamlit page
• Uses Plotly figure (pinch‑zoom ready) to display NDVI/Risk heat‑map.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui.visualization import create_2d_scatter_plot_ndvi_plotly

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets" / "data"


def render_mobile() -> None:
    st.set_page_config(page_title="Crop Health – Mobile", layout="wide")

    # viewport meta so full page can pinch‑zoom
    st.markdown(
        """
        <meta name="viewport"
              content="width=device-width,
                       initial-scale=1,
                       maximum-scale=5.0,
                       user-scalable=yes">
        """,
        unsafe_allow_html=True,
    )

    # tighten padding on phones
    st.markdown(
        """
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

    indice = st.text_input("Vegetation Index", value="NDVI")
    anio   = st.text_input("Year", value="2024")

    qgis_path = ASSETS_DIR / f"INFORME_{indice}_QGIS_{anio}.xlsx"
    st.markdown("---")

    if not qgis_path.exists():
        st.warning(f"QGIS file not found at:\n`{qgis_path}`")
        qgis_path = st.file_uploader("Upload QGIS Excel", type=["xlsx", "xls"])

    if not qgis_path:
        st.stop()

    # read Excel & choose sheet
    try:
        xls = pd.ExcelFile(qgis_path)
    except Exception as e:
        st.error(f"Error opening Excel: {e}")
        st.stop()

    sheet = st.selectbox("Select NDVI sheet", xls.sheet_names)

    try:
        df_qgis = pd.read_excel(qgis_path, sheet_name=sheet)
    except Exception as e:
        st.error(f"Error reading sheet: {e}")
        st.stop()

    # build Plotly figure (supports pinch‑zoom)
    fig = create_2d_scatter_plot_ndvi_plotly(
        qgis_df=df_qgis,
        sheet_name=sheet,
        margin_frac=0.05,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=dict(scrollZoom=True, responsive=True),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_mobile()