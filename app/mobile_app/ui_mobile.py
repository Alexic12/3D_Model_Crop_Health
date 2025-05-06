"""
Lightweight mobile‑friendly version – start small & iterate.
"""

import streamlit as st
from app.data.hpc_calculator import HCPCalc   # reuse core logic

def render_mobile() -> None:
    st.set_page_config(page_title="Crop Health – Mobile", layout="centered")
    st.title("📱 Crop Health (Mobile)")

    indice = st.text_input("Vegetation Index", value="NDVI")
    anio = st.text_input("Year", value="2024")
    if st.button("Run quick risk map"):
        with st.spinner("Crunching…"):
            df_map, risk_info = HCPCalc.compute_risk_results_via_hpc(indice, anio)
        st.dataframe(df_map)
        st.success("Done!")

if __name__ == "__main__":
    render_mobile()