import streamlit as st
import logging
import pandas as pd
import numpy as np
import os

from data_processing import (
    process_uploaded_file,
    load_timeseries_data,
    invert_climate_file_rows,
    rejilla_indice,
    IDW_Index,
    Riesgo,
    bulk_unzip_and_analyze_new_parallel
)
from visualization import (
    create_2d_scatter_plot_ndvi,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation
)

logger = logging.getLogger(__name__)

def render_ui():
    st.title("Crop Health Visualization (Production Version)")

    # Sidebar

    st.markdown(
        """
        <style>
            /* Move the sidebar to the right */
            [data-testid="stSidebar"] {
                position: fixed !important;
                right: 0px !important;
                top: 0px !important;
                height: 100% !important;
                background-color: black !important;
                z-index: 100 !important;
                box-shadow: -5px 0px 10px rgba(0,0,0,0.2);
            }

            /* Adjust the main content so it doesn't overlap */
            [data-testid="stAppViewContainer"] {
                margin-right: 400px; /* Adjust width according to sidebar */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.header("Settings")
        indice = st.text_input("Vegetation Index", value="NDVI")
        anio = st.text_input("Year", value="2024")
        st.write("---")

        grid_size = st.slider("Grid Resolution", 5, 300, 50, step=5)
        z_scale = st.slider("Z Scale", 0.1, 2.0, 1.0, step=0.1)
        smoothness = st.slider("Surface Smoothness (Gaussian)", 0.0, 10.0, 1.0, step=0.1)
        color_map = st.selectbox("Color Map", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
        st.write("---")

        steps_value = st.slider("Time interpolation steps", 1, 20, 10)

        # Bulk NDVI analysis
        st.subheader("Bulk NDVI ZIP Analysis")
        zip_files = st.file_uploader("Upload .zip pairs containing base + ColorMap", type=["zip"], accept_multiple_files=True)
        base_folder = "./upload_data"
        subfolder = os.path.join(base_folder, f"{indice}_{anio}")

        if zip_files:
            if not os.path.exists(subfolder):
                os.makedirs(subfolder, exist_ok=True)
            for zf in zip_files:
                outpath = os.path.join(subfolder, zf.name)
                with open(outpath, "wb") as f:
                    f.write(zf.getbuffer())
            st.success("All ZIPs saved to folder. Ready for analysis.")

        if st.button("Run Bulk Analysis"):
            if not os.path.exists(subfolder):
                st.error("No subfolder found or no files. Please upload first.")
            else:
                esp_xlsx, idw_xlsx, qgis_xlsx = bulk_unzip_and_analyze_new_parallel(indice, anio, base_folder=base_folder)
                if esp_xlsx and os.path.exists(esp_xlsx):
                    st.success(f"Espacial => {esp_xlsx}")
                    with open(esp_xlsx, "rb") as f:
                        st.download_button("Download Espacial", data=f,
                                           file_name=os.path.basename(esp_xlsx))
                if idw_xlsx and os.path.exists(idw_xlsx):
                    st.success(f"IDW => {idw_xlsx}")
                    with open(idw_xlsx, "rb") as f:
                        st.download_button("Download IDW", data=f,
                                           file_name=os.path.basename(idw_xlsx))
                if qgis_xlsx and os.path.exists(qgis_xlsx):
                    st.success(f"QGIS => {qgis_xlsx}")
                    with open(qgis_xlsx, "rb") as f:
                        st.download_button("Download QGIS", data=f,
                                           file_name=os.path.basename(qgis_xlsx))

        # Invert climate file rows
        with st.expander("Invert Climate Excel Rows"):
            climate_file = st.file_uploader("Upload Climate Excel", type=["xlsx", "xls"])
            if climate_file:
                df_inv = invert_climate_file_rows(climate_file)
                if df_inv is not None:
                    st.dataframe(df_inv.head(20))
                    st.download_button(
                        "Download Inverted Excel",
                        data=df_inv.to_excel(index=False),
                        file_name="clima_inverted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    # Main content
    st.write("## Single or Multi-Sheet NDVI Visualization")

    uploaded_xlsx = st.file_uploader("Upload Excel for NDVI Visuals", type=["xlsx", "xls"])
    simulate_button = st.button("Visualize NDVI")

    if uploaded_xlsx and simulate_button:
        data_sheets = load_timeseries_data(uploaded_xlsx)
        if data_sheets:
            # If multi-sheet
            sheet_list = list(data_sheets.keys())
            chosen_sheet = st.selectbox("Select sheet for static 3D & 2D", sheet_list)
            lat_arr = data_sheets[chosen_sheet]["lat"]
            lon_arr = data_sheets[chosen_sheet]["lon"]
            ndvi_mat = data_sheets[chosen_sheet]["ndvi"]

            fig_2d = create_2d_scatter_plot_ndvi(lat_arr, lon_arr, ndvi_mat, chosen_sheet)
            if fig_2d:
                st.pyplot(fig_2d)

            # Flatten to pass to 3D
            x_vals = []
            y_vals = []
            z_vals = []
            for i, latv in enumerate(lat_arr):
                for j, lonv in enumerate(lon_arr):
                    x_vals.append(lonv)
                    y_vals.append(latv)
                    z_vals.append(ndvi_mat[i,j])
            df_3d = pd.DataFrame({"Longitud": x_vals, "Latitud": y_vals, "NDVI": z_vals})
            fig_3d = create_3d_surface_plot(df_3d, grid_size, color_map, z_scale, smoothness)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)

            # Also create multi-sheet interpolation
            fig_time = create_3d_simulation_plot_time_interpolation(
                data_sheets, grid_size, color_map, z_scale, smoothness, steps_value
            )
            if fig_time:
                st.plotly_chart(fig_time, use_container_width=True)

        else:
            # maybe single sheet with columns [Longitud, Latitud, NDVI, Riesgo]
            df_single = process_uploaded_file(uploaded_xlsx)
            if df_single is None:
                st.error("Could not parse single sheet data.")
                return

            # Let's do a wave-based animation
            fig_wave = create_3d_simulation_plot_sea_keypoints(
                df_single,
                grid_size=grid_size,
                color_map=color_map,
                z_scale=z_scale,
                smoothness=smoothness,
                key_frames=50,
                key_points=10,
                wave_amplitude_range=(0.05, 0.4),
                wave_frequency_range=(0.05, 0.2),
                wave_speed_range=(0.5, 2.0),
                influence_sigma=0.05,
                random_seed=42
            )
            if fig_wave:
                st.plotly_chart(fig_wave, use_container_width=True)
