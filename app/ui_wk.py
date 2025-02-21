import streamlit as st
import logging
import pandas as pd
import numpy as np
import os
import streamlit.components.v1 as components


from data_processing import (
    process_uploaded_file,
    load_timeseries_data,
    invert_climate_file_rows,
    bulk_unzip_and_analyze_new_parallel
)
from app.visualization import (
    create_2d_scatter_plot_ndvi,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation
)

logger = logging.getLogger(__name__)

import streamlit as st
import streamlit.components.v1 as components


def render_responsive_carousel(sheet_list):
    """
    (Unused example carousel code, left here if you need it.)
    """
    bootstrap_cdn = """
    <link rel="stylesheet"
          href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    """
    custom_css = """
    <style>
        .carousel-container {
            width: 100%;
            height: 50px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .carousel {
            width: 80%;
            max-width: 100%;
        }
        .carousel-inner {
            display: flex;
            align-items: center;
        }
        .carousel-item {
            flex: 0 0 20%;
            text-align: center;
            opacity: 0.5;
            transition: transform 0.3s ease-in-out, opacity 0.3s;
        }
        .carousel-item.active {
            flex: 0 0 30%;
            font-size: 1.5rem;
            font-weight: bold;
            opacity: 1;
            transform: scale(1.2);
        }
        .carousel-control-prev,
        .carousel-control-next {
            width: 5%;
        }
        .carousel-control-prev-icon,
        .carousel-control-next-icon {
            background-color: black;
        }
        .carousel-text {
            color: black;
            font-size: 1rem;
            padding: 5px;
            white-space: nowrap;
        }
    </style>
    """

    carousel_items = ""
    for idx, sheet_name in enumerate(sheet_list):
        active_class = "active" if idx == 0 else ""
        carousel_items += f"""
        <div class="carousel-item {active_class}">
            <div class="carousel-text">{sheet_name}</div>
        </div>
        """

    carousel_html = f"""
    {bootstrap_cdn}
    {custom_css}
    <div class="carousel-container">
        <div id="sheetCarousel" class="carousel slide carousel-fade" data-ride="carousel" data-interval="2000">
            <div class="carousel-inner">
                {carousel_items}
            </div>
            <a class="carousel-control-prev" href="#sheetCarousel" role="button" data-slide="prev">
                <span class="carousel-control-prev-icon" aria-hidden="true"></span>
                <span class="sr-only">Previous</span>
            </a>
            <a class="carousel-control-next" href="#sheetCarousel" role="button" data-slide="next">
                <span class="carousel-control-next-icon" aria-hidden="true"></span>
                <span class="sr-only">Next</span>
            </a>
        </div>
    </div>
    """
    return carousel_html


def render_ui():
    st.title("Crop Health Visualization - Spectral Analysis BETA")

    # Use custom CSS to move the sidebar to the right
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
                margin-right: 400px; 
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # -------------- SIDEBAR --------------
    with st.sidebar:
        st.header("Settings")
        indice = st.text_input("Vegetation Index", value="NDVI")
        anio = st.text_input("Year", value="2024")
        st.write("---")

        grid_size = st.slider("Grid Resolution", 5, 300, 50, step=5)
        z_scale = st.slider("Z Scale", 0.1, 2.0, 1.0, step=0.1)
        smoothness = st.slider("Surface Smoothness (Gaussian)", 0.0, 10.0, 1.0, step=0.1)
        color_map = st.selectbox("Color Map", ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"])
        steps_value = st.slider("Time interpolation steps", 1, 20, 10)
        st.write("---")

        # Add a field for the Google Maps API Key
        google_api_key = st.text_input("Google Maps API Key", value="AIzaSyB1Vv2XMsTy1AxEowrzOaI5Sn96ffC6HNY")
        

        # ------ Bulk NDVI analysis ------
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
                    # Store the IDW file path in session state so we can visualize later
                    st.session_state["processed_idw_path"] = idw_xlsx

                if qgis_xlsx and os.path.exists(qgis_xlsx):
                    st.success(f"QGIS => {qgis_xlsx}")
                    with open(qgis_xlsx, "rb") as f:
                        st.download_button("Download QGIS", data=f,
                                           file_name=os.path.basename(qgis_xlsx))

        # ------ Invert climate file ------
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

    # -------------- MAIN CONTENT --------------
    st.write("## Single or Multi-Sheet NDVI Visualization")

    # ---- Upload-based approach (manual Excel) ----
    uploaded_xlsx = st.file_uploader("Upload Excel for NDVI Visuals", type=["xlsx", "xls"])
    simulate_button = st.button("Visualize NDVI")

    
    
    #check if INFORME_NDVI_IDW_2024.xlsx exists BUT NDVI AND 2024 ARE SELECTED BY USER
    idw_file = os.path.join(os.path.dirname(os.path.dirname(__file__)),"assets","data",f"INFORME_{indice}_IDW_{anio}.xlsx")
    processed_disabled = True
    processed_path = None
    # ---- Processed-data approach (from IDW output). ----
    if "processed_idw_path" in st.session_state:
        if os.path.exists(st.session_state["processed_idw_path"]):
            processed_disabled = False
            processed_path = st.session_state["processed_idw_path"]
    elif os.path.exists(idw_file):
        #set processed_disabled to False
        processed_disabled = False
        #set processed_path to idw_file
        processed_path = idw_file


    visualize_processed_btn = st.button("Visualize Processed Data", disabled=processed_disabled)
    if visualize_processed_btn:
        st.session_state["show_processed_data"] = True

    show_proc = st.session_state.get("show_processed_data", False)

    # ---------- BLOCK 1: Upload-based NDVI approach ----------
    if uploaded_xlsx and simulate_button:
        data_sheets = load_timeseries_data(uploaded_xlsx)

        if data_sheets:
            # -------------- Multi-sheet --------------
            sheet_list = list(data_sheets.keys())
            chosen_sheet = st.selectbox("Select sheet for static 3D & 2D", sheet_list)

            lat_arr = data_sheets[chosen_sheet]["lat"]
            lon_arr = data_sheets[chosen_sheet]["lon"]
            ndvi_mat = data_sheets[chosen_sheet]["ndvi"]

            # 2D NDVI figure **with** Google Maps background
            fig_2d = create_2d_scatter_plot_ndvi(
                lat_arr, lon_arr, ndvi_mat,
                sheet_name=chosen_sheet,
                google_api_key=google_api_key  # <- pass your key
            )

            # 3D static figure
            x_vals = []
            y_vals = []
            z_vals = []
            for i, latv in enumerate(lat_arr):
                for j, lonv in enumerate(lon_arr):
                    x_vals.append(lonv)
                    y_vals.append(latv)
                    z_vals.append(ndvi_mat[i, j])
            df_3d = pd.DataFrame({"Longitud": x_vals, "Latitud": y_vals, "NDVI": z_vals})
            fig_3d = create_3d_surface_plot(df_3d, grid_size, color_map, z_scale, smoothness)

            if fig_2d and fig_3d:
                w_px, h_px = fig_2d.get_size_inches() * fig_2d.dpi
                fig_3d.update_layout(
                    autosize=False,
                    width=int(w_px),
                    height=int(h_px)
                )

            col1, col2 = st.columns(2)
            with col1:
                if fig_2d:
                    st.pyplot(fig_2d, clear_figure=True)
            with col2:
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True, use_container_height=True)

            # Time-series 3D animation
            fig_time = create_3d_simulation_plot_time_interpolation(
                data_sheets, grid_size, color_map, z_scale, smoothness, steps_value
            )
            if fig_time:
                st.markdown("#### Time-Series 3D Animation")
                st.plotly_chart(fig_time, use_container_width=True)

        else:
            # -------------- Single-sheet --------------
            df_single = process_uploaded_file(uploaded_xlsx)
            if df_single is None:
                st.error("Could not parse single sheet data.")
                return

            if all(col in df_single.columns for col in ("Longitud", "Latitud", "NDVI")):
                lat_vals = df_single["Latitud"].values
                lon_vals = df_single["Longitud"].values
                ndvi_vals = df_single["NDVI"].values

                # 2D NDVI figure with map
                fig_2d = create_2d_scatter_plot_ndvi(
                    lat_vals, lon_vals, ndvi_vals,
                    sheet_name="SingleSheet",
                    google_api_key=google_api_key
                )

                # static 3D
                df_ren = df_single.rename(columns={
                    "Longitud": "Longitud",
                    "Latitud": "Latitud",
                    "NDVI": "NDVI"
                })
                fig_3d_static = create_3d_surface_plot(
                    df_ren,
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness
                )

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

                col1, col2 = st.columns(2)
                with col1:
                    if fig_2d:
                        st.pyplot(fig_2d)
                with col2:
                    if fig_3d_static:
                        st.plotly_chart(fig_3d_static, use_container_width=True)

                if fig_wave:
                    st.markdown("#### Wave-Based 3D Animation")
                    st.plotly_chart(fig_wave, use_container_width=True)
            else:
                st.error("The single-sheet data is missing required columns.")


    # ---------- BLOCK 2: Processed Data Visualization ----------
    if show_proc and (processed_path is not None) and os.path.exists(processed_path):
        st.write(f"### Visualizing Processed Data from: {processed_path}")

        data_sheets = load_timeseries_data(processed_path)
        if data_sheets:
            sheet_list = list(data_sheets.keys())
            chosen_sheet_processed = st.selectbox(
                "Select sheet for processed data (static 3D & 2D)",
                sheet_list,
                key="processed_sheet_selector"
            )

            lat_arr = data_sheets[chosen_sheet_processed]["lat"]
            lon_arr = data_sheets[chosen_sheet_processed]["lon"]
            ndvi_mat = data_sheets[chosen_sheet_processed]["ndvi"]

            fig_2d = create_2d_scatter_plot_ndvi(
                lat_arr, lon_arr, ndvi_mat,
                sheet_name=chosen_sheet_processed,
                google_api_key=google_api_key
            )

            x_vals, y_vals, z_vals = [], [], []
            for i, latv in enumerate(lat_arr):
                for j, lonv in enumerate(lon_arr):
                    x_vals.append(lonv)
                    y_vals.append(latv)
                    z_vals.append(ndvi_mat[i, j])

            df_3d = pd.DataFrame({"Longitud": x_vals, "Latitud": y_vals, "NDVI": z_vals})
            fig_3d = create_3d_surface_plot(df_3d, grid_size, color_map, z_scale, smoothness)

            if fig_2d and fig_3d:
                w_px, h_px = fig_2d.get_size_inches() * fig_2d.dpi
                fig_3d.update_layout(
                    autosize=False,
                    width=int(w_px),
                    height=int(h_px)
                )

            col1, col2 = st.columns(2)
            with col1:
                if fig_2d:
                    st.pyplot(fig_2d, clear_figure=True)
            with col2:
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True, use_container_height=True)

            fig_time = create_3d_simulation_plot_time_interpolation(
                data_sheets, grid_size, color_map, z_scale, smoothness, steps_value
            )
            if fig_time:
                st.markdown("#### Processed Time-Series 3D Animation")
                st.plotly_chart(fig_time, use_container_width=True)

        else:
            # Possibly single-sheet fallback
            df_single = process_uploaded_file(processed_path)
            if df_single is None:
                st.error("Could not parse processed data. It may not match the expected format.")
                return

            if all(col in df_single.columns for col in ("Longitud", "Latitud", "NDVI")):
                lat_vals = df_single["Latitud"].values
                lon_vals = df_single["Longitud"].values
                ndvi_vals = df_single["NDVI"].values

                fig_2d = create_2d_scatter_plot_ndvi(
                    lat_vals, lon_vals, ndvi_vals,
                    sheet_name="ProcessedSingle",
                    google_api_key=google_api_key
                )

                fig_3d_static = create_3d_surface_plot(
                    df_single.rename(columns={"Longitud": "Longitud",
                                              "Latitud": "Latitud",
                                              "NDVI": "NDVI"}),
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness
                )
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

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(fig_2d)
                with col2:
                    if fig_3d_static:
                        st.plotly_chart(fig_3d_static, use_container_width=True)

                if fig_wave:
                    st.markdown("#### Wave-Based 3D Animation (Processed Single-Sheet)")
                    st.plotly_chart(fig_wave, use_container_width=True)
            else:
                st.error("Processed data does not contain the required columns or sheets.")
