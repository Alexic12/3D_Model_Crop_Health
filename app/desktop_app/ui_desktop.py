"""
ui.py

Streamlit UI code for Crop Health Visualization, referencing HPC logic from data_processing.py
and 3D/2D plotting from app.visualization. Includes an Advanced Mode toggle that hides/displays
sliders on the sidebar for a cleaner UI.
"""

import streamlit as st
import logging
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# HPC imports from your data_processing
from app.data.data_processing import (
    process_uploaded_file,
    load_timeseries_data,
    invert_climate_file_rows,
    bulk_unzip_and_analyze_new_parallel,
    Prospectiva,  # HPC function if needed directly
    run_full_hpc_pipeline
)

import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# app.visualization references
import streamlit.components.v1 as components
from app.ui.visualization import (
    create_2d_scatter_plot_ndvi,
    create_2d_scatter_plot_ndvi_interactive_qgis,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file (e.g., for Google Maps API)
load_dotenv()



def render_ui():
    """
    Render the entire Streamlit UI, including the new Advanced Mode toggle for
    hiding or showing advanced sliders on the sidebar.
    """
    st.set_page_config(
        page_title="Crop Health Visualization",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Crop Health Visualization")

    st.write("---")
    page_mode = st.radio(
        "Select Visualization Page",
        ["3D Visualization", "Risk Visualization"],
        index=0
    )
    st.write("---")

    # CSS for styling the sidebar on the right side with a black background
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                position: fixed !important;
                right: 0px !important;
                top: 0px !important;
                height: 100% !important;
                background-color: black !important;
                z-index: 100 !important;
                box-shadow: -5px 0px 10px rgba(0,0,0,0.2);
            }
            [data-testid="stAppViewContainer"] {
                margin-right: 400px; 
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Settings")

        indice = st.text_input("Vegetation Index", value="NDVI")
        anio = st.text_input("Year", value="2024")
        st.write("---")

        

        # Google Maps API Key from .env
        apikey = os.getenv("GOOGLE_MAPS_API_KEY", "AIzaSyB1Vv2XMsTy1AxEowrzOaI5Sn96ffC6HNY")
        print(f"API Key: {apikey}")
        google_api_key = apikey

        st.subheader("Bulk NDVI ZIP Analysis")
        zip_files = st.file_uploader("Upload .zip pairs containing base + ColorMap",
                                     type=["zip"], accept_multiple_files=True)
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
                esp_xlsx, idw_xlsx, qgis_xlsx = bulk_unzip_and_analyze_new_parallel(
                    indice, anio, base_folder=base_folder
                )
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
                    st.session_state["processed_idw_path"] = idw_xlsx

                if qgis_xlsx and os.path.exists(qgis_xlsx):
                    st.success(f"QGIS => {qgis_xlsx}")
                    with open(qgis_xlsx, "rb") as f:
                        st.download_button("Download QGIS", data=f,
                                           file_name=os.path.basename(qgis_xlsx))

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

        # ---- Advanced Mode Toggle ----
        #advanced_mode = st.checkbox("Advanced Mode", value=False)
        # Use an expander that is shown or hidden based on 'advanced_mode'
        with st.expander("Advanced Sliders", expanded=False):
            grid_size = st.slider("Grid Resolution", 5, 300, 50, step=5)
            z_scale = st.slider("Z Scale", 0.1, 2.0, 1.0, step=0.1)
            smoothness = st.slider("Surface Smoothness (Gaussian)", 0.0, 10.0, 1.0, step=0.1)
            color_map = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "cividis"])
            steps_value = st.slider("Time interpolation steps", 1, 20, 10)
            st.write("---")

    # ---------------- MAIN PAGE CONTENT ----------------
    if page_mode == "3D Visualization":
        st.write("## Single or Multi-Sheet NDVI Visualization")

        uploaded_xlsx = st.file_uploader("Upload Excel for NDVI Visuals", type=["xlsx", "xls"])
        simulate_button = st.button("Visualize NDVI")

        # Check for an existing processed IDW path
        idw_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "assets","data",
            f"INFORME_{indice}_IDW_{anio}.xlsx"
        )
        processed_disabled = True
        processed_path = None
        if "processed_idw_path" in st.session_state:
            if os.path.exists(st.session_state["processed_idw_path"]):
                processed_disabled = False
                processed_path = st.session_state["processed_idw_path"]
        elif os.path.exists(idw_file):
            processed_disabled = False
            processed_path = idw_file

        visualize_processed_btn = st.button("Visualize Processed Data", disabled=processed_disabled)
        if visualize_processed_btn:
            st.session_state["show_processed_data"] = True

        show_proc = st.session_state.get("show_processed_data", False)

        if uploaded_xlsx and simulate_button:
            data_sheets = load_timeseries_data(uploaded_xlsx)
            if data_sheets:
                sheet_list = list(data_sheets.keys())
                chosen_sheet = st.selectbox("Select sheet for static 3D & 2D", sheet_list)

                lat_arr = data_sheets[chosen_sheet]["lat"]
                lon_arr = data_sheets[chosen_sheet]["lon"]
                ndvi_mat = data_sheets[chosen_sheet]["ndvi"]

                fig_2d = create_2d_scatter_plot_ndvi(
                    lat_arr,
                    lon_arr,
                    ndvi_mat,
                    sheet_name=chosen_sheet,
                    google_api_key=google_api_key
                )

                # Build DataFrame for 3D
                x_vals, y_vals, z_vals = [], [], []
                for i, latv in enumerate(lat_arr):
                    for j, lonv in enumerate(lon_arr):
                        x_vals.append(lonv)
                        y_vals.append(latv)
                        z_vals.append(ndvi_mat[i, j])
                df_3d = pd.DataFrame({"Longitud": x_vals, "Latitud": y_vals, "NDVI": z_vals})
                fig_3d = create_3d_surface_plot(df_3d, grid_size, color_map, z_scale, smoothness)

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
                    st.markdown("#### Time-Series 3D Animation")
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
                df_single = process_uploaded_file(uploaded_xlsx)
                if df_single is None:
                    st.error("Could not parse single sheet data.")
                    return
                # single-sheet fallback logic ...
                pass

        if show_proc and (processed_path is not None) and os.path.exists(processed_path):
            data_sheets = load_timeseries_data(processed_path)
            if data_sheets:
                sheet_list = list(data_sheets.keys())
                chosen_sheet_processed = st.selectbox(
                    "Select sheet for processed data (static 3D & 2D)",
                    sheet_list,
                    key="processed_sheet_selector"
                )

                col1, col2 = st.columns(2)
                with col1:
                    qgis_file = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "assets","data",
                        f"INFORME_{indice}_QGIS_{anio}.xlsx"
                    )
                    if not os.path.exists(qgis_file):
                        st.error(f"QGIS file not found => {qgis_file}")
                    else:
                        try:
                            xls = pd.ExcelFile(qgis_file)
                            if chosen_sheet_processed not in xls.sheet_names:
                                st.error(f"Sheet '{chosen_sheet_processed}' not in QGIS => {xls.sheet_names}")
                            else:
                                df_qgis = pd.read_excel(qgis_file, sheet_name=chosen_sheet_processed)
                                html_2d = create_2d_scatter_plot_ndvi_interactive_qgis(
                                    qgis_df=df_qgis,
                                    sheet_name=chosen_sheet_processed,
                                    google_api_key=google_api_key,
                                    margin_frac=0.05
                                )
                                if html_2d:
                                    st.components.v1.html(html_2d, height=600, scrolling=False)
                                else:
                                    st.error("Could not create interactive QGIS chart.")
                        except Exception as e:
                            st.error(f"Error reading QGIS => {e}")

                with col2:
                    lat_arr = data_sheets[chosen_sheet_processed]["lat"]
                    lon_arr = data_sheets[chosen_sheet_processed]["lon"]
                    ndvi_mat = data_sheets[chosen_sheet_processed]["ndvi"]

                    x_vals, y_vals, z_vals = [], [], []
                    for i, latv in enumerate(lat_arr):
                        for j, lonv in enumerate(lon_arr):
                            x_vals.append(lonv)
                            y_vals.append(latv)
                            z_vals.append(ndvi_mat[i, j])

                    df_3d = pd.DataFrame({"Longitud": x_vals, "Latitud": y_vals, "NDVI": z_vals})
                    fig_3d = create_3d_surface_plot(df_3d, grid_size, color_map, z_scale, smoothness)
                    if fig_3d:
                        fig_3d.update_layout(autosize=False, width=500, height=500)
                        st.plotly_chart(fig_3d, use_container_width=True, use_container_height=True)

                fig_time = create_3d_simulation_plot_time_interpolation(
                    data_sheets, grid_size, color_map, z_scale, smoothness, steps_value
                )
                if fig_time:
                    st.markdown("#### Time-Series 3D Animation")
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
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

    elif page_mode == "Risk Visualization":
        st.write("## Risk Visualization")

        # 1) HPC Data Option: The user can run HPC or re-use session data
        if "hpc_data" not in st.session_state:
            st.warning("No HPC data found in session. Click the button to run HPC with real data.")
            if st.button("Run HPC Pipeline"):
                try:
                    hpc_data = run_full_hpc_pipeline(indice, anio, base_folder="./upload_data")
                    if hpc_data is None:
                        st.error("HPC pipeline returned None. Check logs or file paths.")
                    else:
                        st.session_state["hpc_data"] = hpc_data
                        st.success("HPC pipeline completed! Data stored in session.")
                except Exception as e:
                    st.error(f"Error running HPC pipeline => {e}")
        else:
            st.info("HPC Data is already in session. Below you can visualize the results.")

        # 2) IDW/QGIS Visualization
        idw_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "assets", "data",
            f"INFORME_{indice}_IDW_{anio}.xlsx"
        )
        processed_disabled = True
        processed_path = None
        if "processed_idw_path" in st.session_state:
            if os.path.exists(st.session_state["processed_idw_path"]):
                processed_disabled = False
                processed_path = st.session_state["processed_idw_path"]
        elif os.path.exists(idw_file):
            processed_disabled = False
            processed_path = idw_file

        data_sheets = None
        if processed_path and os.path.exists(processed_path):
            data_sheets = load_timeseries_data(processed_path)

        if data_sheets:
            sheet_list = list(data_sheets.keys())
            chosen_sheet_processed = st.selectbox(
                "Select sheet for processed data (static 3D & 2D)",
                sheet_list,
                key="processed_sheet_selector"
            )

            # HPC data selection
            if "hpc_data" in st.session_state:
                hpc_data = st.session_state["hpc_data"]
                results = hpc_data.get("results", [])
                point_labels = [f" (Point={r['point_idx']})" for r in results]
                if point_labels:
                    chosen_point = st.selectbox("Select HPC point result", point_labels)
                    chosen_idx = point_labels.index(chosen_point)
                    HPC_info = results[chosen_idx]

            col1, col2 = st.columns(2)
            with col1:
                qgis_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "assets", "data",
                    f"INFORME_{indice}_QGIS_{anio}.xlsx"
                )
                if not os.path.exists(qgis_file):
                    st.error(f"QGIS file not found => {qgis_file}")
                else:
                    try:
                        xls = pd.ExcelFile(qgis_file)
                        if chosen_sheet_processed not in xls.sheet_names:
                            st.error(f"Sheet '{chosen_sheet_processed}' not in QGIS => {xls.sheet_names}")
                        else:
                            df_qgis = pd.read_excel(qgis_file, sheet_name=chosen_sheet_processed)
                            html_2d = create_2d_scatter_plot_ndvi_interactive_qgis(
                                qgis_df=df_qgis,
                                sheet_name=chosen_sheet_processed,
                                google_api_key=google_api_key,
                                margin_frac=0.05
                            )
                            if html_2d:
                                st.components.v1.html(html_2d, height=600, scrolling=False)
                            else:
                                st.error("Could not create interactive QGIS chart.")
                    except Exception as e:
                        st.error(f"Error reading QGIS => {e}")

            with col2:
                lat_arr = data_sheets[chosen_sheet_processed]["lat"]
                lon_arr = data_sheets[chosen_sheet_processed]["lon"]
                ndvi_mat = data_sheets[chosen_sheet_processed]["ndvi"]

                x_vals, y_vals, z_vals = [], [], []
                for i, latv in enumerate(lat_arr):
                    for j, lonv in enumerate(lon_arr):
                        x_vals.append(lonv)
                        y_vals.append(latv)
                        z_vals.append(ndvi_mat[i, j])

                st.markdown("## Monthly Risk Evolution")

                if "hpc_data" in st.session_state:
                    hpc_data = st.session_state["hpc_data"]
                    results = hpc_data.get("results", [])
                    if not results:
                        st.warning("HPC pipeline returned no points. Possibly QGIS file had no data.")
                    else:
                        # HPC_info from the selected point
                        XLDA = HPC_info["XLDA"]       # shape [1000 x 12]
                        VC = HPC_info["VC"]           # list of length 12
                        XInf2 = HPC_info["XInf"]      # shape [12 x 12]

                        fig = go.Figure()
                        n_months = XLDA.shape[1]

                        # Plot the filled-area KDE for HPC data
                        for m in range(1, n_months):
                            month_data = XLDA[:, m]
                            kde = gaussian_kde(month_data)
                            x_range = np.linspace(month_data.min(), month_data.max(), 200)
                            density = kde(x_range)
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=density,
                                mode='lines',
                                fill='tozeroy',
                                name=f'Month {m+1}',
                                hovertemplate="Month: %{text}<br>Loss: %{x:.2f}<br>Density: %{y:.2f}",
                                text=[f"Month {m+1}"] * len(x_range)
                            ))

                        fig.update_layout(
                            xaxis_title="Losses (USD/Month-Zone)",
                            yaxis_title="Density",
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True, use_container_height=True)
                else:
                    st.info("No HPC data loaded. Please click 'Run HPC Pipeline' above to compute real HPC results.")
        else:
            st.info("No IDW data to visualize or file not found. Upload or process .zip for NDVI data if needed.")

        st.markdown("---")

        # 3) HPC Risk Distributions & Table from HPC Data
        if "hpc_data" in st.session_state:
            hpc_data = st.session_state["hpc_data"]
            results = hpc_data.get("results", [])

            if not results:
                st.warning("HPC pipeline returned no points. Possibly QGIS file had no data.")
            else:
                # Just re-using HPC_info for the selected point in the selectbox
                XLDA = HPC_info["XLDA"]
                VC = HPC_info["VC"]
                XInf2 = HPC_info["XInf"]

                # Build HPC Table from XInf2
                # XInf2 columns (example indexing):
                #   0..4 => climate
                #   5 => skew
                #   6 => %C1
                #   7 => %C2
                #   8 => %C3
                #   9 => mean
                #   10 => 75%
                #   11 => 99%
                columns = [
                    "WD","Max C","Min  C","Viento (m/s)","Humedad (%)","Precip. (mm)",
                    f"{indice}","Skewness","%C1","%C2","%C3","Mean (units)",
                    "75% (units)","OpVar-99% (units)"
                ]

                n_months = XLDA.shape[1] if XLDA is not None else 12
                table_data = []
                for row_i in range(n_months):
                    wd       = VC[row_i] if row_i < len(VC) else ""
                    maxC     = XInf2[row_i, 0]
                    minC     = XInf2[row_i, 1]
                    viento   = XInf2[row_i, 2]
                    hum      = XInf2[row_i, 3]
                    prec     = XInf2[row_i, 4]
                    ndvi_val = XInf2[row_i, 8]   # or 9, depends on usage
                    skewv    = XInf2[row_i, 5]
                    pc1      = XInf2[row_i, 6]
                    pc2      = XInf2[row_i, 7]
                    pc3      = XInf2[row_i, 8]
                    mean_val = XInf2[row_i, 9]
                    p75_val  = XInf2[row_i, 10]
                    opv99    = XInf2[row_i, 11]

                    rowdata = [
                        wd, maxC, minC, viento, hum, prec,
                        ndvi_val, skewv, pc1, pc2, pc3,
                        mean_val, p75_val, opv99
                    ]
                    table_data.append(rowdata)

                df_hpc = pd.DataFrame(table_data, columns=columns)
                for col_ in df_hpc.columns.drop("WD"):
                    df_hpc[col_] = df_hpc[col_].astype(float).map("{:.3f}".format)

                st.markdown("### HPC Risk Data Table")
                st.dataframe(df_hpc)
        else:
            st.info("No HPC data loaded. Please click 'Run HPC Pipeline' above to compute real HPC results.")


# ---------------------- MAIN EXECUTION ENTRY ----------------------
if __name__ == "__main__":
    # You can configure logging here or in a separate logging_config file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    render_ui()
