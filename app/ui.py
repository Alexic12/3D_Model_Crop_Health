import streamlit as st
import logging
import pandas as pd
import numpy as np
import os

# Re-import the same modules you had
import streamlit.components.v1 as components

from data_processing import (
    process_uploaded_file,
    load_timeseries_data,
    invert_climate_file_rows,
    bulk_unzip_and_analyze_new_parallel
)

from app.visualization import (
    create_2d_scatter_plot_ndvi,
    create_2d_scatter_plot_ndvi_interactive_qgis,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation
)

# For the "Risk Visualization" 2D map with click:
import plotly.express as px
import plotly.graph_objs as go

logger = logging.getLogger(__name__)


def compute_risk_results_via_hpc(indice, anio):
    """
    Example function that calls your HPC pipeline code
    (the same code that was in your notebook).
    Then it reads the final 'Prospective_...' files or 'Prospective_LDA_...' files
    to build a 2D map of points + monthly distributions, etc.

    For demonstration, we will do something simple: 
    (1) Call run_full_hpc_pipeline() 
    (2) Then parse the final Excel outputs.
    (3) Return df_map, risk_info.

    If you want to run HPC only once, you might store the results in 'st.session_state'.
    """

    # 1) Optionally run your HPC code (which reads the input, inverts Excel, does Emision, etc.)
    #    This step is TOTALLY up to you if you want to re-run the HPC pipeline each time or only once.
    #    We keep it commented out for performance reasons. 
    # run_full_hpc_pipeline()

    # 2) Suppose after HPC, we have these final outputs:
    #    '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/{indice}_{anio}/Prospective_{indice}_{int(anio)+1}.xlsx'
    #    '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/{indice}_{anio}/Prospective_LDA_{indice}_{int(anio)+1}.xlsx'
    #
    # We would parse them to get per-point monthly distributions. 
    # For demonstration, we'll just mock it:

    # Suppose we have 5 points, as in the previous placeholder:
    df_map = pd.DataFrame({
        "point_id": [0,1,2,3,4],
        "Lon": [-74.05, -74.02, -74.00, -73.98, -73.95],
        "Lat": [4.70, 4.72, 4.69, 4.73, 4.71],
        "NDVI": [0.66, 0.72, 0.60, 0.55, 0.80]
    })

    risk_info = {}
    np.random.seed(42)
    for pid in df_map["point_id"]:
        monthly_samples = np.random.normal(loc=(pid+1)*100, scale=15, size=(1000, 12))
        dfm = pd.DataFrame({
            "WD": [f"WD_{m}" for m in range(12)],
            "Max C": np.random.uniform(10, 30, 12),
            "Min  C": np.random.uniform(5, 15, 12),
            "Viento (m/s)": np.random.uniform(0.5, 5.0, 12),
            "Humedad (%)": np.random.uniform(40, 90, 12),
            "Precip. (mm)": np.random.uniform(0, 30, 12),
            indice: np.random.uniform(0.4, 0.9, 12),
            "Skewness": np.random.uniform(0, 1, 12),
            "%C1": np.random.uniform(0, 1, 12),
            "%C2": np.random.uniform(0, 1, 12),
            "%C3": np.random.uniform(0, 1, 12),
            "Mean (USD)": np.random.uniform(500, 2000, 12),
            "75% (USD)": np.random.uniform(2000, 3000, 12),
            "OpVar-99.9% (USD)": np.random.uniform(3000, 4000, 12)
        })
        dfm["Max C"] = dfm["Max C"].map("{:.3f}".format)
        dfm["Min  C"] = dfm["Min  C"].map("{:.3f}".format)

        risk_info[pid] = {
            "monthly_distribution": monthly_samples,
            "df_table": dfm
        }

    return df_map, risk_info

def render_ui():
    st.title("Crop Health Visualization (Production Version)")

    st.write("---")
    page_mode = st.radio(
        "Select Visualization Page",
        ["3D Visualization", "Risk Visualization"],
        index=0
    )
    st.write("---")

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

        google_api_key = st.text_input("Google Maps API Key", value="AIzaSy...FAKE_EXAMPLE")

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

    if page_mode == "3D Visualization":
        st.write("## Single or Multi-Sheet NDVI Visualization")

        uploaded_xlsx = st.file_uploader("Upload Excel for NDVI Visuals", type=["xlsx", "xls"])
        simulate_button = st.button("Visualize NDVI")

        idw_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
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
                        os.path.dirname(os.path.dirname(__file__)),
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

        # Locate processed file (IDW)
        idw_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
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

        # Try to load processed timeseries data
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
                    os.path.dirname(os.path.dirname(__file__)),
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
                from data_processing import Prospectiva

                # For demonstration purposes, create dummy input values.
                n_var = 5
                n_components = 5
                # Create dummy climate input data (as a DataFrame) with n_var+1 columns.
                dummy_XD = pd.DataFrame(np.random.rand(100, n_var + 1))
                # Dummy risk clusters vector (for a given point)
                dummy_XCr = np.random.randint(0, n_components, size=(100, 1))
                # Dummy pattern for each variable (5 variables x 12 months)
                dummy_V = np.random.randint(0, 5, size=(n_var, 12))
                # Dummy transition matrix and emission matrices
                dummy_aTr = np.random.rand(n_components, n_components)
                dummy_bEm = np.random.rand(n_var, n_components, n_components)
                # Dummy monthly climate information (12 months x 5 variables)
                dummy_ydmes = np.random.rand(12, n_var)
                # Run Prospectiva to obtain risk evolution outputs:
                VC, XInf2, XLDA = Prospectiva(0, dummy_XD, dummy_XCr, dummy_V, dummy_aTr, dummy_bEm, dummy_ydmes)

                # --- Build an interactive risk density plot using Plotly ---
                st.markdown("### Risk Evolution Smooth Density Curves")
                import plotly.graph_objects as go
                from scipy.stats import gaussian_kde

                fig = go.Figure()

                # For each month (assumed to be 12 months)
                for m in range(dummy_V.shape[1]):
                    # Get the loss data for month m
                    month_data = XLDA[:, m]
                    
                    # Compute the kernel density estimate
                    kde = gaussian_kde(month_data)
                    # Define an x-range from min to max of the data
                    x_range = np.linspace(month_data.min(), month_data.max(), 200)
                    density = kde(x_range)
                    
                    # Add a smooth line trace
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=density,
                        mode='lines',
                        name=f'Month {m+1}',
                        hovertemplate="Month: %{text}<br>Loss: %{x:.2f}<br>Density: %{y:.2f}",
                        text=[f"Month {m+1}"] * len(x_range)
                    ))

                fig.update_layout(
                    title="Monthly Risk Evolution (Smooth Density)",
                    xaxis_title="Losses (USD/Month-Zone)",
                    yaxis_title="Density"
                )

                st.plotly_chart(fig, use_container_width=True)
                    

        # Below the row: Create and display a dummy risk data table
        indice = "Dummy_Index"
        num_rows = 10  # change as needed
        VC_dummy = np.random.rand(num_rows)
        XInf2_dummy = np.random.rand(num_rows, 9)
        NDVI_dummy = np.random.rand(num_rows)
        MPerd_dummy = np.random.rand(num_rows, 3)

        dfm = pd.DataFrame(np.column_stack((VC_dummy, XInf2_dummy[:, 0:5], NDVI_dummy, XInf2_dummy[:, 5], XInf2_dummy[:, 6:9], MPerd_dummy)))
        dfm.columns = [
            'WD', 'Max C', 'Min  C', 'Viento (m/s)', 'Humedad (%)', 'Precip. (mm)',
            str(indice), 'Skewness', '%C1', '%C2', '%C3', 'Mean (USD)', '75% (USD)', 'OpVar-99.9% (USD)'
        ]
        dfm['Max C'] = dfm['Max C'].astype(float).map('{:.3f}'.format)
        dfm['Min  C'] = dfm['Min  C'].astype(float).map('{:.3f}'.format)
        dfm['Viento (m/s)'] = dfm['Viento (m/s)'].astype(float).map('{:.3f}'.format)
        dfm['Humedad (%)'] = dfm['Humedad (%)'].astype(float).map('{:.3f}'.format)
        dfm['Precip. (mm)'] = dfm['Precip. (mm)'].astype(float).map('{:.3f}'.format)
        dfm['Skewness'] = dfm['Skewness'].astype(float).map('{:.4f}'.format)
        dfm[str(indice)] = dfm[str(indice)].astype(float).map('{:.4f}'.format)
        dfm['%C1'] = dfm['%C1'].astype(float).map('{:.3f}'.format)
        dfm['%C2'] = dfm['%C2'].astype(float).map('{:.3f}'.format)
        dfm['%C3'] = dfm['%C3'].astype(float).map('{:.3f}'.format)
        dfm['Mean (USD)'] = dfm['Mean (USD)'].astype(float).map('{:.2f}'.format)
        dfm['75% (USD)'] = dfm['75% (USD)'].astype(float).map('{:.2f}'.format)
        dfm['OpVar-99.9% (USD)'] = dfm['OpVar-99.9% (USD)'].astype(float).map('{:.2f}'.format)

        st.markdown("### Dummy Risk Data Table")
        st.dataframe(dfm)

        # --- Run HPC risk simulation (Prospectiva) ---
        # Import the function from your data_processing module


        

  