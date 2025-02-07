import streamlit as st
from data_processing import process_uploaded_file, load_timeseries_data
from visualization import (
    create_2d_scatter_plot_ndvi,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation
)
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def render_ui():
    st.title("Crop Health Visualization")
    st.write("Upload an Excel file containing multiple sheets or a single sheet (old approach).")

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("Settings")
        grid_size = st.slider("Grid Resolution", min_value=50, max_value=400, value=50, step=10)
        z_scale = st.slider("Z-axis Scale", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        smoothness = st.slider("Surface Smoothness", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        color_map = st.selectbox("Color Map", options=["Viridis", "Plasma", "Inferno", "Magma"])

        # Time-series interpolation controls
        interpolate_checkbox = st.checkbox("Enable Interpolation between sheets?", value=True)
        steps_value = st.slider("Number of interpolation steps", min_value=1, max_value=20, value=10, step=1)

        # Button to trigger time-series simulation
        simulate_button = st.button("View Simulation")
    
    # --- MAIN CONTENT ---
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # 1) Try loading multi-sheet matrix-based data
        data_sheets = load_timeseries_data(uploaded_file)

        if data_sheets is not None and len(data_sheets) > 0:
            # We have multiple sheets in the order they appear in the Excel file
            sheet_names_in_order = list(data_sheets.keys())  # insertion order in Python 3.7+

            # --- Optional: Debugging info ---
            if st.checkbox("Show time-series data (for debugging)"):
                for sname in sheet_names_in_order:
                    st.subheader(f"Sheet: {sname}")
                    st.write("Longitude array:", data_sheets[sname]["lon"])
                    st.write("Latitude array:", data_sheets[sname]["lat"])
                    ndvi_matrix = data_sheets[sname]["ndvi"]
                    st.write(f"NDVI matrix shape: {ndvi_matrix.shape}")
                    st.write(f"NDVI min: {ndvi_matrix.min()}, max: {ndvi_matrix.max()}")
                    st.dataframe(pd.DataFrame(ndvi_matrix))

            # Let user pick which sheet to show in the static 3D
            chosen_sheet = st.selectbox("Select sheet for Static 3D & 2D View", sheet_names_in_order)
            sheet_data = data_sheets[chosen_sheet]

            # --- 2D SCATTER ---
            lat_array = sheet_data["lat"]
            lon_array = sheet_data["lon"]
            ndvi_2d   = sheet_data["ndvi"]

            

            # --- STATIC 3D ---
            x_vals = []
            y_vals = []
            z_vals = []

            # Flatten NDVI into x,y,z columns for create_3d_surface_plot
            for i, lat_val in enumerate(lat_array):
                for j, lon_val in enumerate(lon_array):
                    x_vals.append(lon_val)
                    y_vals.append(lat_val)
                    z_vals.append(ndvi_2d[i, j])

            # Construct a DataFrame with columns expected by create_3d_surface_plot
            df_static = pd.DataFrame({
                "Longitud": x_vals,
                "Latitud": y_vals,
                "NDVI": z_vals,
                "Riesgo": [1]*len(x_vals)  # dummy
            })

            fig_static = create_3d_surface_plot(
                data=df_static,
                grid_size=grid_size,
                color_map=color_map,
                z_scale=z_scale,
                smoothness=smoothness
            )
            if fig_static:
                st.plotly_chart(fig_static, use_container_width=True)
            else:
                st.error(f"Failed to create static 3D surface plot for sheet: {chosen_sheet}.")

            # --- TIME-SERIES SIMULATION ---
            if simulate_button:
                # If interpolation is disabled, only 1 step => jump from sheet to sheet
                final_steps = steps_value if interpolate_checkbox else 1

                fig_sim = create_3d_simulation_plot_time_interpolation(
                    data_sheets=data_sheets,
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness,
                    steps_between_sheets=final_steps
                )
                if fig_sim:
                    st.plotly_chart(fig_sim, use_container_width=True)
                else:
                    st.error("Failed to create the 3D time-series simulation plot.")

            fig_2d = create_2d_scatter_plot_ndvi(
                lat=lat_array,
                lon=lon_array,
                ndvi_matrix=ndvi_2d,
                sheet_name=chosen_sheet
            )
            
            if fig_2d:
                st.pyplot(fig_2d)
            else:
                st.warning(f"Failed to create 2D NDVI scatter plot for sheet: {chosen_sheet}.")

        else:
            # If multi-sheet parse failed or file is single-sheet with columns [Longitud, Latitud, NDVI, Riesgo]
            data_old = process_uploaded_file(uploaded_file)
            if data_old is not None:
                st.warning("Loaded with old approach (single sheet with required columns).")
                if simulate_button:
                    # Old wave-based approach
                    fig_sim = create_3d_simulation_plot_sea_keypoints(
                        data_old,
                        grid_size=grid_size,
                        color_map=color_map,
                        z_scale=z_scale,
                        smoothness=smoothness,
                        key_frames=50,  # number of frames
                        key_points=10,
                        wave_amplitude_range=(0.05, 0.4),
                        wave_frequency_range=(0.05, 0.2),
                        wave_speed_range=(0.5, 2.0),
                        influence_sigma=0.05,
                        random_seed=42
                    )
                    if fig_sim:
                        st.plotly_chart(fig_sim, use_container_width=True)
                    else:
                        st.error("Failed to create the wave-based 3D simulation plot.")
                else:
                    # Just a static 3D
                    fig_static_old = create_3d_surface_plot(data_old)
                    if fig_static_old:
                        st.plotly_chart(fig_static_old, use_container_width=True)
                    else:
                        st.error("Failed to create the static 3D surface plot.")
            else:
                st.error("Failed to load data with either new or old approach.")
