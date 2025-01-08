
import streamlit as st
from data_processing import process_uploaded_file
from visualization import create_3d_surface_plot, create_3d_simulation_plot_sea_keypoints
import logging

logger = logging.getLogger(__name__)

'''
def render_ui():
    st.title("Crop Health Visualization")
    st.write("Upload an Excel file containing crop health data (-1 to 1).")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        logger.info("File uploaded by user")
        # Process the uploaded file
        data = process_uploaded_file(uploaded_file)

        if data is not None:
            # Create and display the 3D surface plot
            fig = create_3d_surface_plot(data)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create the 3D surface plot.")
        else:
            st.error("Failed to process the file. Please check the data format.")

'''


def render_ui():
    st.title("Crop Health Visualization")
    st.write("Upload an Excel file containing crop health data (-1 to 1).")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        grid_size = st.slider("Grid Resolution", min_value=50, max_value=400, value=200, step=10)
        z_scale = st.slider("Z-axis Scale", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        smoothness = st.slider("Surface Smoothness", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        color_map = st.selectbox("Color Map", options=["Viridis", "Plasma", "Inferno", "Magma"])

        # **New**: button to trigger the simulation
        simulate_button = st.button("View Simulation")
    
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Process the uploaded file
        data = process_uploaded_file(uploaded_file)

        if data is not None:
            if simulate_button:
                # **New**: Create animation if the button is clicked
                """

                fig_sim = create_3d_simulation_plot_sea(
                    data=data,
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness,
                    key_frames=20,      # fewer frames => smaller data
                    wave_count=5,       # adjust as desired
                    wave_amplitude_range=(0.05, 0.1),
                    wave_frequency_range=(0.05, 0.1),
                    wave_speed_range=(0.1, 2.0),
                    random_seed=42
                )
                """
                fig_sim = create_3d_simulation_plot_sea_keypoints(
                    data,
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness,
                    key_frames=100,                # Number of animation frames
                    key_points=20,                # How many anchor points to sample
                    wave_amplitude_range=(0.05, 0.4),
                    wave_frequency_range=(0.05, 0.2),
                    wave_speed_range=(0.5, 2.0),
                    influence_sigma=0.05,         # Larger => smoother, more blended waves
                    random_seed=42
                )
                if fig_sim is not None:
                    st.plotly_chart(fig_sim, use_container_width=True)
                else:
                    st.error("Failed to create the 3D simulation plot.")
            else:
                # Create and display the static 3D surface plot
                fig = create_3d_surface_plot(
                    data=data,
                    grid_size=grid_size,
                    color_map=color_map,
                    z_scale=z_scale,
                    smoothness=smoothness
                )
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create the 3D surface plot.")
        else:
            st.error("Failed to process the file. Please check the data format.")

