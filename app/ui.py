
import streamlit as st
from data_processing import process_uploaded_file
from visualization import create_3d_surface_plot
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
        grid_size = st.slider("Grid Resolution", min_value=50, max_value=200, value=100, step=10)
        color_map = st.selectbox("Color Map", options=["Viridis", "Plasma", "Inferno", "Magma"])
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Process the uploaded file
        data = process_uploaded_file(uploaded_file)

        if data is not None:
            # Create and display the 3D surface plot
            fig = create_3d_surface_plot(data, grid_size=grid_size, color_map=color_map)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to create the 3D surface plot.")
        else:
            st.error("Failed to process the file. Please check the data format.")