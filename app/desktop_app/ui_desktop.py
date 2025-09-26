"""
Streamlit UI for Crop Health Visualization.

Key fixes for deployment:
- Responsive right sidebar (no fixed 400px margin). Uses padding-right with media queries.
- All figures use container width; Plotly margins trimmed.
- HTML component (interactive QGIS/Google map) is embedded with scrolling=True and inside a wrapper that can overflow horizontally if needed.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import gaussian_kde

# Configuration import
from app.config.config import settings

# HPC imports
from app.data.data_processing import (
    process_uploaded_file,
    load_timeseries_data,
    invert_climate_file_rows,
    bulk_unzip_and_analyze_new_parallel,
    Prospectiva,  # noqa: F401  (kept for users that call it elsewhere)
    run_full_hpc_pipeline,
)
from app.data.ghg_capture import (
    process_ghg_data,
    create_risk_matrix_heatmap,
    create_lda_distribution_plot,
    create_management_matrix_heatmap,
    create_cost_benefit_chart,
)

# Visualization helpers
from app.ui.visualization import (
    create_2d_scatter_plot_ndvi,
    create_2d_scatter_plot_ndvi_interactive_qgis,
    create_3d_surface_plot,
    create_3d_simulation_plot_sea_keypoints,
    create_3d_simulation_plot_time_interpolation,
)

logger = logging.getLogger(__name__)


def _responsive_css() -> None:
    """Inject responsive CSS with mobile-first design and accessibility features."""
    st.markdown(
        """
        <style>
          /* CSS Custom Properties for Design System */
          :root {
              --primary-color: #1f77b4;
              --secondary-color: #ff7f0e;
              --success-color: #2ca02c;
              --warning-color: #ff7f0e;
              --error-color: #d62728;
              --background-color: #ffffff;
              --surface-color: #f8f9fa;
              --text-primary: #212529;
              --text-secondary: #6c757d;
              --border-color: #dee2e6;
              --border-radius: 8px;
              --spacing-xs: 0.25rem;
              --spacing-sm: 0.5rem;
              --spacing-md: 1rem;
              --spacing-lg: 1.5rem;
              --spacing-xl: 2rem;
              --font-size-sm: 0.875rem;
              --font-size-base: 1rem;
              --font-size-lg: 1.125rem;
              --font-size-xl: 1.25rem;
              --line-height-base: 1.5;
              --transition-base: 0.15s ease-in-out;
          }

          /* Dark Mode Support */
          @media (prefers-color-scheme: dark) {
              :root {
                  --background-color: #121212;
                  --surface-color: #1e1e1e;
                  --text-primary: #ffffff;
                  --text-secondary: #b3b3b3;
                  --border-color: #333333;
              }
          }

          /* Mobile First Base Styles */
          .main .block-container {
              padding: var(--spacing-sm);
              max-width: 100%;
              margin: 0 auto;
          }

          /* Responsive Sidebar */
          [data-testid="stSidebar"] {
              background-color: var(--surface-color);
              border-right: 1px solid var(--border-color);
              transition: transform var(--transition-base);
              overflow-y: auto;
              overflow-x: hidden;
          }
          
          [data-testid="stSidebar"] > div {
              overflow-y: auto;
              overflow-x: hidden;
              height: 100vh;
          }
          
          /* Force sidebar to right on all screen sizes */
          [data-testid="stSidebar"] > div {
              border-left: 1px solid var(--border-color);
              border-right: none;
          }

          /* Mobile: Collapsible sidebar */
          @media (max-width: 768px) {
              [data-testid="stSidebar"] {
                  position: fixed;
                  top: 0;
                  left: -100%;
                  height: 100vh;
                  width: 280px;
                  z-index: 1000;
                  transform: translateX(-100%);
              }
              
              [data-testid="stSidebar"].sidebar-open {
                  transform: translateX(0);
              }
              
              .main .block-container {
                  padding: var(--spacing-sm);
              }
          }

          /* Tablet Styles */
          @media (min-width: 769px) and (max-width: 1024px) {
              [data-testid="stSidebar"] {
                  position: fixed !important;
                  right: 0 !important;
                  left: auto !important;
                  width: 280px;
                  height: 100vh;
                  top: 0;
                  z-index: 999;
              }
              
              .main .block-container {
                  padding: var(--spacing-md) !important;
                  margin-right: 320px !important;
                  margin-left: var(--spacing-md) !important;
                  max-width: calc(100vw - 360px) !important;
              }
              
              .main {
                  margin-right: 280px !important;
              }
          }

          /* Desktop Styles */
          @media (min-width: 1025px) {
              [data-testid="stSidebar"] {
                  position: fixed !important;
                  right: 0 !important;
                  left: auto !important;
                  top: 0;
                  height: 100vh;
                  width: 320px;
                  box-shadow: -2px 0 8px rgba(0,0,0,0.1);
                  z-index: 999;
              }
              
              .main .block-container {
                  padding: var(--spacing-lg) !important;
                  margin-right: 360px !important;
                  margin-left: var(--spacing-lg) !important;
                  max-width: calc(100vw - 400px) !important;
                  width: calc(100vw - 400px) !important;
              }
              
              .main {
                  margin-right: 320px !important;
              }
          }

          /* Large Desktop */
          @media (min-width: 1400px) {
              .main .block-container {
                  max-width: calc(1200px - 320px) !important;
                  margin-right: 360px !important;
                  margin-left: var(--spacing-lg) !important;
              }
          }
          
          /* Ensure all content respects sidebar */
          @media (min-width: 769px) {
              .stApp > div:first-child {
                  margin-right: 320px !important;
              }
              
              [data-testid="stAppViewContainer"] {
                  margin-right: 320px !important;
              }
              
              .main > div {
                  margin-right: 0 !important;
              }
          }

          /* Responsive Typography */
          h1, h2, h3, h4, h5, h6 {
              line-height: var(--line-height-base);
              margin-bottom: var(--spacing-md);
              color: var(--text-primary);
          }

          /* Responsive Components */
          .stApp iframe {
              width: 100% !important;
              border-radius: var(--border-radius);
          }

          .overflow-wrap {
              width: 100%;
              overflow: auto;
              border-radius: var(--border-radius);
          }

          /* Hide skip navigation links */
          .skip-link {
              display: none !important;
          }
          
          [data-testid="stHeader"] {
              display: none !important;
          }
          
          /* Accessibility Improvements */
          button:focus,
          .stSelectbox > div > div:focus,
          .stTextInput > div > div > input:focus {
              outline: 2px solid var(--primary-color);
              outline-offset: 2px;
          }

          /* High Contrast Mode */
          @media (prefers-contrast: high) {
              :root {
                  --border-color: #000000;
                  --text-primary: #000000;
              }
          }

          /* Reduced Motion */
          @media (prefers-reduced-motion: reduce) {
              * {
                  animation-duration: 0.01ms !important;
                  animation-iteration-count: 1 !important;
                  transition-duration: 0.01ms !important;
              }
          }

          /* Touch-friendly Interactive Elements */
          @media (hover: none) and (pointer: coarse) {
              button, .stSelectbox, .stTextInput {
                  min-height: 44px;
              }
          }

          /* Fix Dropdown Text Visibility - Comprehensive */
          .stSelectbox label {
              color: var(--text-primary) !important;
          }
          
          .stSelectbox > div > div {
              color: var(--text-primary) !important;
              background-color: var(--background-color) !important;
          }
          
          .stSelectbox > div > div > div {
              color: var(--text-primary) !important;
              background-color: var(--background-color) !important;
          }
          
          .stSelectbox select {
              color: var(--text-primary) !important;
              background-color: var(--background-color) !important;
          }
          
          .stSelectbox option {
              color: var(--text-primary) !important;
              background-color: var(--background-color) !important;
          }
          
          /* Target all possible selectbox elements */
          [data-testid="stSelectbox"] {
              color: var(--text-primary) !important;
          }
          
          [data-testid="stSelectbox"] > div {
              color: var(--text-primary) !important;
          }
          
          [data-testid="stSelectbox"] div {
              color: var(--text-primary) !important;
          }
          
          [data-testid="stSelectbox"] span {
              color: var(--text-primary) !important;
          }
          
          /* Force text color on all child elements */
          .stSelectbox * {
              color: var(--text-primary) !important;
          }
          
          /* Ensure dropdown arrow is visible */
          .stSelectbox > div > div::after {
              border-color: var(--text-primary) transparent transparent transparent !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_ui() -> None:
    st.set_page_config(
        page_title="Crop Health Visualization",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.streamlit.io',
            'Report a bug': 'mailto:support@crophealth.com',
            'About': 'Crop Health Visualization Platform v2.0'
        }
    )
    _responsive_css()
    
    # Initialize accessibility features
    from app.ui.accessibility import initialize_accessibility
    initialize_accessibility()
    
    # Add viewport meta tag for mobile responsiveness
    st.markdown(
        '<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">',
        unsafe_allow_html=True
    )
    
    # Initialize responsive layout
    from app.ui.responsive_components import ResponsiveLayout
    ResponsiveLayout.responsive_container()

    st.title("Crop Health Visualization")
    st.write("---")
    page_mode = st.radio("Select Visualization Page", ["3D Visualization", "Prospective Visualization", "Risk Management"], index=0)
    st.write("---")

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("Settings")

        # Get available fields from existing folders
        base_folder = "./upload_data"
        available_fields = []
        if os.path.exists(base_folder):
            for item in os.listdir(base_folder):
                item_path = os.path.join(base_folder, item)
                if os.path.isdir(item_path):
                    available_fields.append(item)
        
        # Field selection dropdown - check for auto-detected field
        auto_detected = st.session_state.get("auto_detected_field", None)
        current_field = st.session_state.get("current_field_name", None)
        
        # Use current field name if available
        if current_field:
            field_name = current_field
            if current_field not in available_fields:
                available_fields.append(current_field)
        elif auto_detected:
            field_name = auto_detected
            if auto_detected not in available_fields:
                available_fields.append(auto_detected)
        
        if available_fields:
            # Set default to current/auto-detected field if available
            default_idx = 0
            if current_field and current_field in available_fields:
                default_idx = available_fields.index(current_field)
            elif auto_detected and auto_detected in available_fields:
                default_idx = available_fields.index(auto_detected)
            
            selected_field = st.selectbox("Select Field", available_fields + ["Create New Field"], index=default_idx)
            if selected_field == "Create New Field":
                field_name = st.text_input("New Field Name", value="")
            else:
                field_name = selected_field
        else:
            field_name = st.text_input("Field Name", value="Perimetro Prev")
        
        indice = st.text_input("Vegetation Index", value="NDVI")
        anio = st.text_input("Year", value="2024")
        st.write("---")

        # Google Maps API Key from configuration
        google_api_key = settings.GOOGLE_MAPS_API_KEY
        if not google_api_key:
            st.info("No GOOGLE_MAPS_API_KEY configured. Interactive basemap may be limited.")

        st.subheader("Bulk NDVI ZIP Analysis")
        uploaded_files = st.file_uploader(
            "Upload .zip pairs (base + ColorMap) and climate Excel file", 
            type=["zip", "xlsx", "xls"], 
            accept_multiple_files=True
        )
        
        import re
        
        if uploaded_files:
            # Separate ZIP files and Excel files
            zip_files = [f for f in uploaded_files if f.name.lower().endswith('.zip')]
            excel_files = [f for f in uploaded_files if f.name.lower().endswith(('.xlsx', '.xls'))]
            
            if zip_files:
                # Extract field name from first ZIP file using improved regex patterns
                first_zip = zip_files[0].name
            detected_field = None
            
            # Try multiple patterns to detect field name (matching data_processing.py)
            patterns = [
                # Pattern 1: "001. Campo_Luna_Roja_NDVI_31ene2022.zip"
                r'\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI',
                # Pattern 2: "001. perimetro__prev_NDVI_31ene2022.zip"
                r'\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI',
                # Pattern 3: Generic field extraction before NDVI
                r'\d+\.\s*(.+?)_NDVI'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, first_zip, re.IGNORECASE)
                if match:
                    raw_field = match.group(1)
                    # Clean up the field name - keep underscores for folder names
                    detected_field = raw_field.replace('__', '_').strip().title().replace(' ', '_')
                    break
            
            if detected_field:
                # Always use detected field name, override current selection
                field_name = detected_field
                st.info(f"Auto-detected field: {detected_field}. Files ready for processing.")
                # Store in session state to update dropdown and files
                st.session_state["auto_detected_field"] = detected_field
                st.session_state["uploaded_zip_files"] = zip_files
                st.session_state["current_field_name"] = detected_field
                # Only rerun once to update dropdown, then set flag to prevent further reruns
                if not st.session_state.get("field_updated", False):
                    st.session_state["field_updated"] = True
                    st.rerun()
            else:
                st.session_state["uploaded_zip_files"] = zip_files
                st.info("Files ready for processing. Click 'Run Bulk Analysis' to process.")
            
            # Handle Excel files (climate data)
            if excel_files:
                st.session_state["uploaded_excel_files"] = excel_files
                climate_names = [f.name for f in excel_files]
                st.info(f"Climate files detected: {', '.join(climate_names)}")

        if st.button("Run Bulk Analysis"):
            if not field_name:
                st.error("Please select or enter a field name.")
            elif "uploaded_zip_files" not in st.session_state:
                st.error("No ZIP files uploaded. Please upload files first.")
            else:
                # Use auto-detected field name if available
                if "current_field_name" in st.session_state:
                    field_name = st.session_state["current_field_name"]
                    st.info(f"Using detected field name: {field_name}")
                
                # Extract field name from ZIP files as backup using improved patterns
                zip_files = st.session_state["uploaded_zip_files"]
                if not field_name or field_name in ["NDVI_2024", "Create New Field"]:
                    first_zip = zip_files[0].name
                    detected_field = None
                    
                    # Try multiple patterns to detect field name
                    patterns = [
                        # Pattern 1: "001. Campo_Luna_Roja_NDVI_31ene2022.zip"
                        r'\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI',
                        # Pattern 2: "001. perimetro__prev_NDVI_31ene2022.zip"
                        r'\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI',
                        # Pattern 3: Generic field extraction before NDVI
                        r'\d+\.\s*(.+?)_NDVI'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, first_zip, re.IGNORECASE)
                        if match:
                            raw_field = match.group(1)
                            # Clean up the field name - keep underscores for folder names
                            detected_field = raw_field.replace('__', '_').strip().title().replace(' ', '_')
                            break
                    
                    if detected_field:
                        field_name = detected_field
                        st.info(f"Re-detected field name from ZIP: {field_name}")
                
                # First, save the uploaded files
                field_folder = os.path.join(base_folder, field_name)
                subfolder = os.path.join(field_folder, indice, anio)
                os.makedirs(subfolder, exist_ok=True)
                
                # Save ZIP files
                for zf in zip_files:
                    outpath = os.path.join(subfolder, zf.name)
                    with open(outpath, "wb") as f:
                        f.write(zf.getbuffer())
                
                # Save Excel files (climate data)
                if "uploaded_excel_files" in st.session_state:
                    excel_files = st.session_state["uploaded_excel_files"]
                    for ef in excel_files:
                        # Save with standard climate naming convention
                        climate_name = f"Clima_{indice}_{anio}.xlsx"
                        outpath = os.path.join(subfolder, climate_name)
                        with open(outpath, "wb") as f:
                            f.write(ef.getbuffer())
                        st.success(f"Climate file saved as: {climate_name}")
                
                st.success(f"All files saved to {field_name}/{indice}/{anio}.")
                
                # Then process the files
                esp_xlsx, idw_xlsx, qgis_xlsx = bulk_unzip_and_analyze_new_parallel(
                    indice, anio, base_folder=field_folder
                )
                if esp_xlsx and os.path.exists(esp_xlsx):
                    st.success(f"Espacial => {esp_xlsx}")
                if idw_xlsx and os.path.exists(idw_xlsx):
                    st.success(f"IDW => {idw_xlsx}")
                    st.session_state["processed_idw_path"] = idw_xlsx
                    st.session_state["current_field"] = field_name
                if qgis_xlsx and os.path.exists(qgis_xlsx):
                    st.success(f"QGIS => {qgis_xlsx}")
                
                # Clear uploaded files and auto-detected field after processing
                if "uploaded_zip_files" in st.session_state:
                    del st.session_state["uploaded_zip_files"]
                if "uploaded_excel_files" in st.session_state:
                    del st.session_state["uploaded_excel_files"]
                if "auto_detected_field" in st.session_state:
                    del st.session_state["auto_detected_field"]
                if "current_field_name" in st.session_state:
                    del st.session_state["current_field_name"]
                if "field_updated" in st.session_state:
                    del st.session_state["field_updated"]
                # Trigger rerun to refresh dropdown
                st.rerun()

        # Always show download buttons if files exist
        if field_name:
            output_dir = os.path.join("assets", "data", field_name)
        else:
            output_dir = os.path.join("assets", "data")
        esp_file = os.path.join(output_dir, f"INFORME_{indice}_Espacial_{anio}.xlsx")
        idw_file = os.path.join(output_dir, f"INFORME_{indice}_IDW_{anio}.xlsx")
        qgis_file = os.path.join(output_dir, f"INFORME_{indice}_QGIS_{anio}.xlsx")
        
        st.write("---")
        st.subheader("Download Processed Files")
        if os.path.exists(esp_file):
            with open(esp_file, "rb") as f:
                st.download_button("Download Espacial", data=f, file_name=os.path.basename(esp_file), key="dl_esp")
        if os.path.exists(idw_file):
            with open(idw_file, "rb") as f:
                st.download_button("Download IDW", data=f, file_name=os.path.basename(idw_file), key="dl_idw")
        if os.path.exists(qgis_file):
            with open(qgis_file, "rb") as f:
                st.download_button("Download QGIS", data=f, file_name=os.path.basename(qgis_file), key="dl_qgis")

        with st.expander("Invert Climate Excel Rows"):
            climate_file = st.file_uploader("Upload Climate Excel", type=["xlsx", "xls"])
            if climate_file:
                df_inv = invert_climate_file_rows(climate_file)
                if df_inv is not None:
                    st.dataframe(df_inv.head(20))
                    # Important: to_excel returns bytes only if using a BytesIO; streamlit handles file-like objects best.
                    from io import BytesIO

                    buf = BytesIO()
                    df_inv.to_excel(buf, index=False)
                    buf.seek(0)
                    st.download_button(
                        "Download Inverted Excel",
                        data=buf,
                        file_name="clima_inverted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        with st.expander("Advanced Sliders", expanded=False):
            grid_size = st.slider("Grid Resolution", 5, 300, 50, step=5)
            z_scale = st.slider("Z Scale", 0.1, 2.0, 1.0, step=0.1)
            smoothness = st.slider("Surface Smoothness (Gaussian)", 0.0, 10.0, 1.0, step=0.1)
            color_map = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "cividis"])
            steps_value = st.slider("Time interpolation steps", 1, 20, 10)
            st.write("---")

    # ---------------- MAIN PAGE CONTENT ----------------
    # Responsive navigation for mobile
    if st.session_state.get('viewport_width', 1200) < 768:
        with st.expander("📱 Navigation Menu", expanded=False):
            page_mode = st.radio("Select Page", ["3D Visualization", "Prospective Visualization", "Risk Management"], horizontal=True)
    
    if page_mode == "3D Visualization":

        # Check for an existing processed IDW path
        current_field = st.session_state.get("current_field", field_name)
        if current_field:
            idw_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets",
                "data",
                current_field,
                f"INFORME_{indice}_IDW_{anio}.xlsx",
            )
        else:
            idw_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets",
                "data",
                f"INFORME_{indice}_IDW_{anio}.xlsx",
            )
        processed_disabled = True
        processed_path = None
        if "processed_idw_path" in st.session_state and os.path.exists(st.session_state["processed_idw_path"]):
            processed_disabled = False
            processed_path = st.session_state["processed_idw_path"]
        elif os.path.exists(idw_file):
            processed_disabled = False
            processed_path = idw_file

        visualize_processed_btn = st.button("Visualize Processed Data", disabled=processed_disabled)
        if visualize_processed_btn:
            st.session_state["show_processed_data"] = True

        show_proc = st.session_state.get("show_processed_data", False)

        if show_proc and (processed_path is not None) and os.path.exists(processed_path):
            data_sheets = load_timeseries_data(processed_path)
            if data_sheets:
                sheet_list = list(data_sheets.keys())
                chosen_sheet_processed = st.selectbox(
                    "Select sheet for processed data (static 3D & 2D)", sheet_list, key="processed_sheet_selector"
                )

                col1, col2 = st.columns([1.3, 1], gap="medium")
                with col1:
                    if current_field:
                        qgis_file = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "assets",
                            "data",
                            current_field,
                            f"INFORME_{indice}_QGIS_{anio}.xlsx",
                        )
                    else:
                        qgis_file = os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            "assets",
                            "data",
                            f"INFORME_{indice}_QGIS_{anio}.xlsx",
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
                                    margin_frac=0.05,
                                )

                                if html_2d:
                                    st.markdown('<div class="overflow-wrap">', unsafe_allow_html=True)
                                    components.html(html_2d, height=700, scrolling=True)  # ← no key
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.error("Could not create interactive QGIS chart.")
                        except Exception as e:
                            st.error(f"Error reading QGIS => {e}")

                with col2:
                    # Show corresponding NDVI ColorMap image instead of 3D plot
                    image_path = None
                    if current_field:
                        upload_folder = os.path.join("upload_data", current_field, indice, anio)
                    else:
                        upload_folder = os.path.join("upload_data", f"{indice}_{anio}")
                    if os.path.exists(upload_folder):
                        for file in os.listdir(upload_folder):
                            if "ColorMap" in file and chosen_sheet_processed in file and file.lower().endswith('.tiff'):
                                image_path = os.path.join(upload_folder, file)
                                break
                    
                    if image_path and os.path.exists(image_path):
                        from PIL import Image
                        import base64
                        from io import BytesIO
                        
                        img = Image.open(image_path)
                        # Convert to base64 for HTML display
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        img_b64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        st.markdown(
                            f'<div style="max-width: 80%; margin: 15% auto 0 auto;">'
                            f'<img src="data:image/png;base64,{img_b64}" '
                            f'style="width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />'
                            f'<p style="text-align: center; margin-top: 8px; color: #666;">NDVI ColorMap - {chosen_sheet_processed}</p>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        # Fallback to 3D plot if no image found
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
                            fig_3d.update_layout(margin=dict(l=0, r=0, t=30, b=0), autosize=True)
                            st.plotly_chart(fig_3d, use_container_width=True, use_container_height=True)

                fig_time = create_3d_simulation_plot_time_interpolation(
                    data_sheets, grid_size, color_map, z_scale, smoothness, steps_value
                )
                if fig_time:
                    fig_time.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                    st.markdown("#### Time-Series 3D Animation")
                    st.plotly_chart(fig_time, use_container_width=True)
            else:
                # Single-sheet processed fallback
                df_single = process_uploaded_file(processed_path)
                if df_single is None:
                    st.error("Could not parse processed data. It may not match the expected format.")
                elif all(col in df_single.columns for col in ("Longitud", "Latitud", "NDVI")):
                    lat_vals = df_single["Latitud"].values
                    lon_vals = df_single["Longitud"].values
                    ndvi_vals = df_single["NDVI"].values

                    fig_2d = create_2d_scatter_plot_ndvi(
                        lat_vals, lon_vals, ndvi_vals, sheet_name="ProcessedSingle", google_api_key=google_api_key
                    )
                    fig_3d_static = create_3d_surface_plot(
                        df_single.rename(columns={"Longitud": "Longitud", "Latitud": "Latitud", "NDVI": "NDVI"}),
                        grid_size=grid_size,
                        color_map=color_map,
                        z_scale=z_scale,
                        smoothness=smoothness,
                    )
                    if fig_3d_static:
                        fig_3d_static.update_layout(margin=dict(l=0, r=0, t=30, b=0), autosize=True)

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
                        random_seed=42,
                    )

                    col1, col2 = st.columns([1.3, 1], gap="medium")
                    with col1:
                        if fig_2d:
                            st.pyplot(fig_2d, use_container_width=True)
                    with col2:
                        if fig_3d_static:
                            st.plotly_chart(fig_3d_static, use_container_width=True)

                    if fig_wave:
                        fig_wave.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                        st.markdown("#### Wave-Based 3D Animation (Processed Single-Sheet)")
                        st.plotly_chart(fig_wave, use_container_width=True)
                else:
                    st.error("Processed data does not contain the required columns or sheets.")

    elif page_mode == "Prospective Visualization":
        st.write("## Prospective Visualization")

        # 1) HPC Data Option: run or reuse
        if "hpc_data" not in st.session_state:
            st.warning("No HPC data found in session. Click the button to run HPC pipeline.")
            if st.button("Run HPC Pipeline"):
                try:
                    if field_name:
                        # Ensure field_name uses underscores for consistency
                        safe_field_name = field_name.replace(' ', '_')
                        field_base_folder = os.path.join("./upload_data", safe_field_name)
                    else:
                        field_base_folder = "./upload_data"
                    hpc_data = run_full_hpc_pipeline(indice, anio, base_folder=field_base_folder)
                    if hpc_data is None:
                        st.error("HPC pipeline returned None. Check logs or file paths.")
                    else:
                        st.session_state["hpc_data"] = hpc_data
                        # Show data source information
                        if hpc_data.get("using_mock_data", False):
                            st.warning("⚠️ HPC pipeline completed using **MOCK climate data** (no real climate file was uploaded). Results are for testing purposes only.")
                        else:
                            st.success("✅ HPC pipeline completed using **REAL climate data**. Results are based on actual weather measurements.")
                except Exception as e:
                    st.error(f"Error running HPC pipeline => {e}")
        else:
            hpc_data = st.session_state["hpc_data"]
            # Show current data source status
            if hpc_data.get("using_mock_data", False):
                st.info("📊 HPC Data loaded (using **MOCK climate data**). Below you can visualize the test results.")
            else:
                st.info("📊 HPC Data loaded (using **REAL climate data**). Below you can visualize the results.")

        # 2) IDW/QGIS Visualization
        current_field = st.session_state.get("current_field", field_name)
        if current_field:
            idw_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets",
                "data",
                current_field,
                f"INFORME_{indice}_IDW_{anio}.xlsx",
            )
        else:
            idw_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "assets",
                "data",
                f"INFORME_{indice}_IDW_{anio}.xlsx",
            )

        processed_disabled = True
        processed_path = None
        if "processed_idw_path" in st.session_state and os.path.exists(st.session_state["processed_idw_path"]):
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
                "Select sheet for processed data (static 3D & 2D)", sheet_list, key="processed_sheet_selector"
            )

            # HPC data selection
            HPC_info = None
            if "hpc_data" in st.session_state:
                hpc_data = st.session_state["hpc_data"]
                results = hpc_data.get("results", [])
                point_labels = [f"(Point={r['point_idx']})" for r in results]
                if point_labels:
                    chosen_point = st.selectbox("Select HPC point result", point_labels)
                    chosen_idx = point_labels.index(chosen_point)
                    HPC_info = results[chosen_idx]

            col1, col2 = st.columns([1.3, 1], gap="medium")
            with col1:
                if current_field:
                    qgis_file = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "assets",
                        "data",
                        current_field,
                        f"INFORME_{indice}_QGIS_{anio}.xlsx",
                    )
                else:
                    qgis_file = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "assets",
                        "data",
                        f"INFORME_{indice}_QGIS_{anio}.xlsx",
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
                                margin_frac=0.05,
                            )

                            if html_2d:
                                # wrapper lets it scroll horizontally if needed
                                st.markdown('<div class="overflow-wrap">', unsafe_allow_html=True)
                                components.html(html_2d, height=700, scrolling=True)  # height must match function's height_px
                                st.markdown("</div>", unsafe_allow_html=True)
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

                if HPC_info is not None:
                    hpc_data = st.session_state["hpc_data"]
                    XLDA = HPC_info["XLDA"]  # shape [1000 x 12]
                    # VC = HPC_info["VC"]    # unused in the density plot below
                    # XInf2 = HPC_info["XInf"]

                    # Add data source indicator to section header
                    data_source = "Mock Data" if hpc_data.get("using_mock_data", False) else "Real Data"
                    st.markdown(f"## Monthly Risk Evolution ({data_source})")

                    fig = go.Figure()
                    n_months = XLDA.shape[1]

                    # Plot the filled-area KDE for HPC data
                    for m in range(1, n_months):
                        month_data = XLDA[:, m]
                        kde = gaussian_kde(month_data)
                        x_range = np.linspace(month_data.min(), month_data.max(), 200)
                        density = kde(x_range)
                        fig.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=density,
                                mode="lines",
                                fill="tozeroy",
                                name=f"Month {m+1}",
                                hovertemplate="Month: %{text}<br>Loss: %{x:.2f}<br>Density: %{y:.2f}",
                                text=[f"Month {m+1}"] * len(x_range),
                            )
                        )

                    fig.update_layout(
                        xaxis_title="Losses (USD/Month-Zone)",
                        yaxis_title="Density",
                        showlegend=True,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig, use_container_width=True, use_container_height=True)
                else:
                    st.info("No HPC data loaded/selected. Run the pipeline or pick a point.")

            st.markdown("---")

            # 3) HPC Risk Distributions & Table from HPC Data
            if "hpc_data" in st.session_state and HPC_info is not None:
                XLDA = HPC_info["XLDA"]
                VC = HPC_info["VC"]
                XInf2 = HPC_info["XInf"]

                columns = [
                    "WD",
                    "Max C",
                    "Min  C",
                    "Viento (m/s)",
                    "Humedad (%)",
                    "Precip. (mm)",
                    f"{indice}",
                    "Skewness",
                    "%C1",
                    "%C2",
                    "%C3",
                    "Mean (units)",
                    "75% (units)",
                    "OpVar-99% (units)",
                ]

                n_months = XLDA.shape[1] if XLDA is not None else 12
                table_data = []
                for row_i in range(n_months):
                    wd = VC[row_i] if row_i < len(VC) else ""
                    maxC = XInf2[row_i, 0]
                    minC = XInf2[row_i, 1]
                    viento = XInf2[row_i, 2]
                    hum = XInf2[row_i, 3]
                    prec = XInf2[row_i, 4]
                    ndvi_val = XInf2[row_i, 8]  # or 9 depending on your schema
                    skewv = XInf2[row_i, 5]
                    pc1 = XInf2[row_i, 6]
                    pc2 = XInf2[row_i, 7]
                    pc3 = XInf2[row_i, 8]
                    mean_val = XInf2[row_i, 9]
                    p75_val = XInf2[row_i, 10]
                    opv99 = XInf2[row_i, 11]

                    table_data.append(
                        [wd, maxC, minC, viento, hum, prec, ndvi_val, skewv, pc1, pc2, pc3, mean_val, p75_val, opv99]
                    )

                df_hpc = pd.DataFrame(table_data, columns=columns)
                for col_ in df_hpc.columns.drop("WD"):
                    df_hpc[col_] = df_hpc[col_].astype(float).map("{:.3f}".format)

                # Add data source indicator to table header
                data_source = "Mock Data" if hpc_data.get("using_mock_data", False) else "Real Data"
                st.markdown(f"### HPC Risk Data Table ({data_source})")
                st.dataframe(df_hpc)
            else:
                st.info("No HPC data loaded. Please click 'Run HPC Pipeline' to compute results.")
        else:
            st.info("No IDW data to visualize or file not found. Upload or process .zip for NDVI data if needed.")

    elif page_mode == "Risk Management":
        st.write("## GHG Capture & Risk Management")
        
        # Information about the corrected implementation
        st.info("""
        🔧 **Enhanced Implementation Features:**
        - ✅ Corrected KMeans clustering for risk level determination
        - ✅ Individual risk modifications with MIo/MIo_G ratios
        - ✅ Reference lines in Risk Profile LDA (Media_O, OpVar_O, Media_G, OpVar_G)
        - ✅ Enhanced financial metrics with CO2 capture calculations
        - ✅ Statistical analysis with skewness measurements
        - ✅ Proper LDAT column stacking for visualization
        """)
        
        # Process GHG data
        if st.button("Process GHG Data"):
            with st.spinner("Processing GHG capture data..."):
                # Ensure field_name uses underscores for consistency
                safe_field_name = field_name.replace(' ', '_') if field_name else None
                ghg_data = process_ghg_data(indice, anio, base_folder="./upload_data", field_name=safe_field_name)
                if ghg_data:
                    st.session_state["ghg_data"] = ghg_data
                    st.success("GHG data processed successfully!")
                else:
                    st.error("Failed to process GHG data. Check if required files exist.")
        
        # Display results if data exists
        if "ghg_data" in st.session_state:
            ghg_data = st.session_state["ghg_data"]
            
            # Display cluster information
            st.subheader("Cluster Analysis")
            st.info(f"Exchange Rate: 1 USD = {ghg_data['usd_cop_rate']:,.0f} COP")
            st.dataframe(ghg_data["clusters"])
            
            # Risk matrices visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Events Matrix")
                fig_events = create_risk_matrix_heatmap(
                    ghg_data["events_matrix"], 
                    "Events Matrix",
                    ghg_data["frequency_labels"],
                    ghg_data["severity_labels"]
                )
                st.plotly_chart(fig_events, use_container_width=True)
            
            with col2:
                st.subheader("Losses Matrix")
                fig_losses = create_risk_matrix_heatmap(
                    ghg_data["losses_matrix"], 
                    "Losses Matrix (USD)",
                    ghg_data["frequency_labels"],
                    ghg_data["severity_labels"]
                )
                st.plotly_chart(fig_losses, use_container_width=True)
            
            # Impact matrix
            st.subheader("Impact Matrix")
            fig_impact = create_risk_matrix_heatmap(
                ghg_data["impact_matrix"], 
                "Impact Matrix",
                ghg_data["frequency_labels"],
                ghg_data["severity_labels"]
            )
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Management results table
            st.subheader("Risk Management Scenarios")
            st.dataframe(ghg_data["results"])
            
            # Management matrices visualization
            st.subheader("Management Matrices")
            if "management_matrices" in ghg_data:
                matrix_cols = st.columns(2)
                matrix_names = list(ghg_data["management_matrices"].keys())
                
                for i, (name, matrix) in enumerate(ghg_data["management_matrices"].items()):
                    col_idx = i % 2
                    with matrix_cols[col_idx]:
                        fig_matrix = create_management_matrix_heatmap(
                            matrix,
                            f"Management Matrix - {name}",
                            ghg_data["frequency_labels"],
                            ghg_data["severity_labels"]
                        )
                        st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Cost-benefit analysis
            st.subheader("Cost-Benefit Analysis")
            fig_cost_benefit = create_cost_benefit_chart(ghg_data["results"])
            st.plotly_chart(fig_cost_benefit, use_container_width=True)
            
            # LDA distribution plot
            st.subheader("Loss Distribution Analysis (LDA)")
            fig_lda = create_lda_distribution_plot(
                ghg_data["lda_data"],
                ghg_data["mgr_labels"],
                ghg_data.get("visualization_lines")  # Add the new reference lines parameter
            )
            st.plotly_chart(fig_lda, use_container_width=True)
        
            # Additional metrics dashboard
            st.subheader("Risk Metrics Summary")
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                baseline_loss = ghg_data["results"].loc["Baseline", "Media (USD)"]
                st.metric("Baseline Loss", f"${baseline_loss:,.0f}")
            
            with metrics_cols[1]:
                best_scenario = ghg_data["results"]["Media (USD)"].idxmin()
                best_loss = ghg_data["results"].loc[best_scenario, "Media (USD)"]
                reduction = ((baseline_loss - best_loss) / baseline_loss) * 100
                st.metric("Best Scenario", best_scenario, f"-{reduction:.1f}%")
            
            with metrics_cols[2]:
                max_vc = ghg_data["results"]["VCap. (USD)"].max()  # Updated column name
                st.metric("Max Value Captured", f"${max_vc:,.0f}")
            
            with metrics_cols[3]:
                total_events = ghg_data["results"]["NE"].iloc[0]
                st.metric("Total Risk Events", f"{total_events:,}")

            # Enhanced metrics from corrected implementation
            st.subheader("Enhanced Risk Analytics (Corrected Implementation)")
            enhanced_cols = st.columns(4)
            
            with enhanced_cols[0]:
                # CO2 Capture metrics
                max_co2 = ghg_data["results"]["TCO2(Ton.)"].max()
                st.metric("Max CO2 Capture", f"{max_co2:.3f} tons")
            
            with enhanced_cols[1]:
                # Skewness analysis
                baseline_skew = ghg_data["results"].loc["Baseline", "C.As."]
                st.metric("Baseline Skewness", f"{baseline_skew:.4f}")
            
            with enhanced_cols[2]:
                # Operational Income
                max_op_income = ghg_data["results"]["IngOp.(USD)"].max()
                st.metric("Max Op. Income", f"${max_op_income:,.0f}")
            
            with enhanced_cols[3]:
                # Reference lines info
                if "visualization_lines" in ghg_data:
                    media_improvement = (ghg_data["visualization_lines"]["media_val_o"] - 
                                       ghg_data["visualization_lines"]["media_val_g"])
                    st.metric("Risk Reduction", f"${media_improvement:,.0f}")
        
        else:
            st.info("Click 'Process GHG Data' to analyze greenhouse gas capture and risk management scenarios.")


# ---------------------- MAIN EXECUTION ENTRY ----------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_ui()
