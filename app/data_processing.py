import pandas as pd
import numpy as np
import streamlit as st
import logging
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def process_uploaded_file(uploaded_file):
    """
    Reads a single-sheet Excel file with columns [Longitud, Latitud, NDVI, Riesgo].
    Ensures NDVI is in [-1.5, 1.5], Riesgo in [1..5].
    Returns a DataFrame if valid, otherwise None.
    """
    try:
        logger.info("Processing uploaded file")
        
        # Read the Excel file into a DataFrame (first sheet by default)
        df = pd.read_excel(uploaded_file)
        
        # Verify required columns
        required_columns = ["Longitud", "Latitud", "NDVI", "Riesgo"]
        if not all(col in df.columns for col in required_columns):
            error_message = (
                "The uploaded file must contain the following columns: "
                f"{', '.join(required_columns)}"
            )
            st.error(error_message)
            logger.error(error_message)
            return None

        # Extract the required columns
        data = df[required_columns].copy()
        
        # Handle missing values
        if data.isnull().values.any():
            st.warning("Missing values detected. Rows with missing values will be dropped.")
            logger.warning("Missing values in data")
            data.dropna(subset=required_columns, inplace=True)
        
        # Validate NDVI range
        if not ((data["NDVI"] >= -1.5) & (data["NDVI"] <= 1.5)).all():
            error_message = "NDVI values must be between -1.5 and 1.5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Validate Riesgo range
        if not ((data["Riesgo"] >= 1) & (data["Riesgo"] <= 5)).all():
            error_message = "Riesgo values must be between 1 and 5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Convert Riesgo to integer
        data["Riesgo"] = data["Riesgo"].astype(int)
        
        logger.info("File processed successfully")
        return data

    except Exception as e:
        error_message = f"An error occurred while processing the file: {e}"
        st.error(error_message)
        logger.exception("Error processing file")
        return None


def load_timeseries_data(uploaded_file):
    """
    Reads an Excel file with multiple sheets, each containing a matrix:
      - First row is discarded.
      - Second row (index=1) has longitude in columns [1..end].
      - First column in rows [2..end] has latitude.
      - Intersection is NDVI data in 2D form.

    Returns a dict { sheet_name: {"lon":..., "lat":..., "ndvi":...} } or None on error.
    """
    try:
        logger.info("Loading time-series data from multiple sheets...")

        excel_obj = pd.ExcelFile(uploaded_file)
        sheet_names = excel_obj.sheet_names
        data_sheets = {}

        for sname in sheet_names:
            logger.info(f"Reading sheet: {sname}")
            df = pd.read_excel(uploaded_file, sheet_name=sname, header=None)
            
            # Basic check
            if df.shape[0] < 3 or df.shape[1] < 2:
                logger.warning(f"Sheet {sname} is too small. Skipping.")
                continue
            
            # The 1D array of longitude is from row=1, columns=1..end
            lon = df.iloc[1, 1:].to_numpy(dtype=float)
            # The 1D array of latitude is from rows=2..end, col=0
            lat = df.iloc[2:, 0].to_numpy(dtype=float)
            # The 2D NDVI matrix is from rows=2..end, columns=1..end
            ndvi_matrix = df.iloc[2:, 1:].to_numpy(dtype=float)
            
            data_sheets[sname] = {
                "lon": lon,
                "lat": lat,
                "ndvi": ndvi_matrix
            }

        if not data_sheets:
            st.error("No valid sheets found or parsing failed in the uploaded file.")
            return None

        logger.info("Successfully loaded all sheets for time-series data.")
        return data_sheets

    except Exception as e:
        error_message = f"An error occurred while loading time-series data: {e}"
        st.error(error_message)
        logger.exception("Error in load_timeseries_data")
        return None


def griddata_points_to_grid(x, y, values, xi, yi):
    """
    Interpolates scattered (x, y, values) onto a regular grid (xi, yi).
    Replaces NaN with 0.0 after interpolation.
    """
    points = np.column_stack((x, y))
    grid_z = griddata(points, values, (xi, yi), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    return grid_z
