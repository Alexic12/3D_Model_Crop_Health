import pandas as pd
import numpy as np
import streamlit as st
import logging
from scipy.interpolate import griddata
import rasterio
from pyproj import Transformer
import logging

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



def rejilla_indice(tiff_binary):
    """
    Reads a single GeoTIFF from an in-memory buffer or file path,
    extracts NDVI values + coordinates, returns a DataFrame.
    """
    try:
        # rasterio can open an in-memory file if it’s a BytesIO, or you can open by path
        # e.g., rasterio.open(tiff_binary) if tiff_binary is an uploaded file-like object
        with rasterio.open(tiff_binary) as src:
            # Read the first band (assumed NDVI)
            band1 = src.read(1)  # shape = (height, width)
            crs = src.crs
            no_data_value = src.nodata if src.nodata is not None else -9999

            # Convert no-data to np.nan
            band1 = np.where(band1 == no_data_value, np.nan, band1)

            width = src.width
            height = src.height

            # We will accumulate all pixel data
            data_rows = []

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            # Loop over each pixel
            for row in range(height):
                for col in range(width):
                    val = band1[row, col]
                    if not np.isnan(val):
                        # Convert pixel row,col => projected coords => lat/lon
                        x, y = src.xy(row, col)
                        # Note: if your CRS is something else, confirm the order is (x, y)
                        # Transform to lat/lon
                        lon, lat = transformer.transform(x, y)

                        data_rows.append([x, y, lon, lat, col, row, val])

            df = pd.DataFrame(data_rows,
                              columns=['UTM-x', 'UTM-y', 'longitud', 'latitud',
                                       'col', 'row', 'NDVI'])
            return df

    except Exception as e:
        logger.exception(f"Error in rejilla_indice: {e}")
        return None


def IDW_Index(df, resolution=5):
    """
    Performs IDW interpolation on a DataFrame containing
    columns [longitud, latitud, NDVI].
    Returns (zidw_grid, idw_dataframe).
    """
    try:
        xm = df['longitud'].values
        ym = df['latitud'].values
        zms = df['NDVI'].values

        xm_min = xm.min()
        xm_max = xm.max()
        ym_min = ym.min()
        ym_max = ym.max()

        # Create a regular grid of resolution 'resolution'
        xg = np.linspace(xm_min, xm_max, resolution)
        yg = np.linspace(ym_min, ym_max, resolution)

        # Output grid
        zidw = np.zeros((resolution, resolution))
        weight_sum = np.zeros((resolution, resolution))

        nKNN = 10  # number of neighbors

        # For each point in the output grid, do a basic IDW from the nearest points
        for i in range(resolution):
            for j in range(resolution):
                gx = xg[i]
                gy = yg[j]

                # Distance from each known NDVI sample
                dist = np.sqrt((xm - gx)**2 + (ym - gy)**2)
                # Sort distances
                sorted_idx = np.argsort(dist)
                # Keep the nearest nKNN
                knn_idx = sorted_idx[:nKNN]
                knn_dist = dist[knn_idx]
                knn_vals = zms[knn_idx]

                # Avoid zero distance
                knn_dist = np.where(knn_dist == 0, 1e-6, knn_dist)

                # IDW weights ~ 1/dist^3, for example
                w = (1.0 / (knn_dist**3))
                wsum = w.sum()
                zidw[j, i] = np.sum(w * knn_vals) / wsum if wsum != 0 else 0.0

        # Prepare a DataFrame version (like your code) with a row for each grid cell
        id_list = []
        idx = 0
        for j in range(resolution):
            for i in range(resolution):
                idx += 1
                id_list.append([idx, xg[i], yg[j], zidw[j, i]])

        df_idw = pd.DataFrame(id_list, columns=['id', 'long-xm', 'long-ym', 'NDVI'])

        # Optionally also return the 2D grid as a DataFrame
        # We'll store the first row as x coords, first col as y coords, etc.
        # Or just keep it as a 2D np.array.

        return zidw, df_idw

    except Exception as e:
        logger.exception(f"Error in IDW_Index: {e}")
        return None, None


def Riesgo(df_idw, clusters):
    """
    Example risk classification:
    Takes the df_idw (columns: id, long-xm, long-ym, NDVI),
    modifies it to have a 'Riesgo' column, updating 'clusters' if needed.
    This is a placeholder for your actual logic.
    """
    try:
        # Suppose you want to match each NDVI with the nearest cluster center
        # and then store the cluster index as 'Riesgo'.
        df_idw['Riesgo'] = 1
        # For demonstration, we do a simple approach:
        # NDVI near clusters => ???

        # This is just a placeholder. You can adapt your own logic.
        # For each row, find cluster with min (NDVI - c)^2
        # Then assign that cluster index + 1
        for i in range(len(df_idw)):
            ndvi_val = df_idw.loc[i, 'NDVI']
            # find nearest cluster
            dist_sq = (clusters - ndvi_val)**2
            nearest_idx = np.argmin(dist_sq)
            df_idw.loc[i, 'Riesgo'] = nearest_idx + 1

        # Possibly update clusters if you do iterative clustering
        # We'll skip that for brevity. Return updated df and the same clusters
        return df_idw, clusters
    except Exception as e:
        logger.exception(f"Error in Riesgo: {e}")
        return df_idw, clusters