
import pandas as pd
import numpy as np
import streamlit as st
import logging

logger = logging.getLogger(__name__)

def process_uploaded_file(uploaded_file):
    try:
        logger.info("Processing uploaded file")
        
        # Read the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Verify that the required columns exist
        required_columns = ["Longitud", "Latitud", "NDVI", "Riesgo"]
        if not all(col in df.columns for col in required_columns):
            error_message = f"The uploaded file must contain the following columns: {', '.join(required_columns)}"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Extract the required columns
        data = df[required_columns].copy()
        
        # Handle missing values
        if data.isnull().values.any():
            st.warning("Missing values detected. Rows with missing values will be dropped.")
            logger.warning("Missing values detected in the data")
            data.dropna(subset=required_columns, inplace=True)
        
        # Validate NDVI values are between -1 and 1
        if not ((data["NDVI"] >= -1.5) & (data["NDVI"] <= 1.5)).all():
            error_message = "NDVI values must be between -1.5 and 1.5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Validate Riesgo values are between 1 and 5
        if not ((data["Riesgo"] >= 1) & (data["Riesgo"] <= 5)).all():
            error_message = "Riesgo values must be between 1 and 5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Convert Riesgo to integer (if it's not already)
        data["Riesgo"] = data["Riesgo"].astype(int)
        
        logger.info("File processed successfully")
        # Return the processed DataFrame
        return data

    except Exception as e:
        error_message = f"An error occurred while processing the file: {e}"
        st.error(error_message)
        logger.exception("An error occurred while processing the file")
        return None