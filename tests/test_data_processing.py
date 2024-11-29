
import pytest
import pandas as pd
from app.data_processing import process_uploaded_file
from io import BytesIO

def test_process_uploaded_file_valid_data():
    # Create a sample DataFrame
    data = {
        "Longitud": [100, 101, 102],
        "Latitud": [0, 1, 2],
        "NDVI": [0.5, -0.3, 0.8],
        "Riesgo": [1, 3, 5]
    }
    df = pd.DataFrame(data)
    # Save to a BytesIO object as an Excel file
    excel_file = BytesIO()
    df.to_excel(excel_file, index=False)
    excel_file.seek(0)

    # Call the function
    processed_data = process_uploaded_file(excel_file)

    # Assertions
    assert processed_data is not None
    assert not processed_data.empty
    assert list(processed_data.columns) == ["Longitud", "Latitud", "NDVI", "Riesgo"]

def test_process_uploaded_file_missing_columns():
    # Create a DataFrame missing the 'NDVI' column
    data = {
        "Longitud": [100, 101, 102],
        "Latitud": [0, 1, 2],
        "Riesgo": [1, 3, 5]
    }
    df = pd.DataFrame(data)
    excel_file = BytesIO()
    df.to_excel(excel_file, index=False)
    excel_file.seek(0)

    # Call the function
    processed_data = process_uploaded_file(excel_file)

    # Assertions
    assert processed_data is None

def test_process_uploaded_file_invalid_ndvi():
    # Create a DataFrame with invalid NDVI values
    data = {
        "Longitud": [100, 101, 102],
        "Latitud": [0, 1, 2],
        "NDVI": [1.5, -1.2, 0.8],
        "Riesgo": [1, 3, 5]
    }
    df = pd.DataFrame(data)
    excel_file = BytesIO()
    df.to_excel(excel_file, index=False)
    excel_file.seek(0)

    # Call the function
    processed_data = process_uploaded_file(excel_file)

    # Assertions
    assert processed_data is None