
import pytest
import pandas as pd
from app.data_processing import process_uploaded_file
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.data.data_processing import (
    bulk_unzip_and_analyze_new_parallel,
    rejilla_indice,
    _resolve_bulk_worker_count,
)

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


def _write_tiff(path: Path, data: np.ndarray) -> None:
    transform = from_origin(500000, 500000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:32618",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)


def test_rejilla_indice_vectorized_reads_valid_pixels(tmp_path):
    base = tmp_path / "Campo_Test_NDVI_01ene2024.tiff"
    color = tmp_path / "Campo_Test_NDVI_ColorMap_01ene2024.tiff"
    data = np.array([[0.1, 0.2], [0.3, -9999]], dtype="float32")
    _write_tiff(base, data)
    _write_tiff(color, data)

    df = rejilla_indice(str(base), str(color))

    assert df is not None
    assert len(df) == 3
    assert list(df.columns) == ["UTM-x", "UTM-y", "longitud", "latitud", "col", "row", "NDVI"]
    assert df["NDVI"].between(-1, 1).all()


def test_bulk_analysis_creates_expected_workbooks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHV_BULK_WORKERS", "2")
    progress_events = []

    folder = tmp_path / "upload_data" / "Campo_Test" / "NDVI" / "2024"
    folder.mkdir(parents=True)

    for prefix, date_token in (("001", "01ene2024"), ("002", "02ene2024")):
        base_tiff = tmp_path / f"Campo_Test_NDVI_{date_token}.tiff"
        color_tiff = tmp_path / f"Campo_Test_NDVI_ColorMap_{date_token}.tiff"
        data = np.array([[0.2, 0.4], [0.6, 0.8]], dtype="float32")
        _write_tiff(base_tiff, data)
        _write_tiff(color_tiff, data)

        zip_path = folder / f"{prefix}. Campo_Test_NDVI_{date_token}.zip"
        with ZipFile(zip_path, "w") as zf:
            zf.write(base_tiff, arcname=base_tiff.name)
            zf.write(color_tiff, arcname=color_tiff.name)

    outputs = bulk_unzip_and_analyze_new_parallel(
        "NDVI",
        "2024",
        base_folder=str(tmp_path / "upload_data" / "Campo_Test"),
        progress_callback=progress_events.append,
    )

    assert all(outputs)
    for output in outputs:
        path = tmp_path / output
        assert path.exists()
        sheets = pd.read_excel(path, sheet_name=None)
        assert "01ene2024" in sheets
        assert "02ene2024" in sheets
    assert progress_events
    assert progress_events[-1]["stage"] == "done"
    assert progress_events[-1]["completed"] == progress_events[-1]["total"] == 2
    assert any(event["stage"] == "process" and event["completed"] > 0 for event in progress_events)


def test_bulk_worker_count_uses_more_than_legacy_two_when_safe(monkeypatch):
    monkeypatch.delenv("CHV_BULK_WORKERS", raising=False)
    monkeypatch.delenv("CHV_BULK_MAX_WORKERS", raising=False)
    monkeypatch.setattr("app.data.data_processing.multiprocessing.cpu_count", lambda: 8)
    monkeypatch.setattr("app.data.data_processing._available_memory_gb", lambda: 32)

    assert _resolve_bulk_worker_count(20) > 2
