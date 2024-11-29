
import pytest
import pandas as pd
from app.visualization import create_3d_surface_plot

def test_create_3d_surface_plot_valid_data():
    # Create a sample DataFrame
    data = {
        "Longitud": [100, 101, 102],
        "Latitud": [0, 1, 2],
        "NDVI": [0.5, -0.3, 0.8],
        "Riesgo": [1, 3, 5]
    }
    df = pd.DataFrame(data)

    # Call the function
    fig = create_3d_surface_plot(df)

    # Assertions
    assert fig is not None
    assert fig.data  # Ensure that data exists in the figure