
import pandas as pd
from app.visualization import (
    create_2d_scatter_plot_ndvi_interactive_qgis,
    create_3d_surface_plot,
)

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


def test_qgis_2d_map_without_google_key_uses_esri_leaflet():
    df = pd.DataFrame(
        {
            "long-xm": [-74.0010, -74.0005, -74.0000, -73.9995],
            "long-ym": [4.0010, 4.0015, 4.0020, 4.0025],
            "NDVI": [0.20, 0.35, 0.55, 0.75],
            "Riesgo": ["Alto", "Medio", "Bajo", "Bajo"],
        }
    )

    html = create_2d_scatter_plot_ndvi_interactive_qgis(df, sheet_name="Test", google_api_key="")

    assert html is not None
    assert "server.arcgisonline.com/ArcGIS/rest/services/World_Imagery" in html
    assert "Esri Satellite" in html
    assert "Punto 0" in html
