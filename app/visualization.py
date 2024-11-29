
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import logging

logger = logging.getLogger(__name__)

def griddata_points_to_grid(x, y, values, xi, yi):
    # Flatten the data
    points = np.column_stack((x, y))
    values_flat = values

    # Grid the data
    grid_z = griddata(points, values_flat, (xi, yi), method='linear')

    # Replace NaN values with zeros or use nearest interpolation
    grid_z = np.nan_to_num(grid_z, nan=0.0)

    return grid_z
def create_3d_surface_plot(data, grid_size=100, color_map="Viridis"):
    try:
        logger.info("Creating 3D surface plot")
        # Extract data for plotting
        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values
        riesgo = data['Riesgo'].values

        # Create a grid for the surface plot using grid_size
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate NDVI values onto the grid
        zi = griddata_points_to_grid(x, y, z, xi, yi)

        # Interpolate Riesgo values onto the grid
        riesgo_i = griddata_points_to_grid(x, y, riesgo, xi, yi)

        # Define a custom colorscale for Riesgo
        colorscale = [
            [0.0, 'green'],    # 1 (Low risk)
            [0.25, 'yellow'],  # 2
            [0.5, 'orange'],   # 3
            [0.75, 'red'],     # 4
            [1.0, 'darkred']   # 5 (High risk)
        ]

        # Create the surface plot
        fig = go.Figure(data=[go.Surface(
            x=xi,
            y=yi,
            z=zi,
            surfacecolor=riesgo_i,
            colorbar=dict(title='Riesgo'),
            colorscale=color_map,
            cmin=1,
            cmax=5,
        )])

        # Update layout
        fig.update_layout(
            title='Crop Health 3D Surface',
            width=1000,   # Increase the width
            height=800,   # Increase the height
            #autosize=False,  # Disable autosize when specifying width and height
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                zaxis=dict(range=[-0.5, 1.5]),
            ),
            autosize=True,
        )

        # Add hover information
        fig.update_traces(
            hovertemplate=
            'Longitude: %{x:.2f}<br>' +
            'Latitude: %{y:.2f}<br>' +
            'NDVI: %{z:.2f}<br>' +
            'Riesgo: %{surfacecolor:.0f}'
        )

        logger.info("3D surface plot created successfully")
        return fig

    except Exception as e:
        error_message = f"An error occurred while creating the visualization: {e}"
        st.error(error_message)
        logger.exception("An error occurred in create_3d_surface_plot")
        return None
