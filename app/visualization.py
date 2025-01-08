
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter  # Import for smoothing
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
def create_3d_surface_plot(data, grid_size=100, color_map="Viridis", z_scale=1.0, smoothness=0.0):
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

        # Apply smoothing to zi if smoothness > 0
        if smoothness > 0:
            zi = gaussian_filter(zi, sigma=smoothness, mode='nearest')

        # Apply Z-axis scaling
        zi *= z_scale

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
            height=1000,   # Increase the height
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


def create_3d_simulation_plot_sea_keypoints(
    data,
    grid_size=50,
    color_map="Viridis",
    z_scale=1.0,
    smoothness=0.0,
    key_frames=100,                # Number of animation frames
    key_points=20,                # How many anchor points to sample
    wave_amplitude_range=(-0.05, 0.7),
    wave_frequency_range=(0.05, 0.4),
    wave_speed_range=(0.5, 2.0),
    influence_sigma=0.1,         # Larger => smoother, more blended waves
    random_seed=42,
    z_min=-0.3,
    z_max=0.9
):
    """
    Creates a 3D animated 'sea-like' surface by placing a limited number
    of random 'anchor points' (key_points). Each anchor has wave params,
    and cells are influenced by anchors based on distance.

    **Now the color is determined by the *current* Z value at each point.**
    cmin/cmax can be set via z_min, z_max to fix the color scale range.
    """
    try:
        logger.info("Creating 3D 'sea' simulation (anchor-based) with NDVI-based coloring")

        # ------------------------------------------------------
        # 1) Interpolate the base NDVI surface
        # ------------------------------------------------------
        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values
        # We do *not* need Riesgo for coloring, unless you want to use it separately
        # but let's still load it in case we need it
        riesgo = data['Riesgo'].values

        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        zi_original = griddata_points_to_grid(x, y, z, xi, yi)

        # Optional smoothing
        if smoothness > 0:
            zi_original = gaussian_filter(zi_original, sigma=smoothness, mode='nearest')

        # Vertical scale
        zi_original *= z_scale

        # If you still want to show Riesgo somewhere, you can. But for coloring,
        # we’ll rely on the changing Z values. 
        # Example: we won't use riesgo_i for surfacecolor now
        # riesgo_i = griddata_points_to_grid(x, y, riesgo, xi, yi)

        # ------------------------------------------------------
        # 2) Sample 'key_points' random anchors & wave parameters
        # ------------------------------------------------------
        np.random.seed(random_seed)

        # Anchor coordinates in the domain
        anchor_x = np.random.uniform(x.min(), x.max(), key_points)
        anchor_y = np.random.uniform(y.min(), y.max(), key_points)

        wave_amps   = np.random.uniform(*wave_amplitude_range, size=key_points)
        wave_freq_x = np.random.uniform(*wave_frequency_range, size=key_points)
        wave_freq_y = np.random.uniform(*wave_frequency_range, size=key_points)
        wave_phase  = np.random.uniform(0, 2*np.pi, key_points)
        wave_speed  = np.random.uniform(*wave_speed_range, size=key_points)

        def anchor_wave(i, xvals, yvals, t):
            angle = wave_phase[i] + 2*np.pi * wave_speed[i] * t
            dx = xvals - anchor_x[i]
            dy = yvals - anchor_y[i]
            return wave_amps[i] * np.sin(wave_freq_x[i]*dx + wave_freq_y[i]*dy + angle)

        def anchor_weight(i, xvals, yvals):
            dx = xvals - anchor_x[i]
            dy = yvals - anchor_y[i]
            dist_sq = dx*dx + dy*dy

            # We'll define sigma in terms of bounding box size
            sigma_sq = (influence_sigma**2) * ((x.max()-x.min())**2 + (y.max()-y.min())**2)
            return np.exp(-dist_sq / (2*sigma_sq))

        # Precompute anchor weights
        anchor_weights = []
        for i in range(key_points):
            wmap = anchor_weight(i, xi, yi)
            anchor_weights.append(wmap)

        # ------------------------------------------------------
        # 3) Build frames
        # ------------------------------------------------------
        frames = []
        for frame_i in range(key_frames):
            t = frame_i / (key_frames - 1) if key_frames > 1 else 0.0

            numerator   = np.zeros_like(zi_original)
            denominator = np.zeros_like(zi_original)

            for i in range(key_points):
                wmap      = anchor_weights[i]
                wave_map  = anchor_wave(i, xi, yi, t)
                numerator   += wmap * wave_map
                denominator += wmap

            denominator = np.where(denominator==0, 1e-9, denominator)
            total_wave = numerator / denominator

            # Combine with base NDVI
            zi_frame = zi_original + total_wave

            # Key difference: color by current zi_frame
            frame_data = go.Surface(
                x=xi,
                y=yi,
                z=zi_frame,             # 3D geometry
                surfacecolor=zi_frame,  # color by NDVI
                colorscale=color_map,
                colorbar=dict(title='NDVI'),
                cmin=z_min,             # or compute a dynamic range
                cmax=z_max
            )
            frames.append(
                go.Frame(data=[frame_data], 
                name=f"frame{frame_i}", 
                layout=dict(scene=dict(camera=None)))
            )

        # ------------------------------------------------------
        # 4) Construct the figure w/ initial surface
        # ------------------------------------------------------
        # For the initial frame, use zi_original for both geometry and color
        fig = go.Figure(
            data=[go.Surface(
                x=xi,
                y=yi,
                z=zi_original,
                surfacecolor=zi_original,
                colorscale=color_map,
                colorbar=dict(title='NDVI'),
                cmin=z_min,
                cmax=z_max
            )],
            frames=frames
        )

        # ------------------------------------------------------
        # 5) Add layout & animation controls
        # ------------------------------------------------------
        fig.update_layout(
            title='Crop Health 3D Sea Simulation (Anchor-based) - Color by Z',
            width=900,
            height=900,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                # If you expect NDVI to exceed [-0.5..1.5], adjust accordingly
                zaxis=dict(range=[z_min, z_max]),
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    #showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, {
                                "frame": {"duration": 33, "redraw": True},
                                "transition": {"duration": 0, "easing": "linear"},
                                "fromcurrent": True,
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate"
                            }])
                    ],
                    x=0.05,
                    xanchor="left",
                    y=1,
                    yanchor="top"
                )
            ]
        )

        logger.info("3D 'sea' simulation (anchor-based) with dynamic NDVI coloring created successfully")
        return fig

    except Exception as e:
        error_message = f"An error occurred in sea simulation with key points (color by Z): {e}"
        logger.exception("Sea simulation error")
        st.error(error_message)
        return None