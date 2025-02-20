import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def create_2d_scatter_plot_ndvi(lat, lon, ndvi_matrix, sheet_name="NDVI Sheet"):
    try:
        plt.style.use('dark_background')
        X, Y = np.meshgrid(lon, lat)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")

        scatter = ax.scatter(
            X.flatten(),
            Y.flatten(),
            c=ndvi_matrix.flatten(),
            cmap='autumn',
            vmin=-1,
            vmax=1
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"NDVI Visualization - {sheet_name}")
        fig.colorbar(scatter, ax=ax, label='NDVI Index')
        fig.tight_layout()

        return fig
    except Exception as e:
        logger.exception("Error creating 2D scatter plot")
        st.error(f"Error creating 2D scatter plot: {e}")
        return None


def create_3d_surface_plot(
    data,
    grid_size=100,
    color_map="Viridis",
    z_scale=1.0,
    smoothness=0.0
):
    try:
        logger.info("Creating 3D surface plot (colored by NDVI)")

        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values

        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate NDVI
        points = np.column_stack((x, y))
        zi = griddata(points, z, (xi, yi), method='linear')
        zi = np.nan_to_num(zi, nan=0.0)

        if smoothness > 0:
            zi = gaussian_filter(zi, sigma=smoothness, mode='nearest')

        zi *= z_scale
        cmin, cmax = zi.min(), zi.max()

        fig = go.Figure(data=[go.Surface(
            x=xi,
            y=yi,
            z=zi,
            surfacecolor=zi,
            colorscale=color_map,
            colorbar=dict(title='NDVI'),
            cmin=cmin,
            cmax=cmax
        )])
        fig.update_layout(
            title='Crop Health 3D Surface (NDVI)',
            width=1000,
            height=1000,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI'
            ),
            autosize=True
        )
        fig.update_traces(
            hovertemplate=(
                'Longitude: %{x:.2f}<br>'
                'Latitude: %{y:.2f}<br>'
                'NDVI: %{z:.3f}'
            )
        )
        return fig

    except Exception as e:
        logger.exception("Error in create_3d_surface_plot")
        st.error(f"Error in create_3d_surface_plot: {e}")
        return None


def create_3d_simulation_plot_sea_keypoints(
    data,
    grid_size=50,
    color_map="Viridis",
    z_scale=1.0,
    smoothness=0.0,
    key_frames=100,
    key_points=20,
    wave_amplitude_range=(-0.05, 0.7),
    wave_frequency_range=(0.05, 0.4),
    wave_speed_range=(0.5, 2.0),
    influence_sigma=0.1,
    random_seed=42,
    z_min=-0.3,
    z_max=0.9
):
    """
    Animated "sea-like" wave simulation on top of NDVI base.
    """
    try:
        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values

        # Make a grid
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        points = np.column_stack((x, y))
        base_z = griddata(points, z, (xi, yi), method='linear')
        base_z = np.nan_to_num(base_z, nan=0.0)
        if smoothness > 0:
            base_z = gaussian_filter(base_z, sigma=smoothness, mode='nearest')
        base_z *= z_scale

        # random anchor points for wave simulation
        np.random.seed(random_seed)
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
            sigma_sq = (influence_sigma**2)*((x.max()-x.min())**2 + (y.max()-y.min())**2)
            return np.exp(-dist_sq/(2*sigma_sq))

        anchor_weights = []
        for i in range(key_points):
            wmap = anchor_weight(i, xi, yi)
            anchor_weights.append(wmap)

        frames = []
        for frame_i in range(key_frames):
            t = frame_i / (key_frames - 1) if key_frames>1 else 0.0
            numerator   = np.zeros_like(base_z)
            denominator = np.zeros_like(base_z)

            for i in range(key_points):
                wmap = anchor_weights[i]
                wave_map = anchor_wave(i, xi, yi, t)
                numerator += wmap*wave_map
                denominator += wmap
            denominator = np.where(denominator==0, 1e-9, denominator)
            total_wave = numerator / denominator

            zi_frame = base_z + total_wave
            frame_data = go.Surface(
                x=xi,
                y=yi,
                z=zi_frame,
                surfacecolor=zi_frame,
                colorscale=color_map,
                colorbar=dict(title='NDVI'),
                cmin=z_min,
                cmax=z_max
            )
            frames.append(
                go.Frame(data=[frame_data], name=f"frame{frame_i}")
            )

        fig = go.Figure(
            data=[go.Surface(
                x=xi,
                y=yi,
                z=base_z,
                surfacecolor=base_z,
                colorscale=color_map,
                colorbar=dict(title='NDVI'),
                cmin=z_min,
                cmax=z_max
            )],
            frames=frames
        )
        fig.update_layout(
            title='3D Sea Simulation (NDVI-based)',
            width=900, height=900,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                zaxis=dict(range=[z_min, z_max])
            ),
            updatemenus=[
                dict(
                    type='buttons',
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
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                    "mode": "immediate"
                                }
                            ]
                        )
                    ],
                    x=0.05, xanchor="left", y=1, yanchor="top"
                )
            ]
        )
        return fig

    except Exception as e:
        st.error(f"Error in sea simulation: {e}")
        logger.exception("Sea simulation error")
        return None


def create_3d_simulation_plot_time_interpolation(
    data_sheets,
    grid_size=100,
    color_map="Viridis",
    z_scale=1.0,
    smoothness=0.0,
    steps_between_sheets=10
):
    """
    Time-series interpolation of NDVI across multiple sheets
    (like your original multi-sheet approach).
    """
    try:
        sheet_order = list(data_sheets.keys())
        if len(sheet_order) < 2:
            st.warning("Only one sheet found => no multi-sheet time interpolation.")
            return None

        # find global lat/lon min/max
        lat_min = float('inf')
        lat_max = float('-inf')
        lon_min = float('inf')
        lon_max = float('-inf')
        flattened = []
        for s in sheet_order:
            la = data_sheets[s]["lat"]
            lo = data_sheets[s]["lon"]
            nd = data_sheets[s]["ndvi"]
            lat_min = min(lat_min, la.min())
            lat_max = max(lat_max, la.max())
            lon_min = min(lon_min, lo.min())
            lon_max = max(lon_max, lo.max())

            # flatten
            x_vals = []
            y_vals = []
            z_vals = []
            M = len(la)
            N = len(lo)
            for i in range(M):
                for j in range(N):
                    x_vals.append(lo[j])
                    y_vals.append(la[i])
                    z_vals.append(nd[i,j])
            flattened.append((x_vals, y_vals, z_vals))

        xi = np.linspace(lon_min, lon_max, grid_size)
        yi = np.linspace(lat_min, lat_max, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # For each sheet => get NDVI grid
        ndvi_grids = []
        global_min = float('inf')
        global_max = float('-inf')
        for (xv, yv, zv) in flattened:
            points = np.column_stack((xv, yv))
            z_grid = griddata(points, zv, (xi, yi), method='linear')
            z_grid = np.nan_to_num(z_grid, nan=0.0)
            if smoothness>0:
                z_grid = gaussian_filter(z_grid, sigma=smoothness, mode='nearest')
            z_grid *= z_scale
            global_min = min(global_min, z_grid.min())
            global_max = max(global_max, z_grid.max())
            ndvi_grids.append(z_grid)

        frames = []
        ndvi_first = ndvi_grids[0]
        for i in range(len(ndvi_grids)-1):
            start_grid = ndvi_grids[i]
            end_grid = ndvi_grids[i+1]
            for step in range(1, steps_between_sheets+1):
                alpha = step / float(steps_between_sheets)
                ndvi_interp = (1-alpha)*start_grid + alpha*end_grid
                fr_data = go.Surface(
                    x=xi,
                    y=yi,
                    z=ndvi_interp,
                    surfacecolor=ndvi_interp,
                    colorscale=color_map,
                    colorbar=dict(title='NDVI'),
                    cmin=global_min,
                    cmax=global_max
                )
                frames.append(
                    go.Frame(data=[fr_data], name=f"frame_{i}_{step}")
                )

        fig = go.Figure(
            data=[go.Surface(
                x=xi,
                y=yi,
                z=ndvi_first,
                surfacecolor=ndvi_first,
                colorscale=color_map,
                colorbar=dict(title='NDVI'),
                cmin=global_min,
                cmax=global_max
            )],
            frames=frames
        )
        fig.update_layout(
            title='Time-Series Interpolation of NDVI',
            width=900,
            height=900,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                zaxis=dict(range=[global_min, global_max])
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, {
                                "frame": {"duration": 100, "redraw": True},
                                "transition": {"duration": 0, "easing": "linear"},
                                "fromcurrent": True,
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[
                                [None],
                                {"frame": {"duration": 0, "redraw": False},
                                 "transition": {"duration": 0},
                                 "mode": "immediate"}
                            ]
                        )
                    ]
                )
            ]
        )
        return fig

    except Exception as e:
        st.error(f"Time-series simulation error: {e}")
        logger.exception("Time-series simulation error")
        return None
