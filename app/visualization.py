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
    """
    Creates a 2D Matplotlib scatter plot of NDVI vs. Longitude & Latitude
    with a dark/transparent background.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import streamlit as st
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Creating 2D NDVI scatter for sheet: {sheet_name}")

        # Use a dark (or any custom) background style
        plt.style.use('dark_background')

        # lat.shape => M, lon.shape => N, ndvi_matrix.shape => (M, N)
        X, Y = np.meshgrid(lon, lat)

        # Create the figure & axes with facecolor="none" or any color you like
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("none")  # Remove figure's own background
        ax.set_facecolor("none")         # Remove Axes background

        # Scatter plot
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

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, label='NDVI Index')

        fig.tight_layout()

        logger.info(f"2D NDVI scatter created successfully for {sheet_name}")
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
    """
    Creates a single (static) 3D surface plot, colored by NDVI.
    Expects `data` to have columns: [Longitud, Latitud, NDVI].
    Ignores any 'Riesgo' column if present.
    """
    try:
        logger.info("Creating 3D surface plot (colored by NDVI)")

        x = data['Longitud'].values  # scattered X coords
        y = data['Latitud'].values   # scattered Y coords
        z = data['NDVI'].values      # NDVI values

        # Create a regular grid
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate NDVI onto the grid
        from data_processing import griddata_points_to_grid
        zi = griddata_points_to_grid(x, y, z, xi, yi)

        # Optional smoothing
        if smoothness > 0:
            zi = gaussian_filter(zi, sigma=smoothness, mode='nearest')

        # Scale the NDVI vertically
        zi *= z_scale

        # Determine min/max for color scale
        cmin = zi.min()
        cmax = zi.max()
        # If you prefer a fixed range, e.g., [-1, 1], do: cmin, cmax = -1, 1

        # Build a Plotly surface
        fig = go.Figure(data=[go.Surface(
            x=xi,
            y=yi,
            z=zi,
            surfacecolor=zi,    # color by NDVI
            colorscale=color_map,
            colorbar=dict(title='NDVI'),
            cmin=cmin,
            cmax=cmax
        )])

        # Layout
        fig.update_layout(
            title='Crop Health 3D Surface (NDVI)',
            width=1000,
            height=1000,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                zaxis=dict(range=[cmin, cmax])  # or auto
            ),
            autosize=True
        )

        # Customize hover info
        fig.update_traces(
            hovertemplate=(
                'Longitude: %{x:.2f}<br>'
                'Latitude: %{y:.2f}<br>'
                'NDVI: %{z:.3f}'
            )
        )

        logger.info("3D surface plot created successfully (NDVI-based)")
        return fig

    except Exception as e:
        error_message = f"An error occurred in create_3d_surface_plot (NDVI-based): {e}"
        st.error(error_message)
        logger.exception("Error in create_3d_surface_plot (NDVI-based)")
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
    Creates a 3D animated 'sea-like' surface with random anchor points & wave params (wave-based simulation).
    Uses NDVI as a base, then adds wave perturbations.
    """
    try:
        logger.info("Creating 3D 'sea' simulation (anchor-based) with NDVI-based coloring")

        # Extract columns
        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values
        # If Riesgo is present, we won't use it for coloring here
        # e.g., riesgo = data['Riesgo'].values

        # Create grid
        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        from data_processing import griddata_points_to_grid
        zi_original = griddata_points_to_grid(x, y, z, xi, yi)

        # Optional smoothing
        if smoothness > 0:
            zi_original = gaussian_filter(zi_original, sigma=smoothness, mode='nearest')

        # Vertical scale
        zi_original *= z_scale

        # Random anchor points for wave simulation
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
            sigma_sq = (influence_sigma**2) * ((x.max()-x.min())**2 + (y.max()-y.min())**2)
            return np.exp(-dist_sq / (2*sigma_sq))

        # Precompute anchor weights for all points in the grid
        anchor_weights = []
        for i in range(key_points):
            wmap = anchor_weight(i, xi, yi)
            anchor_weights.append(wmap)

        # Build frames for the animation
        frames = []
        for frame_i in range(key_frames):
            t = frame_i / (key_frames - 1) if key_frames > 1 else 0.0

            numerator   = np.zeros_like(zi_original)
            denominator = np.zeros_like(zi_original)

            for i in range(key_points):
                wmap = anchor_weights[i]
                wave_map = anchor_wave(i, xi, yi, t)
                numerator   += wmap * wave_map
                denominator += wmap

            denominator = np.where(denominator == 0, 1e-9, denominator)
            total_wave = numerator / denominator

            zi_frame = zi_original + total_wave

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
                go.Frame(
                    data=[frame_data],
                    name=f"frame{frame_i}",
                    layout=dict(scene=dict(camera=None))
                )
            )

        # Construct the figure with the initial surface
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

        # Add layout & animation controls
        fig.update_layout(
            title='Crop Health 3D Sea Simulation (Anchor-based) - Color by Z',
            width=900,
            height=900,
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='NDVI',
                zaxis=dict(range=[z_min, z_max]),
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[
                                None, {
                                    "frame": {"duration": 33, "redraw": True},
                                    "transition": {"duration": 0, "easing": "linear"},
                                    "fromcurrent": True,
                                    "mode": "immediate"
                                }
                            ]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[
                                [None], {
                                    "frame": {"duration": 0, "redraw": False},
                                    "transition": {"duration": 0},
                                    "mode": "immediate"
                                }
                            ]
                        )
                    ],
                    x=0.05,
                    xanchor="left",
                    y=1,
                    yanchor="top"
                )
            ]
        )

        logger.info("3D 'sea' simulation (anchor-based) created successfully")
        return fig

    except Exception as e:
        error_message = f"An error occurred in sea simulation with key points (color by Z): {e}"
        logger.exception("Sea simulation error")
        st.error(error_message)
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
    Time-Series Interpolation of NDVI across multiple sheets (Mirroring the approach in create_3d_surface_plot).
    
    data_sheets: a dict { sheet_name: {"lon": array, "lat": array, "ndvi": 2D array} }
        Each sheet's 'lon' and 'lat' are 1D arrays, shapes: (N,) and (M,)
        ndvi is a 2D array of shape (M, N).

    We'll:
      1) Flatten each sheet's lat, lon, ndvi_2d into scattered points (x, y, z).
      2) Create a regular grid (xi, yi) based on global min/max of all lat/lon.
      3) Interpolate each sheet's NDVI onto that grid (just like create_3d_surface_plot).
      4) Then linearly interpolate from sheet i's grid to sheet i+1's grid in 'steps_between_sheets' frames.

    That ensures we see an actual 3D NDVI surface, not lat/lon as Z.
    """
    import streamlit as st
    import numpy as np
    import plotly.graph_objects as go
    from scipy.ndimage import gaussian_filter
    from data_processing import griddata_points_to_grid
    import logging

    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating time-based 3D simulation by interpolating NDVI (grid-based).")

        # 1) If there's less than 2 sheets, no multi-sheet animation
        sheet_order = list(data_sheets.keys())  # keep the insertion order or your chosen order
        if len(sheet_order) < 2:
            st.warning("Only one sheet found. No multiple sheets to animate.")
            return None

        # 2) Collect lat/lon min/max across ALL sheets so we can build a single global grid
        global_lat_min = float('inf')
        global_lat_max = float('-inf')
        global_lon_min = float('inf')
        global_lon_max = float('-inf')

        # We'll store the "flattened" data for each sheet
        flattened_sheets = []

        for s in sheet_order:
            lat = data_sheets[s]["lat"]  # shape (M,)
            lon = data_sheets[s]["lon"]  # shape (N,)
            ndvi_2d = data_sheets[s]["ndvi"]  # shape (M, N)

            # Update global min/max for lat/lon
            global_lat_min = min(global_lat_min, lat.min())
            global_lat_max = max(global_lat_max, lat.max())
            global_lon_min = min(global_lon_min, lon.min())
            global_lon_max = max(global_lon_max, lon.max())

            # Flatten lat/lon/ndvi => scattered points (x_vals, y_vals, z_vals)
            x_vals = []
            y_vals = []
            z_vals = []
            M = len(lat)
            N = len(lon)
            for i in range(M):
                for j in range(N):
                    x_vals.append(lon[j])   # x => longitude
                    y_vals.append(lat[i])   # y => latitude
                    z_vals.append(ndvi_2d[i, j])

            flattened_sheets.append((x_vals, y_vals, z_vals))

        # 3) Build a global grid
        xi = np.linspace(global_lon_min, global_lon_max, grid_size)
        yi = np.linspace(global_lat_min, global_lat_max, grid_size)
        xi, yi = np.meshgrid(xi, yi)  # shape (grid_size, grid_size)

        # 4) For each sheet, do scattered->grid interpolation => 2D NDVI array
        ndvi_grids = []
        global_min = float('inf')
        global_max = float('-inf')

        for (x_vals, y_vals, z_vals) in flattened_sheets:
            # Interpolate onto the global grid
            z_grid = griddata_points_to_grid(np.array(x_vals), np.array(y_vals), np.array(z_vals), xi, yi)

            # Smooth if needed
            if smoothness > 0:
                z_grid = gaussian_filter(z_grid, sigma=smoothness, mode='nearest')

            # Scale
            z_grid *= z_scale

            # Track global min/max
            global_min = min(global_min, z_grid.min())
            global_max = max(global_max, z_grid.max())

            ndvi_grids.append(z_grid)

        # 5) Build frames by interpolating from sheet i to sheet i+1
        frames = []
        ndvi_first = ndvi_grids[0]

        for i in range(len(ndvi_grids) - 1):
            start_grid = ndvi_grids[i]
            end_grid   = ndvi_grids[i+1]

            for step in range(1, steps_between_sheets + 1):
                alpha = step / float(steps_between_sheets)
                ndvi_interp = (1 - alpha)*start_grid + alpha*end_grid

                frame_data = go.Surface(
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
                    go.Frame(
                        data=[frame_data],
                        name=f"frame_{i}_{step}",
                        layout=dict(scene=dict(camera=None))
                    )
                )

        # 6) Define initial figure using the first grid
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

        # 7) Layout
        fig.update_layout(
            title='Time-Series Interpolation of NDVI (Multiple Sheets, Grid-based)',
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

        logger.info("Time-based interpolation (grid-based) created successfully.")
        return fig

    except Exception as e:
        error_message = f"An error occurred while creating time-series simulation (grid-based): {e}"
        logger.exception("Time-series simulation error (grid-based)")
        st.error(error_message)
        return None
