import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import logging
from io import BytesIO
import matplotlib.image as mpimg
import math

import mpld3
from mpld3 import plugins
from scipy.interpolate import griddata
import plotly.express as px
from typing import Tuple, Dict


# ======== CUSTOM NDVI COLORSCALE ========

def get_custom_ndvi_colorscale():
    """
    Returns a custom Plotly colorscale that matches the 2D interactive plot's
    red-orange-yellow-green color scheme.
    """
    return [
        [0.0, 'red'],
        [0.33, 'orange'], 
        [0.66, 'yellow'],
        [1.0, 'green']
    ]


# ======== MPLD3 for interactive hover tooltips ========


logger = logging.getLogger(__name__)


def compute_google_zoom(lat_min, lat_max, lon_min, lon_max, image_width=640):
    """
    Compute an approximate integer zoom level so that the bounding box
    [lat_min..lat_max, lon_min..lon_max] fits entirely in a 640-pixel
    Google Static Map.
    """
    lat_range = abs(lat_max - lat_min)
    lon_range = abs(lon_max - lon_min)
    deg = max(lat_range, lon_range)
    if deg <= 0:
        return 14

    base_dpp = 360.0 / 256.0
    desired_dpp = deg / float(image_width)
    z_float = math.log2(base_dpp / desired_dpp)
    z_rounded = int(round(z_float))
    z_rounded = max(0, min(21, z_rounded))
    return z_rounded


def fetch_google_static_map(lat_min, lat_max, lon_min, lon_max, api_key, img_size=(640, 640)):
    """
    Fetch a satellite image from Google Static Maps using a computed zoom
    that ensures the bounding box [lat_min..lat_max, lon_min..lon_max]
    is fully visible.
    """
    if not api_key:
        return None

    center_lat = (lat_min + lat_max) / 2.0
    center_lon = (lon_min + lon_max) / 2.0

    width, height = img_size
    zoom = compute_google_zoom(lat_min, lat_max, lon_min, lon_max, image_width=width)

    url = (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={center_lat},{center_lon}"
        f"&zoom={zoom}"
        f"&size={width}x{height}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )

    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            image = mpimg.imread(BytesIO(resp.content))
            return image
        else:
            logger.warning(f"Google Static Maps Error: {resp.status_code}")
            return None
    except Exception as e:
        logger.exception("Failed to fetch Google Static Map")
        return None


def create_2d_scatter_plot_ndvi(
    lat,
    lon,
    ndvi_matrix,
    sheet_name="NDVI Sheet",
    google_api_key=None,
    margin_frac=0.05
):
    """
    Original, non-interactive Matplotlib version
    with a black figure background, white text, 
    and a Google satellite map behind the points.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    try:
        plt.style.use('default')

        # Flatten NDVI if 2D
        if ndvi_matrix.ndim == 2:
            ndvi_data = ndvi_matrix.flatten()
        else:
            ndvi_data = ndvi_matrix

        # Flatten lat/lon if needed
        if lat.ndim == 1 and lon.ndim == 1:
            X, Y = np.meshgrid(lon, lat)  # X=lon, Y=lat
            x_plot = X.flatten()
            y_plot = Y.flatten()
        else:
            x_plot = lon
            y_plot = lat

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))

        # Make the area outside the axes black
        fig.patch.set_facecolor("black")
        # Make the actual axes region transparent, so the map shows through
        ax.set_facecolor("none")

        # White text for axes, ticks, spines
        ax.set_title(f"Visualización NDVI - {sheet_name}", color='white')
        ax.set_xlabel("X (Longitud)", color='white')
        ax.set_ylabel("Y (Latitud)", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

        # Determine bounding box
        lat_min, lat_max = np.min(y_plot), np.max(y_plot)
        lon_min, lon_max = np.min(x_plot), np.max(x_plot)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        lat_min_adj = lat_min - margin_frac * lat_range
        lat_max_adj = lat_max + margin_frac * lat_range
        lon_min_adj = lon_min - margin_frac * lon_range
        lon_max_adj = lon_max + margin_frac * lon_range

        # Attempt to fetch Google Maps background
        map_img = None
        if google_api_key:
            map_img = fetch_google_static_map(
                lat_min_adj, lat_max_adj,
                lon_min_adj, lon_max_adj,
                google_api_key
            )

        # If map is found, display it behind the points
        if map_img is not None:
            ax.imshow(
                map_img,
                extent=[lon_min_adj, lon_max_adj, lat_min_adj, lat_max_adj],
                origin='upper',
                zorder=0
            )

        # Set bounding box, aspect ratio
        ax.set_xlim(lon_min_adj, lon_max_adj)
        ax.set_ylim(lat_min_adj, lat_max_adj)
        ax.set_aspect('equal', 'box')

        # Plot the scatter with NDVI
        sc = ax.scatter(
            x_plot,
            y_plot,
            c=ndvi_data,
            cmap='viridis',
            vmin=-1,
            vmax=1,
            alpha=0.9,
            zorder=1
        )

        # Format axis tick labels with 4 decimals
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

        # Colorbar => also white text & black outline
        cbar = fig.colorbar(sc, ax=ax, label='NDVI Index', shrink=0.6)
        cbar.outline.set_edgecolor('white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        for lbl in cbar.ax.yaxis.get_ticklabels():
            lbl.set_color('white')

        fig.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Error creando gráfico de dispersión 2D: {e}")
        return None
    
# -------------------------------------------------------------------
# PLOTLY MOBILE‑FRIENDLY 2‑D NDVI + RISK HEAT‑MAP
# -------------------------------------------------------------------
import streamlit as st
import logging
def _log(msg: str):
    """Write to Streamlit + logger simultaneously."""
    st.text(msg)
    logger.info(msg)



def create_2d_scatter_plot_ndvi_plotly(
    qgis_df: pd.DataFrame,
    sheet_name: str = "NDVI Sheet",
    margin_frac: float = 0.05,
    compact_mode: bool = True,
    debug: bool = False,               # set True to print debug info in Streamlit + logs
) -> Tuple[go.Figure, pd.DataFrame]:
    required = {"long-xm", "long-ym", "NDVI", "Riesgo"}
    missing = required - set(qgis_df.columns)
    if missing:
        raise ValueError(f"QGIS DF missing required columns: {missing}")

    # ── Debug: raw dtypes ─────────────────────────────────────────────
    if debug:
        _log("🧩 RAW DTYPES")
        _log(qgis_df[list(required)].dtypes.to_string())

    # ── Coerce to numeric ────────────────────────────────────────────
    x      = pd.to_numeric(qgis_df["long-xm"], errors="coerce")
    y      = pd.to_numeric(qgis_df["long-ym"], errors="coerce")
    ndvi   = pd.to_numeric(qgis_df["NDVI"],    errors="coerce")
    riesgo = qgis_df["Riesgo"]

    if debug:
        sample = pd.DataFrame({"x": x, "y": y, "ndvi": ndvi}).head()
        _log("\n🧩 SAMPLE AFTER COERCION")
        _log(sample.to_string(index=False))

    valid = x.notna() & y.notna() & ndvi.notna()
    if debug:
        _log(f"\n🧩 VALID ROWS: {valid.sum()} / {len(qgis_df)}")

    x, y, ndvi, riesgo = x[valid], y[valid], ndvi[valid], riesgo[valid]
    if x.empty:
        st.error("❌ No hay filas de coordenadas válidas para graficar.")
        return go.Figure(), pd.DataFrame()

    # ── Layout knobs ─────────────────────────────────────────────────
    font_size   = 12 if compact_mode else 18
    marker_size = 10 if compact_mode else 18
    margin_vals = dict(l=5, r=5, t=40, b=5) if compact_mode else dict(l=10, r=10, t=50, b=10)
    height      = 450 if compact_mode else None

    # ── Safe-padding so zero spans don’t collapse ───────────────────
    lon_span = x.max() - x.min()
    lat_span = y.max() - y.min()
    if lon_span == 0: lon_span = 2e-5    # ~= 2 m at equator
    if lat_span == 0: lat_span = 2e-5
    lon_pad = margin_frac * lon_span
    lat_pad = margin_frac * lat_span

    # ── Build SVG scatter trace ─────────────────────────────────────
    scatter = go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=marker_size,
            color=ndvi,
            colorscale=get_custom_ndvi_colorscale(),
            cmin=-1, cmax=1,
            colorbar=dict(title="NDVI"),
            line=dict(width=2, color="white"),
            opacity=0.9,
        ),
        customdata=np.stack([ndvi, riesgo, y, x], axis=-1),
        hovertemplate="NDVI: %{customdata[0]:.2f}<br>Riesgo: %{customdata[1]}<extra></extra>",
        showlegend=False,
    )

    fig = go.Figure(scatter)
    fig.update_layout(
        title=dict(text=f"NDVI Interactivo – {sheet_name}", font=dict(size=font_size+4, color="white")),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white", size=font_size),
        margin=margin_vals,
        height=height,
        dragmode="pan",
        clickmode="event+select",
        xaxis=dict(
            title="Longitud",
            range=[x.min() - lon_pad, x.max() + lon_pad],
            fixedrange=True,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitud",
            range=[y.min() - lat_pad, y.max() + lat_pad],
            fixedrange=True,
            scaleanchor="x",
            showgrid=False,
            zeroline=False,
        ),
    )

    table_data = pd.DataFrame({
        "NDVI": ndvi.round(4),
        "Riesgo": riesgo,
        "Latitud": y.round(4),
        "Longitud": x.round(4),
        "Orden": np.arange(1, len(x) + 1),
    })

    return fig, table_data


def create_2d_scatter_plot_ndvi_interactive_qgis(
    qgis_df: pd.DataFrame,
    sheet_name="NDVI Sheet",
    google_api_key=None,
    margin_frac=0.05
):
    """
    Interactive Matplotlib + mpld3 scatter that shows NDVI & Riesgo from QGIS data
    on a black tooltip with white text. The figure background is black outside the axes,
    and the axes region is transparent so the map is visible behind the points.
    """

    try:
        # 1) Ensure required columns
        for col in ["long-xm", "long-ym", "NDVI", "Riesgo"]:
            if col not in qgis_df.columns:
                st.error(f"Falta la columna '{col}' en el DataFrame QGIS => no se puede graficar.")
                return None

        x_plot = qgis_df["long-xm"].values  # longitude
        y_plot = qgis_df["long-ym"].values  # latitude
        ndvi_vals = qgis_df["NDVI"].values
        riesgo_vals = qgis_df["Riesgo"].values

        # 2) Compute color limits
        cmin, cmax = np.min(ndvi_vals), np.max(ndvi_vals)

        # 3) Create figure
        plt.style.use('default')
        plt.rcParams.update({'xtick.color': 'white', 'ytick.color': 'white'})
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        ax.set_xlabel("Longitud", fontsize=12, fontweight='bold', color='white')  
        ax.set_ylabel("Latitud", fontsize=12, fontweight='bold', color='white')
        ax.set_title(f"NDVI Interactivo - {sheet_name}", fontsize=20, fontweight='bold', color='white')

        for label in ax.get_xticklabels(): label.set_color('white')
        for label in ax.get_yticklabels(): label.set_color('white')
        for spine in ax.spines.values(): spine.set_edgecolor('white')

        # 4) Bounding box
        lon_min, lon_max = x_plot.min(), x_plot.max()
        lat_min, lat_max = y_plot.min(), y_plot.max()
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        lon_min_adj = lon_min - margin_frac * lon_range
        lon_max_adj = lon_max + margin_frac * lon_range
        lat_min_adj = lat_min - margin_frac * lat_range
        lat_max_adj = lat_max + margin_frac * lat_range

        # 5) Fetch Google Static Map
        map_img = None
        if google_api_key:
            map_img = fetch_google_static_map(
                lat_min_adj, lat_max_adj,
                lon_min_adj, lon_max_adj,
                google_api_key
            )
            print("AVAILABLE Map Key---------------------------")
        else:
            print("NO Map Key---------------------------")

        # 6) Show Google map
        if map_img is not None:
            ax.imshow(
                map_img,
                extent=[lon_min_adj, lon_max_adj, lat_min_adj, lat_max_adj],
                origin='upper',
                zorder=0
            )

        ax.set_xlim(lon_min_adj, lon_max_adj)
        ax.set_ylim(lat_min_adj, lat_max_adj)
        ax.set_aspect('equal', 'box')

        # 7) RBF interpolation for full heatmap coverage
        from matplotlib import cm
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.interpolate import Rbf

        grid_res = 300
        grid_lon = np.linspace(lon_min_adj, lon_max_adj, grid_res)
        grid_lat = np.linspace(lat_min_adj, lat_max_adj, grid_res)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        rbf_interpolator = Rbf(x_plot, y_plot, ndvi_vals, function='linear')
        grid_ndvi_rbf = rbf_interpolator(grid_x, grid_y)
        masked_grid_ndvi = np.clip(grid_ndvi_rbf, cmin, cmax)

        # Custom colormap with dynamic min/max
        vmin, vmax = float(ndvi_vals.min()), float(ndvi_vals.max())
        colors = ['red', 'orange', 'yellow', 'green']
        cmap = LinearSegmentedColormap.from_list('custom_ndvi', colors, N=256)
        
        ax.imshow(
            masked_grid_ndvi,
            extent=(lon_min_adj, lon_max_adj, lat_min_adj, lat_max_adj),
            origin='lower',
            cmap=cmap,
            alpha=0.55,
            vmin=vmin,
            vmax=vmax,
            zorder=1
        )

        # 8) Scatter plot
        point_alpha = 0.5
        point_size = 300
        sc = ax.scatter(
            x_plot,
            y_plot,
            c=ndvi_vals,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=point_alpha,
            s=point_size,
            zorder=2
        )

        # 8.1) Annotate points
        indices_sorted = sorted(range(len(x_plot)), key=lambda i: (-y_plot[i], x_plot[i]))
        y_offset = -0.015 * (lat_max_adj - lat_min_adj)
        for order, idx in enumerate(indices_sorted, start=1):
            ax.text(
                x_plot[idx], y_plot[idx] + y_offset,
                str(order),
                fontsize=20,
                color='white',
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=2
            )

        # 9) Tick formatting
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

        for label in ax.get_xticklabels(): label.set_fontsize(10)
        for label in ax.get_yticklabels(): label.set_fontsize(10)

        fig.tight_layout()

        # 10) Tooltips
        labels = []
        for i in range(len(x_plot)):
            nd = ndvi_vals[i]
            rg = riesgo_vals[i]
            lat = y_plot[i]
            lon = x_plot[i]
            lbl = (
                f"<div class='custom-tooltip' "
                f"style='background-color:black; color:white; padding:8px; "
                f"border: 1px solid white; border-radius:5px;'>"
                f"NDVI={nd:.4f}<br>Lat={lat:.4f}<br>Long={lon:.4f}<br>Riesgo={rg}"
                f"</div>"
            )
            labels.append(lbl)

        tooltip = plugins.PointHTMLTooltip(sc, labels=labels, css="font-size:12px; font-family:sans-serif;")
        plugins.connect(fig, tooltip)

        html_str = mpld3.fig_to_html(fig)

        # 11) JavaScript
        custom_js = """
        <script>
            function adjustTooltipPosition(event, d) {
                let tooltip = document.querySelector(".mpld3-tooltip");
                if (!tooltip) return;
                let chartWidth = window.innerWidth;
                let chartHeight = window.innerHeight;
                let tooltipWidth = tooltip.offsetWidth;
                let tooltipHeight = tooltip.offsetHeight;
                let mouseX = event.pageX;
                let mouseY = event.pageY;
                if (mouseX + tooltipWidth > chartWidth) {
                    tooltip.style.left = (mouseX - tooltipWidth - 10) + "px";
                } else {
                    tooltip.style.left = (mouseX + 10) + "px";
                }
                if (mouseY + tooltipHeight > chartHeight) {
                    tooltip.style.top = (mouseY - tooltipHeight - 10) + "px";
                } else {
                    tooltip.style.top = (mouseY + 10) + "px";
                }
            }
            document.addEventListener("mousemove", function(event) {
                let tooltip = document.querySelector(".mpld3-tooltip");
                if (tooltip) adjustTooltipPosition(event);
            });
        </script>
        """

        # 12) CSS
        custom_css = """
        <style>
            .mpld3-xaxis text, .mpld3-yaxis text,
            .mpld3-axes text, .mpld3-title text {
                fill: white !important;
                font-weight: bold !important;
            }
            .mpld3-tooltip {
                background-color: black !important;
                color: white !important;
                font-weight: bold !important;
                border: 1px solid white !important;
                padding: 5px;
                border-radius: 5px;
                position: absolute;
                z-index: 1000;
                pointer-events: none;
                max-width: 200px;
                word-wrap: break-word;
            }
        </style>
        """

        html_str = custom_css + html_str + custom_js
        return html_str

    except Exception as e:
        st.error(f"Error creando dispersión 2D interactiva QGIS: {e}")
        logger.exception("Error in create_2d_scatter_plot_ndvi_interactive_qgis")
        return None


def create_3d_surface_plot(
    data,
    grid_size=100,
    color_map="viridis",
    z_scale=1.0,
    smoothness=0.0
):
    try:
        logger.info("Creating 3D surface plot (colored by NDVI)")
        y = data['Longitud'].values
        x = data['Latitud'].values
        z = data['NDVI'].values

        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        points = np.column_stack((x, y))
        zi = griddata(points, z, (xi, yi), method='linear')
        zi = np.nan_to_num(zi, nan=0.0)

        if smoothness > 0:
            zi = gaussian_filter(zi, sigma=smoothness, mode='nearest')

        zi *= z_scale
        cmin, cmax = zi.min(), zi.max()
        
        # Use custom NDVI colorscale to match 2D interactive plot
        custom_colorscale = get_custom_ndvi_colorscale()

        fig = go.Figure(data=[go.Surface(
            x=xi,
            y=yi,
            z=zi,
            surfacecolor=zi,
            colorscale=custom_colorscale,
            colorbar=dict(title='NDVI'),
            cmin=cmin,
            cmax=cmax
        )])
        fig.update_layout(
            title='Superficie 3D de Salud de Cultivos (NDVI)',
            width=1000,
            height=1000,
            scene=dict(
                xaxis_title='Longitud',
                yaxis_title='Latitud',
                zaxis_title='NDVI'
            ),
            autosize=True
        )
        fig.update_traces(
            hovertemplate=(
                'Longitud: %{x:.2f}<br>'
                'Latitud: %{y:.2f}<br>'
                'NDVI: %{z:.3f}'
            )
        )
        return fig

    except Exception as e:
        logger.exception("Error in create_3d_surface_plot")
        st.error(f"Error en create_3d_surface_plot: {e}")
        return None


def create_3d_simulation_plot_sea_keypoints(
    data,
    grid_size=50,
    color_map="viridis",
    z_scale=1.0,
    smoothness=0.0,
    key_frames=100,
    key_points=20,
    wave_amplitude_range=(-0.05, 0.7),
    wave_frequency_range=(0.05, 0.4),
    wave_speed_range=(0.5, 2.0),
    influence_sigma=0.1,
    random_seed=42,
    z_min=0,
    z_max=0.9
):
    try:
        x = data['Longitud'].values
        y = data['Latitud'].values
        z = data['NDVI'].values

        xi = np.linspace(x.min(), x.max(), grid_size)
        yi = np.linspace(y.min(), y.max(), grid_size)
        xi, yi = np.meshgrid(xi, yi)

        base_z = griddata((x, y), z, (xi, yi), method='linear')
        base_z = np.nan_to_num(base_z, nan=0.0)
        if smoothness > 0:
            base_z = gaussian_filter(base_z, sigma=smoothness, mode='nearest')
        base_z *= z_scale

        np.random.seed(random_seed)
        anchor_x = np.random.uniform(x.min(), x.max(), key_points)
        anchor_y = np.random.uniform(y.min(), y.max(), key_points)

        wave_amps = np.random.uniform(*wave_amplitude_range, size=key_points)
        wave_freq_x = np.random.uniform(*wave_frequency_range, size=key_points)
        wave_freq_y = np.random.uniform(*wave_frequency_range, size=key_points)
        wave_phase = np.random.uniform(0, 2*np.pi, key_points)
        wave_speed = np.random.uniform(*wave_speed_range, size=key_points)

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
            t = frame_i / (key_frames - 1) if key_frames > 1 else 0.0
            numerator = np.zeros_like(base_z)
            denominator = np.zeros_like(base_z)

            for k_i in range(key_points):
                wmap = anchor_weights[k_i]
                wave_map = anchor_wave(k_i, xi, yi, t)
                numerator += wmap * wave_map
                denominator += wmap

            denominator = np.where(denominator==0, 1e-9, denominator)
            total_wave = numerator / denominator
            zi_frame = base_z + total_wave

            frame_data = go.Surface(
                x=xi,
                y=yi,
                z=zi_frame,
                surfacecolor=zi_frame,
                colorscale=get_custom_ndvi_colorscale(),
                colorbar=dict(title='NDVI'),
                cmin=z_min,
                cmax=z_max
            )
            frames.append(go.Frame(data=[frame_data], name=f"frame{frame_i}"))

        fig = go.Figure(
            data=[go.Surface(
                x=xi,
                y=yi,
                z=base_z,
                surfacecolor=base_z,
                colorscale=get_custom_ndvi_colorscale(),
                colorbar=dict(title='NDVI'),
                cmin=z_min,
                cmax=z_max
            )],
            frames=frames
        )
        fig.update_layout(
            title='Simulación 3D Marina (basada en NDVI)',
            width=900, height=900,
            scene=dict(
                xaxis_title='Longitud',
                yaxis_title='Latitud',
                zaxis_title='NDVI',
                zaxis=dict(range=[z_min, z_max])
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Reproducir',
                            method='animate',
                            args=[None, {
                                "frame": {"duration": 33, "redraw": True},
                                "transition": {"duration": 0, "easing": "linear"},
                                "fromcurrent": True,
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label='Pausar',
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
        st.error(f"Error en simulación marina: {e}")
        logger.exception("Sea simulation error")
        return None


def create_3d_simulation_plot_time_interpolation(
    data_sheets,
    grid_size=100,
    color_map="viridis",
    z_scale=1.0,
    smoothness=0.0,
    steps_between_sheets=10
):
    """
    Time-series interpolation of NDVI across multiple sheets.
    """
    try:
        sheet_order = list(data_sheets.keys())
        if len(sheet_order) < 2:
            st.warning("Only one sheet found => no multi-sheet time interpolation.")
            return None

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

            x_vals, y_vals, z_vals = [], [], []
            M = len(la)
            N = len(lo)
            for i in range(M):
                for j in range(N):
                    x_vals.append(lo[j])
                    y_vals.append(la[i])
                    z_vals.append(nd[i, j])
            flattened.append((x_vals, y_vals, z_vals))

        xi = np.linspace(lon_min, lon_max, grid_size)
        yi = np.linspace(lat_min, lat_max, grid_size)
        xi, yi = np.meshgrid(xi, yi)

        ndvi_grids = []
        global_min = 0
        global_max = 0.9

        for (xv, yv, zv) in flattened:
            points = np.column_stack((xv, yv))
            z_grid = griddata(points, zv, (xi, yi), method='linear')
            z_grid = np.nan_to_num(z_grid, nan=0.0)
            if smoothness > 0:
                z_grid = gaussian_filter(z_grid, sigma=smoothness, mode='nearest')
            z_grid *= z_scale
            ndvi_grids.append(z_grid)

        frames = []
        ndvi_first = ndvi_grids[0]
        # Use custom NDVI colorscale to match 2D interactive plot
        custom_colorscale = get_custom_ndvi_colorscale()
        
        # Create spheres and text for original data points
        sphere_frames = []
        for sheet_idx, (x_vals, y_vals, z_vals) in enumerate(flattened):
            # Create spheres at original data points
            sphere_trace = go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=[z * z_scale + 0.02 for z in z_vals],  # Slight offset above surface
                mode='markers+text',
                marker=dict(
                    size=6,
                    color=z_vals,
                    colorscale=custom_colorscale,
                    cmin=global_min,
                    cmax=global_max,
                    opacity=0.9,
                    line=dict(width=1, color='white')
                ),
                text=[f'{ndvi:.3f}' for ndvi in z_vals],
                textposition="top center",
                textfont=dict(size=8, color="white"),
                hovertemplate='<b>NDVI:</b> %{marker.color:.3f}<br>' +
                             '<b>Lon:</b> %{x:.6f}<br>' +
                             '<b>Lat:</b> %{y:.6f}<extra></extra>',
                name=f'Puntos NDVI - Hoja {sheet_idx + 1}',
                showlegend=False
            )
            sphere_frames.append(sphere_trace)
        
        for i in range(len(ndvi_grids) - 1):
            start_grid = ndvi_grids[i]
            end_grid = ndvi_grids[i + 1]
            
            # Get corresponding sphere data for interpolation
            start_spheres = sphere_frames[i]
            end_spheres = sphere_frames[i + 1] if i + 1 < len(sphere_frames) else sphere_frames[i]
            
            for step in range(1, steps_between_sheets + 1):
                alpha = step / float(steps_between_sheets)
                ndvi_interp = (1 - alpha) * start_grid + alpha * end_grid
                
                # Create interpolated surface
                fr_surface = go.Surface(
                    x=xi,
                    y=yi,
                    z=ndvi_interp,
                    surfacecolor=ndvi_interp,
                    colorscale=custom_colorscale,
                    colorbar=dict(title='NDVI'),
                    cmin=global_min,
                    cmax=global_max,
                    showscale=True
                )
                
                # Interpolate sphere positions and colors for smooth animation
                if i + 1 < len(sphere_frames):
                    interp_x = [(1 - alpha) * start_spheres.x[j] + alpha * end_spheres.x[j] 
                               for j in range(len(start_spheres.x))]
                    interp_y = [(1 - alpha) * start_spheres.y[j] + alpha * end_spheres.y[j] 
                               for j in range(len(start_spheres.y))]
                    interp_z = [(1 - alpha) * start_spheres.z[j] + alpha * end_spheres.z[j] 
                               for j in range(len(start_spheres.z))]
                    interp_colors = [(1 - alpha) * start_spheres.marker.color[j] + alpha * end_spheres.marker.color[j] 
                                   for j in range(len(start_spheres.marker.color))]
                else:
                    interp_x = list(start_spheres.x)
                    interp_y = list(start_spheres.y)
                    interp_z = list(start_spheres.z)
                    interp_colors = list(start_spheres.marker.color)
                
                # Create interpolated spheres
                fr_spheres = go.Scatter3d(
                    x=interp_x,
                    y=interp_y,
                    z=interp_z,
                    mode='markers+text',
                    marker=dict(
                        size=6,
                        color=interp_colors,
                        colorscale=custom_colorscale,
                        cmin=global_min,
                        cmax=global_max,
                        opacity=0.9,
                        line=dict(width=1, color='white')
                    ),
                    text=[f'{ndvi:.3f}' for ndvi in interp_colors],
                    textposition="top center",
                    textfont=dict(size=8, color="white"),
                    hovertemplate='<b>NDVI:</b> %{marker.color:.3f}<br>' +
                                 '<b>Lon:</b> %{x:.6f}<br>' +
                                 '<b>Lat:</b> %{y:.6f}<extra></extra>',
                    name='Puntos NDVI',
                    showlegend=False
                )
                
                frames.append(go.Frame(data=[fr_surface, fr_spheres], name=f"frame_{i}_{step}"))

        # ✅ Fixed: White Axis Labels & Tick Labels
        # Create initial surface
        initial_surface = go.Surface(
            x=xi,
            y=yi,
            z=ndvi_first,
            surfacecolor=ndvi_first,
            colorscale=custom_colorscale,
            colorbar=dict(title='NDVI'),
            cmin=global_min,
            cmax=global_max
        )
        
        # Create initial spheres for first data sheet
        if sphere_frames:
            initial_spheres = sphere_frames[0]
        else:
            # Fallback if no sphere frames
            initial_spheres = go.Scatter3d(x=[], y=[], z=[], mode='markers')
        
        fig = go.Figure(
            data=[initial_surface, initial_spheres],
            frames=frames
        )

        fig.update_layout(
            title='Interpolación de Series Temporales de NDVI',
            width=900,
            height=900,
            scene=dict(
                xaxis=dict(
                    title=dict(text='Longitud', font=dict(color="white")),
                    tickfont=dict(color="white"),
                    color="white"  # ✅ Forces white axis elements
                ),
                yaxis=dict(
                    title=dict(text='Latitud', font=dict(color="white")),  # ✅ White Title
                    tickfont=dict(color="white")  # ✅ White Tick Labels
                ),
                zaxis=dict(
                    title=dict(text='NDVI', font=dict(color="white")),  # ✅ White Title
                    tickfont=dict(color="white"),  # ✅ White Tick Labels
                    range=[global_min, global_max]
                )
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    buttons=[
                        dict(
                            label='Reproducir',
                            method='animate',
                            args=[None, {
                                "frame": {"duration": 100, "redraw": True},
                                "transition": {"duration": 0, "easing": "linear"},
                                "fromcurrent": True,
                                "mode": "immediate"
                            }]
                        ),
                        dict(
                            label='Pausar',
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
                    ]
                )
            ]
        )
        return fig
    except Exception as e:
        st.error(f"Error de simulación de series temporales: {e}")
        logger.exception("Time-series simulation error")
        return None