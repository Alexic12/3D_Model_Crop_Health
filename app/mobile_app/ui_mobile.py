import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from openpyxl import load_workbook, Workbook
# ────────────────────────── HELPERS ──────────────────────────
from zipfile import BadZipFile
from openpyxl.utils.exceptions import InvalidFileException

logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets" / "data"


# ────────────────────────── HELPERS ──────────────────────────
def _load_or_init_manual_file(manual_fn: Path) -> Dict[str, pd.DataFrame]:
    """
    Return {sheet_name: DataFrame}.  If the xlsx is missing *or*
    corrupted, delete it and return an empty dict.
    """
    if manual_fn.exists():
        try:
            return pd.read_excel(manual_fn, sheet_name=None)
        except (InvalidFileException, BadZipFile, KeyError) as exc:
            logger.warning(f"Corrupted manual file – recreating. ({exc})")
            manual_fn.unlink(missing_ok=True)          # delete bad file
    return {}


def _ensure_sheet_template(df_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a 25-row template with Latitud / Longitud / NDVI / Riesgo / Nuevo Riesgo.
    """
    # Extract original Riesgo data if available, otherwise use 0
    original_riesgo = [0] * 25
    if "Riesgo" in df_sheet.columns:
        original_riesgo = df_sheet["Riesgo"][:25].tolist()
    
    return pd.DataFrame({
        "Latitud":      df_sheet["long-ym"][:25].tolist(),
        "Longitud":     df_sheet["long-xm"][:25].tolist(), 
        "NDVI":         df_sheet["NDVI"][:25].tolist(),
        "Riesgo":       original_riesgo,
        "Nuevo Riesgo": [0] * 25
    })

def _create_full_excel_async(qgis_path, manual_fn: Path, field_name: str) -> None:
    """
    Create complete Excel file with all sheets from QGIS data asynchronously.
    """
    try:
        xls = pd.ExcelFile(qgis_path)
        manual_fn.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(manual_fn, engine="openpyxl", mode="w") as writer:
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(qgis_path, sheet_name=sheet_name)
                    df["long-xm"] = pd.to_numeric(df["long-xm"], errors="coerce")
                    df["long-ym"] = pd.to_numeric(df["long-ym"], errors="coerce")
                    df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")
                    df = df.dropna(subset=["long-xm", "long-ym", "NDVI"]).reset_index(drop=True)
                    
                    if not df.empty:
                        template_df = _ensure_sheet_template(df)
                        template_df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    logger.warning(f"Skipping sheet {sheet_name}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error creating full Excel file: {e}")


def _save_riesgo_sheet(sheet_name: str, df_sheet: pd.DataFrame, manual_fn: Path) -> None:
    """
    Safely create/update one sheet inside Data_Riesgo_MANUAL.xlsx.

    • If file missing/corrupt → create fresh with mode="w".
    • Else open in append mode and atomically replace the sheet.
    """
    manual_fn.parent.mkdir(parents=True, exist_ok=True)

    # ── validate existing workbook ────────────────────────────────────
    append_ok = False
    if manual_fn.exists():
        try:
            load_workbook(manual_fn)
            append_ok = True
        except (InvalidFileException, BadZipFile, KeyError):
            logger.warning("⚠️  Manual file corrupt — recreating from scratch.")
            manual_fn.unlink(missing_ok=True)

    mode = "a" if append_ok else "w"

    # ── build ExcelWriter kwargs --------------------------------------
    writer_kwargs = dict(
        path    = manual_fn,
        engine  = "openpyxl",
        mode    = mode,
    )
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"   # only legal in append

    # ── write sheet ----------------------------------------------------
    try:
        with pd.ExcelWriter(**writer_kwargs) as writer:
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as exc:
        st.error(f"❌ Error al guardar el archivo manual: {exc}")
        logger.exception("Error saving riesgo sheet")
# ──────────────────────── MAIN MOBILE APP ───────────────────────
def render_mobile() -> None:
    st.set_page_config(
        page_title="Salud de Cultivos – Móvil", 
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Import mobile optimizations
    from app.ui.mobile_optimizations import inject_mobile_optimizations, MobileOptimizer, PerformanceOptimizer
    
    # Apply mobile optimizations
    #inject_mobile_optimizations()
    
    # Mobile-first responsive design
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            /* CSS Custom Properties for Design System */
            :root {
                --primary-color: #1f77b4;
                --secondary-color: #ff7f0e;
                --success-color: #2ca02c;
                --warning-color: #ff7f0e;
                --error-color: #d62728;
                --background-color: #ffffff;
                --surface-color: #f8f9fa;
                --text-primary: #212529;
                --text-secondary: #6c757d;
                --border-color: #dee2e6;
                --border-radius: 8px;
                --spacing-xs: 0.25rem;
                --spacing-sm: 0.5rem;
                --spacing-md: 1rem;
                --spacing-lg: 1.5rem;
                --spacing-xl: 2rem;
                --font-size-sm: 0.875rem;
                --font-size-base: 1rem;
                --font-size-lg: 1.125rem;
                --font-size-xl: 1.25rem;
                --line-height-base: 1.5;
                --transition-base: 0.15s ease-in-out;
            }

            /* Dark Mode Support */
            @media (prefers-color-scheme: dark) {
                :root {
                    --background-color: #121212;
                    --surface-color: #1e1e1e;
                    --text-primary: #ffffff;
                    --text-secondary: #b3b3b3;
                    --border-color: #333333;
                }
            }

            
            /* Mobile-first base styles */
            .main .block-container {
                padding: 0.5rem !important;
                max-width: 100% !important;
            }
            
            /* Touch-friendly map container */
            .stApp > div > div > div > div {
                padding: 0 !important;
            }

            .stSelectbox select {
                color: var(--text-primary) !important;
                background-color: var(--background-color) !important;
            }

            /* Fix Dropdown Text Visibility - Comprehensive */
            .stSelectbox label {
                color: var(--text-primary) !important;
            }

            .stSelectbox > div > div {
                color: var(--text-primary) !important;
                background-color: var(--background-color) !important;
            }
            
            .stSelectbox > div > div > div {
                color: var(--text-primary) !important;
                background-color: var(--background-color) !important;
            }
            
            .stSelectbox select {
                color: var(--text-primary) !important;
                background-color: var(--background-color) !important;
            }
            
            .stSelectbox option {
                color: var(--text-primary) !important;
                background-color: var(--background-color) !important;
            }
                
            /* Optimize for small screens */
            @media (max-width: 480px) {
                .main .block-container {
                    padding: 0.25rem !important;
                }
                
                h1 {
                    font-size: 1.5rem !important;
                }
                
                h2, h3 {
                    font-size: 1.25rem !important;
                }
            }
            
            /* Landscape orientation adjustments */
            @media (orientation: landscape) and (max-height: 500px) {
                .main .block-container {
                    padding-top: 0.25rem !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📱 Salud de Cultivos (Móvil)")
    st.markdown("---")

    # --- field selection (matching desktop) -----------------------------
    import os
    base_folder = "./upload_data"
    available_fields = []
    if os.path.exists(base_folder):
        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            if os.path.isdir(item_path):
                available_fields.append(item)
    
    # Field selection dropdown with proper display
    if available_fields:
        field_options = available_fields + ["Crear Nuevo Campo"]
        selected_field = st.selectbox(
            "Seleccionar Campo", 
            field_options,
            key="field_selector"
        )
        if selected_field == "Crear Nuevo Campo":
            field_name = st.text_input("Nombre del Nuevo Campo", value="", key="new_field_input")
        else:
            field_name = selected_field
    else:
        field_name = st.text_input("Nombre del Campo", value="Perimetro Prev", key="default_field_input")
    
    # --- user inputs -----------------------------------------------------
    indice = st.text_input("Índice de Vegetación", value="NDVI")
    anio   = st.text_input("Año", value="2024")
    
    # Satellite imagery selection
    satellite_option = st.selectbox(
        "Tipo de Imagen Satelital",
        [
            "Google Satellite", 
            "Google Hybrid", 
            "Esri Satellite", 
            "USGS Satellite",
            "Street Map"
        ],
        index=0,  # Default to Google Satellite
        key="satellite_selector",
        help="Selecciona el tipo de imagen de fondo para el mapa. Google Hybrid incluye etiquetas sobre la imagen satelital."
    )

    # Clear success messages when field or sheet changes
    current_context = f"{field_name}_{indice}_{anio}"
    if "last_context" not in st.session_state:
        st.session_state["last_context"] = current_context
    elif st.session_state["last_context"] != current_context:
        # Clear all success messages when context changes
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith("success_msg_")]
        for key in keys_to_remove:
            del st.session_state[key]
        st.session_state["last_context"] = current_context
    
    # Use field-specific path if field is selected
    if field_name:
        qgis_path = ASSETS_DIR / field_name / f"INFORME_{indice}_QGIS_{anio}.xlsx"
        # Field-specific manual file with field name in filename
        safe_field_name = field_name.replace(' ', '_').replace('/', '_')
        manual_fn = ASSETS_DIR / field_name / f"Data_Riesgo_MANUAL_{safe_field_name}.xlsx"
    else:
        qgis_path = ASSETS_DIR / f"INFORME_{indice}_QGIS_{anio}.xlsx"
        manual_fn = ASSETS_DIR / "Data_Riesgo_MANUAL.xlsx"
    
    # Display satellite imagery information
    if satellite_option in ["Google Satellite", "Google Hybrid"]:
        st.info("🛰️ **Google Satellite**: Imágenes de alta resolución, actualizadas frecuentemente. Ideal para análisis detallado de cultivos.")
    elif satellite_option == "Esri Satellite":
        st.info("🛰️ **Esri Satellite**: Imágenes de múltiples fuentes, buena cobertura global. Excelente para análisis agrícola.")
    elif satellite_option == "USGS Satellite":
        st.info("🛰️ **USGS Satellite**: Imágenes oficiales del gobierno de EE.UU., muy precisas para análisis científico.")
    
    st.markdown("---")

    if not qgis_path.exists():
        st.warning(f"Archivo QGIS no encontrado en `{qgis_path}`")
        qgis_path = st.file_uploader("Subir Excel QGIS", type=["xlsx", "xls"])
    if not qgis_path:
        st.stop()
    
    # Initialize Excel file asynchronously if it doesn't exist
    if field_name and not manual_fn.exists():
        import threading
        def init_excel():
            _create_full_excel_async(qgis_path, manual_fn, field_name)
        
        if f"excel_init_{safe_field_name}" not in st.session_state:
            st.session_state[f"excel_init_{safe_field_name}"] = True
            thread = threading.Thread(target=init_excel)
            thread.daemon = True
            thread.start()
            st.info(f"📊 Inicializando archivo Excel para {field_name}...")

    try:
        xls = pd.ExcelFile(qgis_path)
    except Exception as exc:
        st.error(f"Error abriendo Excel: {exc}")
        st.stop()

    sheet_name = st.selectbox(
        "Seleccionar hoja NDVI", 
        xls.sheet_names,
        key="sheet_selector"
    )
    try:
        df = pd.read_excel(qgis_path, sheet_name=sheet_name)
    except Exception as exc:
        st.error(f"Error leyendo hoja: {exc}")
        st.stop()

    # --- clean numeric cols ---------------------------------------------
    df["long-xm"] = pd.to_numeric(df["long-xm"], errors="coerce")
    df["long-ym"] = pd.to_numeric(df["long-ym"], errors="coerce")
    df["NDVI"]    = pd.to_numeric(df["NDVI"],    errors="coerce")
    df = df.dropna(subset=["long-xm", "long-ym", "NDVI"]).reset_index(drop=True)
    if df.empty:
        st.error("No hay datos válidos de coordenadas / NDVI.")
        return

    # --- folium map with satellite imagery --------------------------------
    # Define satellite tile configurations
    satellite_configs = {
        "Google Satellite": {
            "tiles": 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            "attr": 'Google Satellite',
            "name": 'Google Satellite'
        },
        "Google Hybrid": {
            "tiles": 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            "attr": 'Google Hybrid (Satellite + Labels)',
            "name": 'Google Hybrid'
        },
        "Esri Satellite": {
            "tiles": 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            "attr": 'Esri World Imagery',
            "name": 'Esri Satellite'
        },
        "USGS Satellite": {
            "tiles": 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
            "attr": 'USGS National Map',
            "name": 'USGS Satellite'
        },
        "Street Map": {
            "tiles": 'OpenStreetMap',
            "attr": 'OpenStreetMap',
            "name": 'Street Map'
        }
    }
    
    # Create map with selected satellite imagery
    selected_config = satellite_configs[satellite_option]
    m = folium.Map(
        location=[df["long-ym"].mean(), df["long-xm"].mean()],
        zoom_start=17, 
        control_scale=True,
        tiles=selected_config["tiles"],
        attr=selected_config["attr"]
    )
    
    # Add all layers with layer control for switching
    for name, config in satellite_configs.items():
        if name != satellite_option:  # Don't add the currently selected one again
            folium.TileLayer(
                tiles=config["tiles"],
                attr=config["attr"],
                name=config["name"],
                overlay=False,
                control=True
            ).add_to(m)
    
    # Add layer control to switch between different satellite providers
    folium.LayerControl(position='topright', collapsed=True).add_to(m)
    
    # Add fullscreen button for mobile users
    from folium.plugins import Fullscreen
    Fullscreen(
        position='topleft',
        title='Pantalla Completa',
        title_cancel='Salir Pantalla Completa',
        force_separate_button=True
    ).add_to(m)
    
    # Add locate control for mobile GPS
    from folium.plugins import LocateControl
    LocateControl(auto_start=False, position='topleft').add_to(m)
    
    # Enhanced NDVI colormap with better visibility on satellite imagery
    cmap = cm.LinearColormap(
        colors=["#0000FF", "#00FF00", "#FFFF00", "#FF0000"], 
        vmin=-1, vmax=1, 
        caption="NDVI (Índice de Vegetación)"
    )
    cmap.add_to(m)
    
    # Add custom legend HTML for better visibility
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 8px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                ">
    <p style="margin: 0; font-weight: bold; font-size: 13px;">Leyenda NDVI:</p>
    <p style="margin: 2px 0;"><i style="background:#FF0000;width:10px;height:10px;display:inline-block;margin-right:5px;border-radius:50%;"></i> > 0.5 (Vegetación Densa)</p>
    <p style="margin: 2px 0;"><i style="background:#FFFF00;width:10px;height:10px;display:inline-block;margin-right:5px;border-radius:50%;"></i> 0.0 - 0.5 (Vegetación Moderada)</p>
    <p style="margin: 2px 0;"><i style="background:#00FF00;width:10px;height:10px;display:inline-block;margin-right:5px;border-radius:50%;"></i> -0.5 - 0.0 (Vegetación Escasa)</p>
    <p style="margin: 2px 0;"><i style="background:#0000FF;width:10px;height:10px;display:inline-block;margin-right:5px;border-radius:50%;"></i> < -0.5 (Sin Vegetación/Agua)</p>
    </div>
    '''
    
    # Add the legend to the map using proper folium API
    folium.Element(legend_html).add_to(m)

    # Enhanced markers for better visibility on satellite imagery
    for point_idx, (idx, row) in enumerate(df.iterrows()):
        ndvi_color = cmap(row["NDVI"])
        point_num = point_idx + 1
        
        # Create larger, translucent circle with point number
        folium.CircleMarker(
            location=[row["long-ym"], row["long-xm"]],
            radius=15,  # Bigger circle for better visibility
            color='white',  # White border for contrast against satellite imagery
            weight=2,  # Border thickness
            fill=True, 
            fillColor=ndvi_color,
            fillOpacity=0.3,  # 70% translucent (0.3 = 30% opacity)
            popup=folium.Popup(
                f"<b>Punto {point_num}</b><br>"
                f"NDVI: {row['NDVI']:.3f}<br>"
                f"Lat: {row['long-ym']:.6f}<br>"
                f"Lon: {row['long-xm']:.6f}",
                max_width=200
            ),
            tooltip=f"Punto {point_num} - NDVI: {row['NDVI']:.3f}"
        ).add_to(m)
        
        # Add point number as white text inside the circle
        folium.Marker(
            location=[row["long-ym"], row["long-xm"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 30px;
                    height: 30px;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                    text-shadow: 1px 1px 2px black;
                    pointer-events: none;
                ">
                    {point_num}
                </div>
                """,
                icon_size=(30, 30),
                icon_anchor=(15, 15),
                class_name="point-number"
            )
        ).add_to(m)

    map_data = st_folium(m, width=None, height=500, returned_objects=["last_clicked"])
    
    # Add helpful information about satellite imagery usage
    with st.expander("ℹ️ Información sobre Imágenes Satelitales", expanded=False):
        st.markdown("""
        **🛰️ Consejos para usar imágenes satelitales:**
        
        - **Google Satellite**: Mejor resolución para análisis detallado de cultivos
        - **Google Hybrid**: Incluye nombres de lugares y carreteras sobre la imagen
        - **Esri Satellite**: Buena alternativa con diferentes perspectivas temporales
        - **USGS Satellite**: Imágenes oficiales, ideales para investigación científica
        
        **📱 Controles del mapa:**
        - 🔍 **Zoom**: Pellizca para hacer zoom o usa los botones +/-
        - 📍 **GPS**: Presiona el botón de ubicación para centrar en tu posición
        - 🖼️ **Pantalla completa**: Usa el botón para ver el mapa a pantalla completa
        - 🗂️ **Capas**: Cambia entre diferentes tipos de imagen usando el control de capas
        
        **🌱 Interpretación NDVI:**
        - **Rojo**: Vegetación muy densa y saludable
        - **Amarillo**: Vegetación moderada
        - **Verde**: Vegetación escasa o estresada
        - **Azul**: Suelo desnudo, agua o sin vegetación
        """)
    
    # cosmetic gap reduction
    st.markdown(
        """<style>.element-container:has(> iframe)+div{margin-top:-2rem!important;}</style>""",
        unsafe_allow_html=True
    )
    click = map_data.get("last_clicked")

    # --- point-edit panel ------------------------------------------------
    if click:
        # 1) manual DataFrame for this NDVI sheet
        manual_dict = _load_or_init_manual_file(manual_fn)
        riesgo_df = manual_dict.get(str(sheet_name))
        if riesgo_df is None:
            riesgo_df = _ensure_sheet_template(df)

        # 2) find nearest NDVI point
        lat, lon  = click["lat"], click["lng"]
        d2        = (df["long-ym"] - lat)**2 + (df["long-xm"] - lon)**2
        idx       = d2.idxmin()
        point     = df.loc[idx]
        point_num = idx + 1

        # 3) UI fields
        original_riesgo = pd.to_numeric(riesgo_df.loc[idx, "Riesgo"], errors="coerce")
        original_riesgo = int(original_riesgo) if not pd.isna(original_riesgo) else 0
        
        current_nuevo_riesgo = pd.to_numeric(riesgo_df.loc[idx, "Nuevo Riesgo"], errors="coerce")
        current_nuevo_riesgo = int(current_nuevo_riesgo) if not pd.isna(current_nuevo_riesgo) else 0

        st.markdown(f"### Punto {point_num}")
        st.write(f"**Latitud:** {point['long-ym']:.6f} • **Longitud:** {point['long-xm']:.6f}")
        st.text_input("NDVI", value=f"{point['NDVI']:.4f}", disabled=True, key=f"ndvi_display_{idx}")
        st.text_input("Riesgo Original", value=str(original_riesgo), disabled=True, key=f"riesgo_display_{idx}")
        st.text_input("Nuevo Riesgo Actual", value=str(current_nuevo_riesgo), disabled=True, key=f"nuevo_riesgo_display_{idx}")

        new_riesgo = st.selectbox(
            "Seleccionar Nuevo Riesgo", 
            list(range(7)),
            index=current_nuevo_riesgo, 
            key=f"riesgo_selector_{idx}",
            format_func=lambda x: f"Riesgo {x}"
        )

        if st.button("Actualizar", key=f"update_btn_{idx}"):
            # 4) update DataFrame & save (only Nuevo Riesgo column)
            riesgo_df.at[idx, "Nuevo Riesgo"] = new_riesgo
            _save_riesgo_sheet(str(sheet_name), riesgo_df, manual_fn)

            # Store success message in session state for persistence
            success_key = f"success_msg_{field_name}_{sheet_name}"
            st.session_state[success_key] = (
                f"✅ Nuevo Riesgo actualizado a **{new_riesgo}** para Punto **{point_num}** "
                f"en hoja **{sheet_name}**"
            )
            st.rerun()
        
        # Display persistent success message if exists
        success_key = f"success_msg_{field_name}_{sheet_name}"
        if success_key in st.session_state:
            st.success(st.session_state[success_key])

    # Always show download button
    st.markdown("---")
    if manual_fn.exists():
        with open(manual_fn, "rb") as fh:
            st.download_button(
                label=f"⬇️ Descargar {manual_fn.name}",
                data=fh,
                file_name=manual_fn.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_manual_btn",
            )
    else:
        st.info("📊 El archivo Excel estará disponible para descarga una vez inicializado o después de la primera actualización de punto.")

    if not click:
        st.info("Toca un punto en el mapa para editar su riesgo.")


# ────────────────────────── ENTRY POINT ──────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_mobile()
