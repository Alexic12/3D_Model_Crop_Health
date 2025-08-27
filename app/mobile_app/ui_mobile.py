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
        page_title="Crop Health – Mobile", 
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Import mobile optimizations
    from app.ui.mobile_optimizations import inject_mobile_optimizations, MobileOptimizer, PerformanceOptimizer
    
    # Apply mobile optimizations
    inject_mobile_optimizations()
    
    # Mobile-first responsive design
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            /* Mobile-first base styles */
            .main .block-container {
                padding: 0.5rem !important;
                max-width: 100% !important;
            }
            
            /* Touch-friendly map container */
            .stApp > div > div > div > div {
                padding: 0 !important;
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

    st.title("📱 Crop Health (Mobile)")
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
        field_options = available_fields + ["Create New Field"]
        selected_field = st.selectbox(
            "Select Field", 
            field_options,
            key="field_selector"
        )
        if selected_field == "Create New Field":
            field_name = st.text_input("New Field Name", value="", key="new_field_input")
        else:
            field_name = selected_field
    else:
        field_name = st.text_input("Field Name", value="Perimetro Prev", key="default_field_input")
    
    # --- user inputs -----------------------------------------------------
    indice = st.text_input("Vegetation Index", value="NDVI")
    anio   = st.text_input("Year", value="2024")

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
    
    st.markdown("---")

    if not qgis_path.exists():
        st.warning(f"QGIS file not found at `{qgis_path}`")
        qgis_path = st.file_uploader("Upload QGIS Excel", type=["xlsx", "xls"])
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
            st.info(f"📊 Initializing Excel file for {field_name}...")

    try:
        xls = pd.ExcelFile(qgis_path)
    except Exception as exc:
        st.error(f"Error opening Excel: {exc}")
        st.stop()

    sheet_name = st.selectbox(
        "Select NDVI sheet", 
        xls.sheet_names,
        key="sheet_selector"
    )
    try:
        df = pd.read_excel(qgis_path, sheet_name=sheet_name)
    except Exception as exc:
        st.error(f"Error reading sheet: {exc}")
        st.stop()

    # --- clean numeric cols ---------------------------------------------
    df["long-xm"] = pd.to_numeric(df["long-xm"], errors="coerce")
    df["long-ym"] = pd.to_numeric(df["long-ym"], errors="coerce")
    df["NDVI"]    = pd.to_numeric(df["NDVI"],    errors="coerce")
    df = df.dropna(subset=["long-xm", "long-ym", "NDVI"]).reset_index(drop=True)
    if df.empty:
        st.error("No valid coordinate / NDVI data.")
        return

    # --- folium map ------------------------------------------------------
    m = folium.Map(
        location=[df["long-ym"].mean(), df["long-xm"].mean()],
        zoom_start=17, control_scale=True
    )
    cmap = cm.LinearColormap(["blue", "green", "yellow", "red"], vmin=-1, vmax=1, caption="NDVI")
    cmap.add_to(m)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["long-ym"], row["long-xm"]],
            radius=6,
            color=cmap(row["NDVI"]),
            fill=True, fill_color=cmap(row["NDVI"]), fill_opacity=0.85,
        ).add_to(m)

    map_data = st_folium(m, width="100%", height=500, returned_objects=["last_clicked"])
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
        st.info("📊 Excel file will be available for download once initialized or after first point update.")

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
