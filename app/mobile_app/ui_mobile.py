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
MANUAL_FN  = ASSETS_DIR / "Data_Riesgo_MANUAL.xlsx"      # <── manual riesgo file


# ────────────────────────── HELPERS ──────────────────────────
def _load_or_init_manual_file() -> Dict[str, pd.DataFrame]:
    """
    Return {sheet_name: DataFrame}.  If the xlsx is missing *or*
    corrupted, delete it and return an empty dict.
    """
    if MANUAL_FN.exists():
        try:
            return pd.read_excel(MANUAL_FN, sheet_name=None)
        except (InvalidFileException, BadZipFile, KeyError) as exc:
            logger.warning(f"Corrupted manual file – recreating. ({exc})")
            MANUAL_FN.unlink(missing_ok=True)          # delete bad file
    return {}


def _ensure_sheet_template(df_sheet: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a 25-row template with Point / Riesgo / NDVI / Lat / Lon.
    """
    return pd.DataFrame({
        "Point":    list(range(1, 26)),
        "Riesgo":   [0] * 25,
        "NDVI":     df_sheet["NDVI"][:25].tolist(),
        "Latitud":  df_sheet["long-ym"][:25].tolist(),
        "Longitud": df_sheet["long-xm"][:25].tolist()
    })


def _save_riesgo_sheet(sheet_name: str, df_sheet: pd.DataFrame) -> None:
    """
    Safely create/update one sheet inside Data_Riesgo_MANUAL.xlsx.

    • If file missing/corrupt → create fresh with mode="w".
    • Else open in append mode and atomically replace the sheet.
    """
    MANUAL_FN.parent.mkdir(parents=True, exist_ok=True)

    # ── validate existing workbook ────────────────────────────────────
    append_ok = False
    if MANUAL_FN.exists():
        try:
            load_workbook(MANUAL_FN)
            append_ok = True
        except (InvalidFileException, BadZipFile, KeyError):
            logger.warning("⚠️  Manual file corrupt — recreating from scratch.")
            MANUAL_FN.unlink(missing_ok=True)

    mode = "a" if append_ok else "w"

    # ── build ExcelWriter kwargs --------------------------------------
    writer_kwargs = dict(
        path    = MANUAL_FN,
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
    st.set_page_config(page_title="Crop Health – Mobile", layout="wide")

    # --- viewport tweak for small screens --------------------------------
    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            @media (max-width: 768px){
                .main .block-container{padding-left:.5rem!important;padding-right:.5rem!important;}
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📱 Crop Health (Mobile)")
    st.markdown("---")

    # --- user inputs -----------------------------------------------------
    indice = st.text_input("Vegetation Index", value="NDVI")
    anio   = st.text_input("Year", value="2024")

    qgis_path = ASSETS_DIR / f"INFORME_{indice}_QGIS_{anio}.xlsx"
    st.markdown("---")

    if not qgis_path.exists():
        st.warning(f"QGIS file not found at `{qgis_path}`")
        qgis_path = st.file_uploader("Upload QGIS Excel", type=["xlsx", "xls"])
    if not qgis_path:
        st.stop()

    try:
        xls = pd.ExcelFile(qgis_path)
    except Exception as exc:
        st.error(f"Error opening Excel: {exc}")
        st.stop()

    sheet_name = st.selectbox("Select NDVI sheet", xls.sheet_names)
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
        manual_dict = _load_or_init_manual_file()
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
        current_val = pd.to_numeric(riesgo_df.loc[idx, "Riesgo"], errors="coerce")
        current_riesgo = int(current_val) if not pd.isna(current_val) else 0

        st.markdown(f"### Punto {point_num}")
        st.write(f"**Latitud:** {point['long-ym']:.6f} • **Longitud:** {point['long-xm']:.6f}")
        st.text_input("NDVI",   value=f"{point['NDVI']:.4f}",   disabled=True)
        st.text_input("Riesgo Actual", value=str(current_riesgo), disabled=True)

        new_riesgo = st.selectbox(
            "Nuevo Riesgo", options=list(range(7)),
            index=current_riesgo, key="riesgo_selector"
        )

        if st.button("Actualizar", key="update_btn"):
            # 4) update DataFrame & save
            riesgo_df.at[idx, "Riesgo"] = new_riesgo
            _save_riesgo_sheet(str(sheet_name), riesgo_df)

            st.success(
                f"✅ Riesgo actualizado a **{new_riesgo}** para Punto **{point_num}** "
                f"en hoja **{sheet_name}**"
            )

        if MANUAL_FN.exists():
            with open(MANUAL_FN, "rb") as fh:
                st.download_button(
                    label="⬇️ Descargar Data_Riesgo_MANUAL.xlsx",
                    data=fh,
                    file_name=MANUAL_FN.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_manual_btn",
                )

    else:
        st.info("Toca un punto en el mapa para editar su riesgo.")


# ────────────────────────── ENTRY POINT ──────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_mobile()
