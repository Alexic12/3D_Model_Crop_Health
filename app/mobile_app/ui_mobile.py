"""
Mobile Streamlit UI – Prospective HPC view.

Replaces the previous QGIS/NDVI map editor with a 12-month prospective
visualization sourced from the desktop-generated HPC workbook:

    assets/data/{field}/INFORME_{indice}_HPC_{anio}.xlsx

For each month (1..12) the user can:
  • See all HPC points colored by month-specific risk (OpVar-99% USD).
  • Tap / click a point to inspect its monthly metrics.
  • Set a manual "Nuevo Riesgo" (0..6) for that point + month.
  • Updates persist to:
        assets/data/{field}/Data_Riesgo_Prospectivo_MANUAL_{safe_field}.xlsx
    (single sheet "Riesgos", one row per point, columns Mes_1..Mes_12).

Touch and mouse clicks both work via streamlit-folium's last_clicked.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zipfile import BadZipFile

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from folium.plugins import Fullscreen, LocateControl
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from streamlit_folium import st_folium

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ASSETS_DIR = BASE_DIR / "assets" / "data"

N_MONTHS = 12
MONTH_COLS = [f"Mes_{i + 1}" for i in range(N_MONTHS)]
RISK_LEVELS = list(range(7))  # 0..6


# ────────────────────────── HPC LOADING ──────────────────────────
def _coerce_float(val) -> Optional[float]:
    try:
        if val is None or (isinstance(val, str) and not val.strip()):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_point_sheet(df: pd.DataFrame) -> List[Dict[str, Optional[float]]]:
    """
    Parse a per-point HPC sheet written by create_complete_excel_data.
    Header row is the one whose first cell equals "Mes".
    """
    header_idx = None
    for i, val in enumerate(df.iloc[:, 0].tolist()):
        if isinstance(val, str) and val.strip().lower() == "mes":
            header_idx = i
            break
    if header_idx is None:
        return []

    headers = [str(c).strip() if c is not None else "" for c in df.iloc[header_idx].tolist()]
    body = df.iloc[header_idx + 1 : header_idx + 1 + N_MONTHS].copy()
    body.columns = headers

    months: List[Dict[str, Optional[float]]] = []
    for _, row in body.iterrows():
        months.append({
            "mean_usd": _coerce_float(row.get("Mean (USD)")),
            "p75_usd": _coerce_float(row.get("75% (USD)")),
            "opvar99_usd": _coerce_float(row.get("OpVar-99% (USD)")),
            "pc1": _coerce_float(row.get("%C1")),
            "pc2": _coerce_float(row.get("%C2")),
            "pc3": _coerce_float(row.get("%C3")),
        })
    while len(months) < N_MONTHS:
        months.append({k: None for k in ("mean_usd", "p75_usd", "opvar99_usd", "pc1", "pc2", "pc3")})
    return months


def _load_hpc_workbook(hpc_path: Path) -> Optional[Dict]:
    try:
        sheets = pd.read_excel(hpc_path, sheet_name=None, header=None)
    except (InvalidFileException, BadZipFile, ValueError, FileNotFoundError) as exc:
        logger.warning(f"Cannot read HPC workbook: {exc}")
        return None

    resumen_key = next((k for k in sheets if "Resumen" in k), None)
    if resumen_key is None:
        return None
    resumen = sheets[resumen_key]

    indice = None
    for i in range(min(6, len(resumen))):
        cell0 = str(resumen.iloc[i, 0]) if pd.notna(resumen.iloc[i, 0]) else ""
        if "Índice" in cell0 and resumen.shape[1] > 1:
            indice = str(resumen.iloc[i, 1]).strip()
            break

    header_idx = None
    for i, val in enumerate(resumen.iloc[:, 0].tolist()):
        if isinstance(val, str) and val.strip().lower() == "punto":
            header_idx = i
            break
    if header_idx is None:
        return None

    points: List[Dict] = []
    for _, row in resumen.iloc[header_idx + 1 :].iterrows():
        label = row.iloc[0]
        if not isinstance(label, str):
            continue
        m = re.search(r"(\d+)", label)
        if not m:
            continue
        point_num = int(m.group(1))
        lat = _coerce_float(row.iloc[1] if resumen.shape[1] > 1 else None)
        lon = _coerce_float(row.iloc[2] if resumen.shape[1] > 2 else None)

        point_sheet_key = next(
            (k for k in sheets if k.endswith(f"Punto {point_num}") or k == f"📍 Punto {point_num}"),
            None,
        )
        months: List[Dict[str, Optional[float]]] = []
        if point_sheet_key is not None:
            months = _parse_point_sheet(sheets[point_sheet_key])
        available = bool(months) and any(m.get("opvar99_usd") is not None for m in months)
        points.append({
            "point_num": point_num,
            "lat": lat,
            "lon": lon,
            "available": available,
            "months": months,
        })

    points.sort(key=lambda p: p["point_num"])
    return {"points": points, "indice": indice}


# ────────────────────────── MANUAL RISK FILE ──────────────────────────
def _manual_template(points: List[Dict]) -> pd.DataFrame:
    rows = []
    for p in points:
        row = {"Punto": p["point_num"], "Latitud": p["lat"], "Longitud": p["lon"]}
        for col in MONTH_COLS:
            row[col] = 0
        rows.append(row)
    return pd.DataFrame(rows, columns=["Punto", "Latitud", "Longitud", *MONTH_COLS])


def _load_manual_risks(manual_fn: Path, points: List[Dict]) -> pd.DataFrame:
    if manual_fn.exists():
        try:
            df = pd.read_excel(manual_fn, sheet_name="Riesgos")
            for col in MONTH_COLS:
                if col not in df.columns:
                    df[col] = 0
            if "Punto" not in df.columns:
                raise KeyError("Punto column missing")
            existing = set(df["Punto"].astype(int).tolist())
            for p in points:
                if p["point_num"] not in existing:
                    new_row = {"Punto": p["point_num"], "Latitud": p["lat"], "Longitud": p["lon"]}
                    for col in MONTH_COLS:
                        new_row[col] = 0
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            return df.sort_values("Punto").reset_index(drop=True)
        except (InvalidFileException, BadZipFile, KeyError, ValueError) as exc:
            logger.warning(f"Manual prospective file corrupt – recreating ({exc}).")
            manual_fn.unlink(missing_ok=True)
    return _manual_template(points)


def _save_manual_risks(manual_fn: Path, df: pd.DataFrame) -> None:
    manual_fn.parent.mkdir(parents=True, exist_ok=True)
    append_ok = False
    if manual_fn.exists():
        try:
            load_workbook(manual_fn)
            append_ok = True
        except (InvalidFileException, BadZipFile, KeyError):
            manual_fn.unlink(missing_ok=True)
    mode = "a" if append_ok else "w"
    writer_kwargs: Dict = dict(path=manual_fn, engine="openpyxl", mode=mode)
    if mode == "a":
        writer_kwargs["if_sheet_exists"] = "replace"
    with pd.ExcelWriter(**writer_kwargs) as writer:
        df.to_excel(writer, sheet_name="Riesgos", index=False)


# ────────────────────────── MAP HELPERS ──────────────────────────
SATELLITE_CONFIGS = {
    "Google Satellite": {
        "tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "attr": "Google Satellite",
        "name": "Google Satellite",
    },
    "Google Hybrid": {
        "tiles": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        "attr": "Google Hybrid (Satellite + Labels)",
        "name": "Google Hybrid",
    },
    "Esri Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "name": "Esri Satellite",
    },
    "Street Map": {
        "tiles": "OpenStreetMap",
        "attr": "OpenStreetMap",
        "name": "Street Map",
    },
}


def _month_values(points: List[Dict], month_idx: int, key: str) -> List[Optional[float]]:
    return [p["months"][month_idx].get(key) if p["months"] else None for p in points]


def _build_map(
    points: List[Dict],
    manual_df: pd.DataFrame,
    month_idx: int,
    satellite_option: str,
) -> folium.Map:
    valid_pts = [p for p in points if p["lat"] is not None and p["lon"] is not None]
    if not valid_pts:
        center = [0.0, 0.0]
    else:
        center = [
            sum(p["lat"] for p in valid_pts) / len(valid_pts),
            sum(p["lon"] for p in valid_pts) / len(valid_pts),
        ]

    cfg = SATELLITE_CONFIGS[satellite_option]
    fmap = folium.Map(
        location=center, zoom_start=17, control_scale=True,
        tiles=cfg["tiles"], attr=cfg["attr"],
    )
    for name, c in SATELLITE_CONFIGS.items():
        if name != satellite_option:
            folium.TileLayer(tiles=c["tiles"], attr=c["attr"], name=c["name"],
                             overlay=False, control=True).add_to(fmap)
    folium.LayerControl(position="topright", collapsed=True).add_to(fmap)
    Fullscreen(position="topleft", title="Pantalla Completa",
               title_cancel="Salir", force_separate_button=True).add_to(fmap)
    LocateControl(auto_start=False, position="topleft").add_to(fmap)

    opv = [v for v in _month_values(points, month_idx, "opvar99_usd") if v is not None]
    if opv:
        vmin, vmax = min(opv), max(opv)
        if vmax <= vmin:
            vmax = vmin + 1.0
        cmap = cm.LinearColormap(
            colors=["#2ca02c", "#ffff00", "#ff7f0e", "#d62728"],
            vmin=vmin, vmax=vmax,
            caption=f"OpVar-99% USD – Mes {month_idx + 1}",
        )
        cmap.add_to(fmap)
    else:
        cmap = None

    manual_lookup = manual_df.set_index("Punto")[MONTH_COLS[month_idx]].to_dict()

    for p in points:
        if p["lat"] is None or p["lon"] is None:
            continue
        month_data = p["months"][month_idx] if p["months"] else {}
        opv99 = month_data.get("opvar99_usd")
        nuevo = manual_lookup.get(p["point_num"], 0)
        try:
            nuevo_int = int(nuevo) if pd.notna(nuevo) else 0
        except (TypeError, ValueError):
            nuevo_int = 0

        if not p["available"] or opv99 is None:
            fill = "#888888"
            fill_opacity = 0.25
            tooltip = f"Punto {p['point_num']} – sin datos"
        else:
            fill = cmap(opv99) if cmap is not None else "#1f77b4"
            fill_opacity = 0.55
            tooltip = f"Punto {p['point_num']}"

        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=15, color="white", weight=2,
            fill=True, fillColor=fill, fillOpacity=fill_opacity,
            tooltip=tooltip,
        ).add_to(fmap)

        folium.Marker(
            location=[p["lat"], p["lon"]],
            icon=folium.DivIcon(
                html=(
                    f"<div style='display:flex;align-items:center;justify-content:center;"
                    f"width:30px;height:30px;color:white;font-weight:bold;font-size:12px;"
                    f"text-shadow:1px 1px 2px black;pointer-events:none;'>"
                    f"{p['point_num']}</div>"
                ),
                icon_size=(30, 30), icon_anchor=(15, 15),
                class_name="point-number",
            ),
            interactive=False,
        ).add_to(fmap)
    return fmap


def _nearest_point(points: List[Dict], lat: float, lon: float) -> Optional[Dict]:
    best: Tuple[float, Optional[Dict]] = (float("inf"), None)
    for p in points:
        if p["lat"] is None or p["lon"] is None:
            continue
        d2 = (p["lat"] - lat) ** 2 + (p["lon"] - lon) ** 2
        if d2 < best[0]:
            best = (d2, p)
    return best[1]


def _extract_point_num_from_event(map_data: Dict, points: List[Dict]) -> Optional[int]:
    """Resolve the selected point from the most reliable folium click event.

    Prefers fresh click coordinates over tooltip/popup text because streamlit-folium
    can return stale tooltip strings across reruns.
    """
    clicked_obj = map_data.get("last_object_clicked")
    if isinstance(clicked_obj, dict):
        lat = clicked_obj.get("lat")
        lon = clicked_obj.get("lng")
        if lat is not None and lon is not None:
            nearest = _nearest_point(points, lat, lon)
            if nearest is not None:
                return nearest["point_num"]

    clicked = map_data.get("last_clicked")
    if isinstance(clicked, dict):
        lat = clicked.get("lat")
        lon = clicked.get("lng")
        if lat is not None and lon is not None:
            nearest = _nearest_point(points, lat, lon)
            if nearest is not None:
                return nearest["point_num"]

    return None


# ──────────────────────── MAIN MOBILE APP ───────────────────────
def render_mobile() -> None:
    st.set_page_config(
        page_title="Salud de Cultivos – Móvil",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <style>
            .main .block-container { padding: 0.5rem !important; max-width: 100% !important; }
            .stApp > div > div > div > div { padding: 0 !important; }
            @media (max-width: 480px) {
                .main .block-container { padding: 0.25rem !important; }
                h1 { font-size: 1.5rem !important; }
                h2, h3 { font-size: 1.25rem !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📱 Salud de Cultivos – Visualización Prospectiva (Móvil)")
    st.markdown("---")

    import os
    base_folder = "./upload_data"
    available_fields = [
        item for item in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, item))
    ] if os.path.exists(base_folder) else []

    if available_fields:
        field_options = available_fields + ["Crear Nuevo Campo"]
        selected_field = st.selectbox("Seleccionar Campo", field_options, key="field_selector")
        if selected_field == "Crear Nuevo Campo":
            field_name = st.text_input("Nombre del Nuevo Campo", value="", key="new_field_input")
        else:
            field_name = selected_field
    else:
        field_name = st.text_input("Nombre del Campo", value="Perimetro Prev", key="default_field_input")

    if not field_name:
        st.info("Selecciona o ingresa un nombre de campo para continuar.")
        st.stop()

    indice = st.text_input("Índice de Vegetación", value="NDVI")
    anio = st.text_input("Año", value="2024")
    satellite_option = st.selectbox(
        "Tipo de Imagen Satelital",
        list(SATELLITE_CONFIGS.keys()),
        index=0, key="satellite_selector",
    )

    safe_field_name = field_name.replace(" ", "_").replace("/", "_")
    hpc_path = ASSETS_DIR / field_name / f"INFORME_{indice}_HPC_{anio}.xlsx"
    manual_fn = ASSETS_DIR / field_name / f"Data_Riesgo_Prospectivo_MANUAL_{safe_field_name}.xlsx"

    st.markdown("---")

    if not hpc_path.exists():
        st.warning(
            "⚠️ **No hay datos prospectivos aún.**\n\n"
            f"Ejecuta el pipeline HPC desde la versión de escritorio para el "
            f"campo **{field_name}**, índice **{indice}**, año **{anio}**. "
            "Una vez finalizado, el archivo se guardará automáticamente en:\n\n"
            f"`{hpc_path}`"
        )
        st.stop()

    hpc = _load_hpc_workbook(hpc_path)
    if hpc is None or not hpc.get("points"):
        st.error(f"No se pudo leer el archivo HPC: `{hpc_path}`")
        st.stop()
    points = hpc["points"]

    month_label = st.selectbox(
        "Seleccionar Mes",
        [f"Mes {i + 1}" for i in range(N_MONTHS)],
        key="month_selector",
    )
    month_idx = int(month_label.split()[1]) - 1

    manual_df = _load_manual_risks(manual_fn, points)
    selection_state_key = f"selected_point_{field_name}_{indice}_{anio}"
    if selection_state_key not in st.session_state:
        st.session_state[selection_state_key] = None

    fmap = _build_map(points, manual_df, month_idx, satellite_option)
    map_data = st_folium(
        fmap, width=None, height=520,
        returned_objects=["last_clicked", "last_object_clicked"],
        key=f"map_{field_name}_{indice}_{anio}_{month_idx}",
    )

    st.caption(
        "Toca o haz clic en un punto para ver sus métricas y editar el riesgo "
        "para el mes seleccionado. Los puntos en gris no tienen datos prospectivos."
    )

    if map_data:
        selected_point_num = _extract_point_num_from_event(map_data, points)
        if selected_point_num is not None:
            st.session_state[selection_state_key] = selected_point_num

    selected_point_num = st.session_state.get(selection_state_key)
    selected_point = next((p for p in points if p["point_num"] == selected_point_num), None)

    if selected_point is not None:
        nearest = selected_point
        point_num = nearest["point_num"]
        month_data = nearest["months"][month_idx] if nearest["months"] else {}
        month_col = MONTH_COLS[month_idx]
        current_values = manual_df[manual_df["Punto"] == point_num][month_col].tolist()
        current_nuevo_int = 0
        if current_values:
            raw = current_values[0]
            try:
                if pd.notna(raw):
                    current_nuevo_int = int(raw)
            except (TypeError, ValueError):
                current_nuevo_int = 0

        st.markdown(f"### Punto {point_num} – {month_label}")
        st.write(
            f"**Latitud:** {nearest['lat']:.6f} • **Longitud:** {nearest['lon']:.6f}"
        )

        if not nearest["available"] or month_data.get("opvar99_usd") is None:
            st.info("Sin datos prospectivos HPC para este punto/mes.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Mean (USD)",
                f"{month_data['mean_usd']:,.2f}" if month_data.get("mean_usd") is not None else "—",
            )
            col2.metric(
                "75% (USD)",
                f"{month_data['p75_usd']:,.2f}" if month_data.get("p75_usd") is not None else "—",
            )
            col3.metric("OpVar-99% (USD)", f"{month_data['opvar99_usd']:,.2f}")
            col4, col5, col6 = st.columns(3)
            col4.metric("%C1", f"{month_data['pc1']:.3f}" if month_data.get("pc1") is not None else "—")
            col5.metric("%C2", f"{month_data['pc2']:.3f}" if month_data.get("pc2") is not None else "—")
            col6.metric("%C3", f"{month_data['pc3']:.3f}" if month_data.get("pc3") is not None else "—")

        st.markdown(f"**Riesgo guardado actual:** {current_nuevo_int}")
        new_riesgo = st.selectbox(
            "Seleccionar Nuevo Riesgo",
            RISK_LEVELS,
            index=current_nuevo_int if current_nuevo_int in RISK_LEVELS else 0,
            key=f"riesgo_sel_{point_num}_{month_idx}",
            format_func=lambda x: f"Riesgo {x}",
        )

        if st.button("Actualizar", key=f"update_btn_{point_num}_{month_idx}"):
            manual_df.loc[manual_df["Punto"] == point_num, MONTH_COLS[month_idx]] = new_riesgo
            manual_df.loc[manual_df["Punto"] == point_num, "Latitud"] = nearest["lat"]
            manual_df.loc[manual_df["Punto"] == point_num, "Longitud"] = nearest["lon"]
            try:
                _save_manual_risks(manual_fn, manual_df)
                st.session_state[f"prosp_msg_{field_name}"] = (
                    f"✅ Nuevo Riesgo **{new_riesgo}** guardado para Punto **{point_num}** "
                    f"en **{month_label}**."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Error al guardar: {exc}")
                logger.exception("save manual risks failed")
    else:
        st.info("Toca o haz clic en un punto del mapa para editar su riesgo prospectivo.")

    msg_key = f"prosp_msg_{field_name}"
    if msg_key in st.session_state:
        st.success(st.session_state[msg_key])

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
        st.info("📊 El archivo de riesgos manuales se creará al actualizar el primer punto.")


# ────────────────────────── ENTRY POINT ──────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    render_mobile()
