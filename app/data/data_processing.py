import os
import zipfile
import logging
import datetime
import ctypes
import numpy as np
import pandas as pd
import rasterio
from dataclasses import dataclass
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import skew as skewfunc
import streamlit as st
import random
import seaborn as sns
import concurrent.futures  # For parallelism
import multiprocessing     # To get CPU count
from openpyxl import load_workbook, Workbook
from openpyxl.utils.exceptions import InvalidFileException
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)
n_components=5
n_var=5
titulos=['Máx grado C','Mín grado C','Viento (m/s)','Humedad (%)','Precipitaciones (mm)']

_BULK_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class BulkRasterTask:
    k_val: int
    base_path: str
    color_path: str
    sheet_name: str


def _available_memory_gb():
    """Best-effort available RAM in GB without adding a psutil dependency."""
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 ** 3)
        return None

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        return None


def _env_int(name, default=None):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
        return value if value > 0 else default
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected a positive integer.", name, raw)
        return default


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
        return value if value > 0 else default
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected a positive number.", name, raw)
        return default


def _resolve_bulk_worker_count(task_count):
    """Choose a worker count that scales up but avoids saturating weak machines."""
    if task_count <= 0:
        return 0

    explicit = _env_int("CHV_BULK_WORKERS")
    if explicit:
        return max(1, min(task_count, explicit))

    cpu_count = multiprocessing.cpu_count()
    cpu_limit = max(1, cpu_count - 1) if cpu_count > 2 else 1

    available_gb = _available_memory_gb()
    if available_gb is None:
        memory_limit = cpu_limit
    else:
        per_worker_gb = _env_float("CHV_BULK_MEMORY_PER_WORKER_GB", 1.25)
        reserved_gb = _env_float("CHV_BULK_RESERVED_MEMORY_GB", 2.0)
        usable_gb = max(0.5, available_gb - reserved_gb)
        memory_limit = max(1, int(usable_gb / max(per_worker_gb, 0.25)))

    max_workers = _env_int("CHV_BULK_MAX_WORKERS")
    limits = [task_count, cpu_limit, memory_limit]
    if max_workers:
        limits.append(max_workers)
    return max(1, min(limits))


def _prepare_bulk_child_environment():
    """Prevent BLAS/OpenMP oversubscription inside process workers."""
    for name in _BULK_THREAD_ENV_VARS:
        os.environ.setdefault(name, "1")


def _notify_progress(progress_callback, **event):
    if progress_callback is None:
        return
    try:
        progress_callback(event)
    except Exception:
        logger.exception("Bulk progress callback failed.")


def process_uploaded_file(uploaded_file):
    """
    Reads a single-sheet Excel with columns ["Longitud", "Latitud", "NDVI", "Riesgo"] (optionally).
    Returns the DataFrame or None on error.
    """
    try:
        df = pd.read_excel(uploaded_file)
        # "Riesgo" might not always be present, so let's just check the minimal columns
        required = ["Longitud", "Latitud", "NDVI"]
        if not all(c in df.columns for c in required):
            st.error("Excel must contain columns: Longitud, Latitud, NDVI (Riesgo optional).")
            return None
        ndvi_values = pd.to_numeric(df["NDVI"], errors="coerce")
        if ndvi_values.isna().any() or not ndvi_values.between(-1, 1).all():
            st.error("NDVI values must be numeric and between -1 and 1.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading single-sheet file: {e}")
        return None


def load_timeseries_data(uploaded_file):
    """
    Reads a multi-sheet Excel. Each sheet is expected to have NDVI data in a matrix format:
        Row 0, Col>0 => Longitudes
        Col 0, Row>0 => Latitudes
    Data in the interior => NDVI matrix.
    Returns dict { sheet_name: {"lon":..., "lat":..., "ndvi":...}, ...} or None on error.
    """
    try:
        excel_obj = pd.ExcelFile(uploaded_file)
        sheet_names = excel_obj.sheet_names
        data_sheets = {}
        for sname in sheet_names:
            df = excel_obj.parse(sheet_name=sname, header=None)
            if df.shape[0] < 3 or df.shape[1] < 2:
                continue
            # First row => skip the first column, read rest as lon
            lon = df.iloc[1, 1:].to_numpy(float)
            # First column => skip the first row, read rest as lat
            lat = df.iloc[2:, 0].to_numpy(float)
            # NDVI matrix => from row=2 onward, col=1 onward
            ndvi_matrix = df.iloc[2:, 1:].to_numpy(float)
            data_sheets[sname] = {
                "lon": lon,
                "lat": lat,
                "ndvi": ndvi_matrix
            }
        return data_sheets if data_sheets else None

    except Exception as e:
        st.error(f"Error loading multi-sheet time-series: {e}")
        return None


def invert_climate_file_rows(file_buffer, output_filename=None):
    """
    Loads an Excel with climate data, inverts row order, returns the DataFrame or saves to disk.
    """
    try:
        df = pd.read_excel(file_buffer)
        df_inverted = df.iloc[::-1]
        if output_filename:
            df_inverted.to_excel(output_filename, index=False)
            return None
        else:
            return df_inverted
    except Exception as e:
        logger.exception(f"Error in invert_climate_file_rows: {e}")
        return None


def rejilla_indice(ruta_imagen, ruta_color):
    """
    Reads the “base NDVI” from `ruta_imagen` (GeoTIFF) + color map from `ruta_color` (not used to compute NDVI,
    but used if you have a specific reason to confirm color-based differences).
    Returns a DataFrame with columns:
        [UTM-x, UTM-y, longitud, latitud, col, row, NDVI].
    """
    try:
        logger.info(f"[rejilla_indice] Processing base='{ruta_imagen}', color='{ruta_color}'")
        if not os.path.exists(ruta_color):
            logger.warning("[rejilla_indice] Color map file does not exist: %s", ruta_color)
            return None

        with rasterio.open(ruta_imagen) as src:
            band1 = src.read(1, masked=True)
            crs = src.crs
            logger.debug(f"[rejilla_indice] CRS={crs}, shape=({src.height}, {src.width})")
            if crs is None:
                logger.warning("[rejilla_indice] Missing CRS in %s", ruta_imagen)
                return None

            band_data = band1.filled(np.nan).astype(float)
            valid_mask = np.isfinite(band_data)
            if not valid_mask.any():
                logger.warning("[rejilla_indice] No valid pixels in %s", ruta_imagen)
                return pd.DataFrame(columns=["UTM-x", "UTM-y", "longitud", "latitud", "col", "row", "NDVI"])

            rowm, colm = np.nonzero(valid_mask)
            ndvi_values = band_data[rowm, colm].astype(float)

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xm, ym = rasterio.transform.xy(src.transform, rowm, colm, offset="center")
            xm = np.asarray(xm, dtype=float)
            ym = np.asarray(ym, dtype=float)
            lonm, latm = transformer.transform(xm, ym)

        df = pd.DataFrame({
            "UTM-x": xm,
            "UTM-y": ym,
            "longitud": lonm,
            "latitud": latm,
            "col": colm,
            "row": rowm,
            "NDVI": ndvi_values
        })
        logger.info(f"[rejilla_indice] Created DataFrame with {len(df)} points.")
        return df

    except Exception as e:
        logger.exception(f"[rejilla_indice] Error: {e}")
        return None


def _idw_index_core(df, resolution=5, k_neighbors=10):
    """
    Simple IDW interpolation core function for demonstration.  
    Returns (dfidw, dfidw_2).
    """
    try:
        logger.info(f"[IDW_Index] Starting IDW with resolution={resolution}, k_neighbors={k_neighbors}")
        xm = df["longitud"].values
        ym = df["latitud"].values
        zms = df["NDVI"].values

        x_min, x_max = xm.min(), xm.max()
        y_min, y_max = ym.min(), ym.max()

        xg = np.linspace(x_min, x_max, resolution)
        yg = np.linspace(y_min, y_max, resolution)

        zidw = np.zeros((resolution, resolution), dtype=float)

        N = resolution * resolution
        ids = np.arange(N)
        long_xm = np.zeros(N, dtype=float)
        long_ym = np.zeros(N, dtype=float)
        ndvi_vals = np.zeros(N, dtype=float)

        idx_out = 0
        for i in range(resolution):
            for j in range(resolution):
                gx = xg[i]
                gy = yg[j]
                long_xm[idx_out] = gx
                long_ym[idx_out] = gy
                dist = np.sqrt((gx - xm)**2 + (gy - ym)**2)
                k_indices = np.argsort(dist)[:k_neighbors]
                dist_nn = dist[k_indices]
                dist_nn[dist_nn == 0] = 1e-12
                w = 1.0 / dist_nn**3
                weighted_z = w * zms[k_indices]
                val = weighted_z.sum() / w.sum()
                zidw[i, j] = val
                ndvi_vals[idx_out] = val
                idx_out += 1

        df_spread = np.zeros((resolution+1, resolution+1), dtype=object)
        df_spread[0, 0] = 0
        for c in range(resolution):
            df_spread[0, c+1] = round(xg[c], 6)
        for r in range(resolution):
            df_spread[r+1, 0] = round(yg[r], 6)
        for r in range(resolution):
            for c in range(resolution):
                df_spread[r+1, c+1] = round(zidw[r, c], 6)

        dfidw = pd.DataFrame(df_spread)
        dfidw_2 = pd.DataFrame({
            "id": ids,
            "long-xm": long_xm,
            "long-ym": long_ym,
            "NDVI": ndvi_vals
        })

        logger.info(f"[IDW_Index] IDW complete. (dfidw shape={dfidw.shape}, dfidw_2 rows={len(dfidw_2)})")
        return dfidw, dfidw_2

    except Exception as e:
        logger.exception("[IDW_Index] Error:")
        return None, None


def _riesgo(df_idw_2, XC):
    """
    Dummy "cluster" approach to assign some "Riesgo" classification.
    """
    try:
        logger.info("[Riesgo] Starting cluster assignment.")
        z = df_idw_2["NDVI"].values
        n = len(z)
        nXC = np.zeros((n, 1))
        for iteration in range(2):
            XC = np.sort(XC)[::-1]
            for k in range(min(n, 25)):
                dist_sq = (XC - z[k])**2
                nearest = np.argmin(dist_sq)
                XC[nearest] = (XC[nearest] + z[k]) / 2.0
                nXC[k, 0] = nearest
        df_idw_2["Riesgo"] = (nXC[:, 0] + 1).astype(int)
        logger.info("[Riesgo] Completed risk assignment.")
        return df_idw_2, XC
    except Exception as e:
        logger.exception("[Riesgo] Error:")
        return df_idw_2, XC


def save_df_to_excel(xlsx_path, df_in, sheet_name):
    """
    Appends or writes DataFrame to an Excel file in `sheet_name`.
    Creates or re-creates the file as needed.
    """
    from openpyxl.utils.exceptions import InvalidFileException
    try:
        logger.info(f"[save_df_to_excel] Writing sheet='{sheet_name}' to '{xlsx_path}' (rows={len(df_in)})")
        with pd.ExcelWriter(xlsx_path, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
            df_in.to_excel(writer, index=False, sheet_name=sheet_name)
    except FileNotFoundError:
        logger.info(f"[save_df_to_excel] Creating new Excel file: '{xlsx_path}'")
        with pd.ExcelWriter(xlsx_path, mode='w', engine='openpyxl') as writer:
            df_in.to_excel(writer, index=False, sheet_name=sheet_name)
    except InvalidFileException:
        logger.info(f"[save_df_to_excel] InvalidFile => re-creating '{xlsx_path}'")
        with pd.ExcelWriter(xlsx_path, mode='w', engine='openpyxl') as writer:
            df_in.to_excel(writer, index=False, sheet_name=sheet_name)


def extract_date_from_filename(filename):
    """
    Tries extracting date patterns like '06ene2024' from the filename using improved regex.
    """
    import re
    # More specific pattern for Spanish month abbreviations
    patterns = [
        r'(\d{1,2}(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)\d{4})',  # Spanish months
        r'(\d{1,2}[a-zA-Z]{3}\d{4})',  # Generic 3-letter month
        r'(\d{1,2}[a-zA-Z]{3}\d{2})'   # 2-digit year fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _build_bulk_tasks(folder_path, colorMap_keyword):
    """Index ZIP/TIFF pairs once in the parent process."""
    import re

    all_files = os.listdir(folder_path)
    tiff_files = [f for f in all_files if f.lower().endswith((".tif", ".tiff"))]
    zip_files = [f for f in all_files if f.lower().endswith(".zip")]
    tasks = []
    max_k = 0

    for zip_file in sorted(zip_files):
        prefix_match = re.match(r"^(\d{3})\.", zip_file)
        if prefix_match:
            k_val = int(prefix_match.group(1))
            max_k = max(max_k, k_val)
        elif zip_file[:3].isdigit():
            k_val = int(zip_file[:3])
            max_k = max(max_k, k_val)
        else:
            logger.warning("[bulk index] ZIP has no numeric prefix => %s", zip_file)
            continue

        date_str = extract_date_from_filename(zip_file)
        if not date_str:
            logger.warning("[bulk index] ZIP has no supported date token => %s", zip_file)
            continue

        matching_tiffs = [f for f in tiff_files if date_str.lower() in f.lower()]
        color_file = next(
            (f for f in matching_tiffs if colorMap_keyword.lower() in f.lower()),
            None,
        )
        base_file = next(
            (f for f in matching_tiffs if colorMap_keyword.lower() not in f.lower()),
            None,
        )
        if not base_file or not color_file:
            logger.warning("[bulk index] k=%s missing TIFF pair for date=%s", k_val, date_str)
            continue

        sheet_name = extract_date_from_filename(base_file) or f"{str(k_val).zfill(3)}_data"
        tasks.append(BulkRasterTask(
            k_val=k_val,
            base_path=os.path.join(folder_path, base_file),
            color_path=os.path.join(folder_path, color_file),
            sheet_name=sheet_name,
        ))

    tasks.sort(key=lambda task: task.k_val)
    return tasks, max_k, all_files


def _process_one_k(task, XC_list):
    """
    Function for parallel execution: Rejilla + IDW + Riesgo for a single prefix K.
    Returns (k_val, sheet_name, df_espacial, df_idw, df_qgis).
    """
    logger.info(f"[_process_one_k] Worker starts for k={task.k_val}")
    logger.info(
        "[_process_one_k] k=%s, base='%s', color='%s', sheet='%s'",
        task.k_val,
        os.path.basename(task.base_path),
        os.path.basename(task.color_path),
        task.sheet_name,
    )

    df_esp = rejilla_indice(task.base_path, task.color_path)
    if df_esp is None or df_esp.empty:
        logger.warning(f"[_process_one_k] k={task.k_val} => No data in df_esp => skipping.")
        return (task.k_val, task.sheet_name, None, None, None)

    df_idw, df_idw_2 = _idw_index_core(df_esp)  # IDW
    if df_idw is None or df_idw_2 is None:
        logger.warning(f"[_process_one_k] k={task.k_val} => IDW failed => partial data.")
        return (task.k_val, task.sheet_name, df_esp, None, None)

    # Convert back to numpy array
    XC = np.array(XC_list)
    df_qgis, _ = _riesgo(df_idw_2, XC.copy())

    logger.info(f"[_process_one_k] k={task.k_val} => success => returning data (sheet='{task.sheet_name}')")
    return (task.k_val, task.sheet_name, df_esp, df_idw, df_qgis)


def _safe_excel_sheet_name(sheet_name):
    invalid_chars = set('[]:*?/\\')
    clean = "".join("_" if char in invalid_chars else char for char in str(sheet_name))
    return (clean or "Sheet")[:31]


def _unique_sheet_name(sheet_name, used_names):
    base = _safe_excel_sheet_name(sheet_name)
    candidate = base
    counter = 1
    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _close_bulk_writers(writers):
    for writer in writers.values():
        writer.close()


def _write_bulk_result(result, paths, writers, used_sheet_names):
    _k_val, sheet_name, df_esp, df_idw, df_qgis = result
    if not sheet_name or df_esp is None:
        return

    sheet = _unique_sheet_name(sheet_name, used_sheet_names)
    for key, df in (("espacial", df_esp), ("idw", df_idw), ("qgis", df_qgis)):
        if df is None:
            continue
        if key not in writers:
            writers[key] = pd.ExcelWriter(paths[key], mode="w", engine="openpyxl")
        df.to_excel(writers[key], index=False, sheet_name=sheet)


def bulk_unzip_and_analyze_new_parallel(
    indice, anio, base_folder="./upload_data", colorMap_keyword="ColorMap", progress_callback=None
):
    """
    Main bulk analysis pipeline:
      1) Unzip .zip pairs.
      2) Identify (base .tiff) + (ColorMap .tiff) by numeric prefix (001, 002, etc.).
      3) For each pair => Rejilla + IDW + Riesgo => generate Espacial, IDW, QGIS output.
      4) Store final Excel output files into 'assets/data/field_name/' instead of the local subfolder.
    Returns (espacial_xlsx, idw_xlsx, qgis_xlsx) or (None, None, None) on error.
    """
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Starting => indice='{indice}', anio='{anio}'")
    folder_path = os.path.join(base_folder, indice, anio)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Created folder_path='{folder_path}'")

    # 1) Unzip all .zip in folder_path
    zip_candidates = [file_ for file_ in os.listdir(folder_path) if file_.lower().endswith(".zip")]
    _notify_progress(
        progress_callback,
        stage="extract",
        completed=0,
        total=max(len(zip_candidates), 1),
        message=f"Extrayendo {len(zip_candidates)} archivo(s) ZIP...",
    )
    for idx, file_ in enumerate(zip_candidates, start=1):
        if file_.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, file_)
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Unzipping => {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder_path)
                # optional rename logic if needed
                # (commented out because your naming might differ)
                # prefix = file_[0:5]
                # nlist = zf.namelist()
            _notify_progress(
                progress_callback,
                stage="extract",
                completed=idx,
                total=max(len(zip_candidates), 1),
                message=f"ZIP extraído: {file_}",
            )

    # 2) Gather color map .tiff
    _notify_progress(
        progress_callback,
        stage="index",
        completed=0,
        total=1,
        message="Indexando pares TIFF para análisis...",
    )
    tasks, max_k, all_files = _build_bulk_tasks(folder_path, colorMap_keyword)
    _notify_progress(
        progress_callback,
        stage="index",
        completed=1,
        total=1,
        message=f"{len(tasks)} par(es) TIFF listo(s) para procesar.",
        task_total=len(tasks),
    )
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] All files in {folder_path}: {all_files}")
    
    color_files = [
        f for f in all_files
        if colorMap_keyword in f and f.lower().endswith('.tiff')
    ]
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Found {len(color_files)} color-map files: {color_files}")
    if not color_files:
        msg = f"No files containing '{colorMap_keyword}' found."
        logger.error(msg)
        st.error(msg)
        return None, None, None

    # Always detect field name from ZIP files, not folder structure
    import re
    field_name = "Unknown_Field"  # fallback
    
    # Try to detect from actual ZIP files in the folder
    if os.path.exists(folder_path):
        zip_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.zip')]
        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Detecting field from ZIP files: {zip_files[:3]}...")
        
        for file in zip_files:
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Trying to match: {file}")
            # Improved pattern for various field name formats
            patterns = [
                # Pattern 1: "001. Campo_Luna_Roja_NDVI_31ene2022.zip"
                r'\d+\.\s*([^_\s]+(?:[_\s][^_\s]+)*?)_+NDVI',
                # Pattern 2: "001. perimetro__prev_NDVI_31ene2022.zip"
                r'\d+\.\s*([^_]+(?:__[^_]+)*?)_+NDVI',
                # Pattern 3: Generic field extraction before NDVI
                r'\d+\.\s*(.+?)_NDVI'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, file, re.IGNORECASE)
                if match:
                    raw_field = match.group(1)
                    # Clean up the field name - keep underscores for folder names
                    field_name = raw_field.replace('__', '_').strip().title().replace(' ', '_')
                    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Detected field name: '{field_name}' from file: {file}")
                    break
            
            if field_name != "Unknown_Field":
                break
            else:
                logger.info(f"[bulk_unzip_and_analyze_new_parallel] No match for: {file}")
    
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Using field name: {field_name}")
    
    # We'll store final results in assets/data/field_name
    output_dir = os.path.join("assets", "data", field_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Filenames for final Excel
    espacial_xlsx = os.path.join(output_dir, f"INFORME_{indice}_Espacial_{anio}.xlsx")
    idw_xlsx      = os.path.join(output_dir, f"INFORME_{indice}_IDW_{anio}.xlsx")
    qgis_xlsx     = os.path.join(output_dir, f"INFORME_{indice}_QGIS_{anio}.xlsx")

    # Remove older versions if they exist
    for path_ in [espacial_xlsx, idw_xlsx, qgis_xlsx]:
        if os.path.exists(path_):
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Removing old {path_}")
            os.remove(path_)

    # 3) Validate indexed ZIP/TIFF pairs
    zip_files = [f for f in all_files if f.lower().endswith('.zip')]
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] ZIP files: {zip_files}")
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Computed max_k={max_k}")
    if max_k == 0:
        warn_msg = "No numeric prefixes found in ZIP files (like '001.')."
        logger.warning(warn_msg)
        st.warning(warn_msg)
        return None, None, None
    if not tasks:
        warn_msg = "No valid ZIP/TIFF pairs were found for bulk analysis."
        logger.warning(warn_msg)
        st.warning(warn_msg)
        return None, None, None

    # Shared cluster seeds
    XC = np.sort(np.random.uniform(0, 1, 5))
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Initial cluster seeds (XC)={XC}")

    # 4) Process in adaptive batches to use available local compute safely.
    max_workers = _resolve_bulk_worker_count(len(tasks))
    batch_size = max(4, max_workers * 2)
    executor_kind = os.getenv("CHV_BULK_EXECUTOR", "process").strip().lower()
    if executor_kind not in {"process", "thread", "serial"}:
        logger.warning("Invalid CHV_BULK_EXECUTOR=%r; using process.", executor_kind)
        executor_kind = "process"
    if max_workers == 1:
        executor_kind = "serial"
    if executor_kind == "process":
        _prepare_bulk_child_environment()
    logger.info(
        "[bulk_unzip_and_analyze_new_parallel] Processing %s valid files in batches of %s with %s %s worker(s)",
        len(tasks),
        batch_size,
        max_workers,
        executor_kind,
    )
    _notify_progress(
        progress_callback,
        stage="process",
        completed=0,
        total=len(tasks),
        message=f"Procesando {len(tasks)} imagen(es) con {max_workers} worker(s) ({executor_kind})...",
        workers=max_workers,
        executor=executor_kind,
    )
    
    results = []
    XC_list = XC.tolist()
    paths = {
        "espacial": espacial_xlsx,
        "idw": idw_xlsx,
        "qgis": qgis_xlsx,
    }
    writers = {}
    used_sheet_names = set()
    completed_tasks = 0
    
    # Process in batches
    try:
        for batch_start in range(0, len(tasks), batch_size):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            logger.info(
                "[bulk_unzip_and_analyze_new_parallel] Processing batch %s: files %s-%s",
                batch_start // batch_size + 1,
                batch_start + 1,
                batch_end,
            )

            if executor_kind == "serial":
                for task in batch_tasks:
                    try:
                        res = _process_one_k(task, XC_list)
                        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Completed k={task.k_val}")
                        results.append(res)
                        _write_bulk_result(res, paths, writers, used_sheet_names)
                        completed_tasks += 1
                        _notify_progress(
                            progress_callback,
                            stage="process",
                            completed=completed_tasks,
                            total=len(tasks),
                            message=f"Procesado {completed_tasks}/{len(tasks)}: {task.sheet_name}",
                            k_val=task.k_val,
                            sheet_name=task.sheet_name,
                        )
                    except Exception as e:
                        logger.exception(f"[bulk_unzip_and_analyze_new_parallel] Task for K={task.k_val} failed: {e}")
                        completed_tasks += 1
                        _notify_progress(
                            progress_callback,
                            stage="process",
                            completed=completed_tasks,
                            total=len(tasks),
                            message=f"Error en K={task.k_val}; continuando con el siguiente archivo.",
                            k_val=task.k_val,
                            error=str(e),
                        )
                continue

            executor_cls = (
                concurrent.futures.ProcessPoolExecutor
                if executor_kind == "process"
                else concurrent.futures.ThreadPoolExecutor
            )
            with executor_cls(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_process_one_k, task, XC_list): task.k_val
                    for task in batch_tasks
                }
                for future in concurrent.futures.as_completed(future_map):
                    k_val = future_map[future]
                    try:
                        res = future.result()
                        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Completed k={k_val}")
                        results.append(res)
                        _write_bulk_result(res, paths, writers, used_sheet_names)
                        completed_tasks += 1
                        _notify_progress(
                            progress_callback,
                            stage="process",
                            completed=completed_tasks,
                            total=len(tasks),
                            message=f"Procesado {completed_tasks}/{len(tasks)}: K={k_val}",
                            k_val=k_val,
                        )
                    except Exception as e:
                        logger.exception(f"[bulk_unzip_and_analyze_new_parallel] Task for K={k_val} failed: {e}")
                        completed_tasks += 1
                        _notify_progress(
                            progress_callback,
                            stage="process",
                            completed=completed_tasks,
                            total=len(tasks),
                            message=f"Error en K={k_val}; continuando con el siguiente archivo.",
                            k_val=k_val,
                            error=str(e),
                        )
    finally:
        _notify_progress(
            progress_callback,
            stage="write",
            completed=0,
            total=1,
            message="Guardando libros Excel finales...",
        )
        _close_bulk_writers(writers)
        _notify_progress(
            progress_callback,
            stage="write",
            completed=1,
            total=1,
            message="Libros Excel guardados.",
        )

    # Results already written during processing to avoid memory issues
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Completed processing {len(results)} files for field '{field_name}'")

    logger.info("[bulk_unzip_and_analyze_new_parallel] Done.")
    _notify_progress(
        progress_callback,
        stage="done",
        completed=len(tasks),
        total=len(tasks),
        message=f"Análisis masivo completado: {len(results)}/{len(tasks)} archivo(s) procesado(s).",
        successful=len(results),
    )
    return (espacial_xlsx, idw_xlsx, qgis_xlsx)


# -------------------------------------------------------------------
# 1) Fix random seeds for reproducibility
# -------------------------------------------------------------------

def Emision(i2, XDe, NIT, NR):
    """
    Matches the Jupyter 'Emision' approach: 
      - Need >= 400 data points
      - scale 0..1
      - build lags=NR => 30*N
      - MLP => skip last 360 => forecast => classify => Vp1, Vp2
    """
    try:
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from keras.regularizers import l2
        from keras.callbacks import EarlyStopping

        XDe = pd.Series(XDe).astype(float).values
        if len(XDe) < 400:
            return None, None, None, None
        rng = XDe.max() - XDe.min()
        if rng == 0:
            return None, None, None, None
        XDen = (XDe - XDe.min()) / rng

        npr = 30
        # build XDst
        XDst = np.zeros((len(XDe) - 30*(NR-1), NR))
        for k in range(NR):
            for i in range(len(XDe) - 30*(NR-1)):
                XDst[i,k] = XDen[i + k*30]

        # build y => skip last 360
        ydst = np.zeros(len(XDe) - 30*(NR-1))
        for i in range(len(XDe)-(360 + 30*(NR-1))):
            ydst[i] = XDen[i + (360 + 30*(NR-1))]

        # MLP
        model = Sequential()
        model.add(Dense(100, activation='relu', use_bias=False, input_dim=NR, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(50, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(25, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(1, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='mse', metrics=['acc'])

        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)

        XDn = XDst[:-360,:]
        ydn = ydst[:-360]
        if len(XDn) < 10:  # or some minimal threshold
            return None, None, None, None

        model.fit(XDn, ydn, epochs=NIT, batch_size=250, callbacks=[es], verbose=0)

        # forecast next 360
        XDpn = XDst[-360:,:]
        yp_scaled = model.predict(XDpn)
        yp = yp_scaled*rng + XDe.min()

        # deciles
        XC1p = np.percentile(yp, [10,20,30,40,50,60,70,80,90,100])
        # classification
        Vp1 = np.zeros((12,1))
        Vp2 = np.zeros((12,1))
        for i in range(12):
            idx = (i+1)*30 - 1
            val = yp[idx]
            d2 = np.abs(XC1p - val)
            near_idx = np.argmin(d2)
            Vp1[i] = near_idx
            # offset approach
            if near_idx < 5:
                Vp2[i] = 4 - near_idx
            else:
                Vp2[i] = near_idx - 5

        return Vp1, Vp2, yp, XC1p

    except Exception as e:
        logger.exception("Error in Emision:")
        return None, None, None, None


def MatricesTransicion(XD, XD3, n_var, punto):
    """
    LDA => 1 - NDVI  [XD.iloc[:, n_var] must be NDVI]
    KMeans(5) => XCr => transition => emission for climate columns=0..4
    random expansions for T => columns 0,1
    """
    try:
        n_components = 5

        # NDVI => last col => XD.iloc[:, n_var]
        LDA = np.array(XD.iloc[:, n_var], dtype=float)
        LDA = 1.0 - LDA

        # read lat/lon from QGIS
        lonp = XD3[0, punto, 1]
        latp = XD3[0, punto, 2]

        seed = np.random.randint(1000)
        km = KMeans(n_clusters=n_components, random_state=seed)
        km.fit(LDA.reshape(-1,1))

        centers = sorted(km.cluster_centers_.flatten(), reverse=False)
        centers = np.array(centers).reshape(1, n_components)

        XCr = np.zeros((len(LDA),1))
        for i in range(len(LDA)):
            diff_ = np.abs(centers - LDA[i])
            c_idx = np.argmin(diff_, axis=1)[0]
            XCr[i] = c_idx
            # update
            centers[0, c_idx] = (centers[0, c_idx] + LDA[i])/2.0

        # transition
        MTr = np.zeros((n_components,n_components))
        for i in range(len(XCr)-1):
            fila = int(XCr[i,0])
            col  = int(XCr[i+1,0])
            MTr[fila, col] += 1
        row_sums = MTr.sum(axis=1)
        row_sums[row_sums == 0] = 1e-9
        MTr = MTr / row_sums[:, None]

        # emission => climate columns => WD
        WD = np.zeros((len(LDA), n_var), dtype=float)
        for j in range(n_var):
            WD[:,j] = np.array(XD.iloc[:, j], dtype=float)

        # expansions for T => j=0,1
        for j in [0,1]:
            for i2 in range(len(WD)):
                WD[i2,j] = (0.9 + 0.2*random.random())*WD[i2,j]

        bEm = np.zeros((n_var, n_components, n_components))
        for j in range(n_var):
            km2 = KMeans(n_clusters=2*n_components, random_state=seed)
            km2.fit(WD[:, j].reshape(-1,1))
            c2 = sorted(km2.cluster_centers_.flatten(), reverse=False)
            c2 = np.array(c2)
            climate_class = np.zeros(len(LDA), dtype=int)
            for i2 in range(len(LDA)):
                dd = np.abs(c2 - WD[i2, j])
                idx = np.argmin(dd)
                if idx < 5:
                    climate_class[i2] = 4 - idx
                else:
                    climate_class[i2] = idx - 5

            for i2 in range(len(LDA)):
                fila = int(XCr[i2, 0])
                col  = climate_class[i2]
                bEm[j, fila, col] += 1

            # normalize
            for row_ in range(n_components):
                row_sum = bEm[j, row_, :].sum()
                if row_sum == 0:
                    row_sum = len(LDA)
                bEm[j, row_, :] /= row_sum

        return MTr, bEm, XCr, lonp, latp

    except Exception as e:
        logger.exception(f"Error in MatricesTransicion for point={punto}")
        return None, None, None, None, None


def _scalar_float(value, default=np.nan):
    """Return a Python float from scalar-like NumPy/Pandas values."""
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def Prospectiva(i1, XD, XCr, V, aTr, bEm, ydmes):
    """
    Replicates your Jupyter HPC approach:
      - LDA=1-NDVI => alpha => draw => XLDA => stats => XInf => etc.
    """
    import random
    n_components = 5
    n_var        = 5

    LDA = np.array(XD.iloc[:, n_var], dtype=float)
    LDA = 1.0 - LDA

    # cluster proportion
    nC = np.zeros((n_components,1))
    inr= np.zeros(n_components)
    for j in range(n_components):
        ccount = (XCr==j).sum()
        nC[j, 0] = ccount/len(LDA)
        inr[j] = nC[j, 0]

    # build HPC arrays
    XLDA = np.zeros((1000, V.shape[1]))
    XInf = np.zeros((V.shape[1], 12))

    # state: row=0
    XInf[0, 0] = ydmes[0,0]
    XInf[0, 1] = ydmes[0,1]
    XInf[0, 2] = ydmes[0,2]
    XInf[0, 3] = ydmes[0,3]
    XInf[0, 4] = ydmes[0,4]
    XInf[0, 5] = np.round(_scalar_float(skewfunc(LDA)),3)
    XInf[0, 6] = np.round(_scalar_float(nC[0, 0] + nC[1, 0]),3)
    XInf[0, 7] = np.round(_scalar_float(nC[2, 0] + nC[3, 0]),3)
    XInf[0, 8] = np.round(_scalar_float(nC[4, 0]),3)
    XInf[0, 9] = np.round(LDA.mean(),3)
    XInf[0,10] = np.round(np.percentile(LDA,75),3)
    XInf[0,11] = np.round(np.percentile(LDA,99),3)

    # build VC
    VC = []
    for col_m in range(V.shape[1]):
        val = V[4, col_m]
        if   val==0: VC.append(f"High {col_m+1}")
        elif val==1: VC.append(f"Average {col_m+1}")
        elif val==2: VC.append(f"Low {col_m+1}")
        elif val==3: VC.append(f"Very Low {col_m+1}")
        elif val==4: VC.append(f"Dry {col_m+1}")

    alpha = np.zeros((V.shape[1], n_components))
    # alpha[0]
    for m in range(n_var):
        alpha[0,:] += inr*bEm[m, :, V[m,0]]
    alpha[0,:] /= alpha[0,:].sum()

    # t=0 => draw
    NDm = np.int32(1000*alpha[0,:])
    m1  = -1
    for k2 in range(n_components):
        idxs = np.where(XCr==k2)[0]
        LDAm = LDA[idxs]
        mmu  = LDAm.mean()
        ssig = np.sqrt(LDAm.var())
        for i2 in range(NDm[k2]):
            m1 += 1
            # factor=2*sigma
            XLDA[m1, 0] = _scalar_float((0.8+0.4*random.random())*np.random.normal(mmu, 2*ssig))

    # fill row=1 => HPC stats
    XInf[1,0] = ydmes[0,0]
    XInf[1,1] = ydmes[0,1]
    XInf[1,2] = ydmes[0,2]
    XInf[1,3] = ydmes[0,3]
    XInf[1,4] = ydmes[0,4]
    XInf[1,5] = np.round(_scalar_float(skewfunc(XLDA[:,0])),3)
    XInf[1,6] = np.round(_scalar_float(alpha[0,0]+alpha[0,1]),3)
    XInf[1,7] = np.round(_scalar_float(alpha[0,2]+alpha[0,3]),3)
    XInf[1,8] = np.round(_scalar_float(alpha[0,4]),3)
    XInf[1,9] = np.round(XLDA[:,0].mean(),3)
    XInf[1,10]= np.round(np.percentile(XLDA[:,0],75),3)
    XInf[1,11]= np.round(np.percentile(XLDA[:,0],99),3)

    # loop t=1..11 => draw
    for t in range(1, V.shape[1]):
        alpha[t,:] = 0
        for m in range(n_var):
            alpha[t,:] += alpha[t-1,:].dot(aTr)*bEm[m, :, V[m,t]]
        alpha[t,:] /= alpha[t,:].sum()

        NDm = np.int32(1000*alpha[t,:])
        m1  = -1
        for k2 in range(n_components):
            idxs = np.where(XCr==k2)[0]
            LDAm = LDA[idxs]
            mmu  = LDAm.mean()
            ssig = np.sqrt(LDAm.var())
            for i2 in range(NDm[k2]):
                m1 += 1
                # factor=1*sigma
                XLDA[m1, t] = _scalar_float((0.9 + 0.2*random.random())*np.random.normal(mmu, ssig))

        XInf[t,0]  = ydmes[t,0]
        XInf[t,1]  = ydmes[t,1]
        XInf[t,2]  = ydmes[t,2]
        XInf[t,3]  = ydmes[t,3]
        XInf[t,4]  = ydmes[t,4]
        XInf[t,5]  = np.round(_scalar_float(skewfunc(XLDA[:,t-1])),3)
        XInf[t,6]  = np.round(_scalar_float(alpha[t,0]+alpha[t,1]),3)
        XInf[t,7]  = np.round(_scalar_float(alpha[t,2]+alpha[t,3]),3)
        XInf[t,8]  = np.round(_scalar_float(alpha[t,4]),3)
        XInf[t,9]  = np.round(XLDA[:,t-1].mean(),3)
        XInf[t,10] = np.round(np.percentile(XLDA[:,t-1],75),3)
        XInf[t,11] = np.round(np.percentile(XLDA[:,t-1],99),3)

    return np.array(VC), XInf, XLDA


def run_full_hpc_pipeline(indice: str, anio: str, base_folder: str = "./upload_data", progress_callback=None):
    """
    1) Read & invert => 'Clima_{indice}_{anio}_O.xlsx'
    2) Filter 'Fuente de datos' != '-' 
    3) Select columns => [7,8,11,12,13,4] => rename => [MaxC,MinC,Viento,Humedad,Precip,NDVI]
    4) Build V => Emision => ydmes
    5) Read QGIS => build HPC
    """
    import tensorflow as tf

    _notify_progress(
        progress_callback,
        stage="start",
        completed=0,
        total=1,
        message="Inicializando pipeline HPC...",
    )

    random.seed(123)
    np.random.seed(123)
    tf.random.set_seed(123)

    folder_path = os.path.join(base_folder, indice, anio)
    clima_xlsx  = os.path.join(folder_path, f"Clima_{indice}_{anio}.xlsx")
    
    # Track whether we're using real or mock data
    using_mock_data = False
    
    # Debug: show what files exist
    logger.info(f"Looking for climate file: {clima_xlsx}")
    if os.path.exists(folder_path):
        files = os.listdir(folder_path)
        logger.info(f"Files in {folder_path}: {files}")
    else:
        logger.error(f"Folder does not exist: {folder_path}")
        _notify_progress(
            progress_callback,
            stage="error",
            completed=0,
            total=1,
            message=f"No existe la carpeta de entrada: {folder_path}",
        )
        return None
        
    if not os.path.exists(clima_xlsx):
        logger.warning(f"Climate file not found => {clima_xlsx}. Creating mock climate data.")
        using_mock_data = True
        # Create mock climate data for testing
        
        # Generate 400+ rows of mock climate data
        n_rows = 450
        mock_data = {
            'Fecha': pd.date_range('2022-01-01', periods=n_rows, freq='D'),
            'Fuente de datos': ['Sensor'] * n_rows,
            'Latitud': np.random.uniform(4.5, 4.8, n_rows),
            'Longitud': np.random.uniform(-74.2, -73.8, n_rows),
            'NDVI': np.random.uniform(0.3, 0.9, n_rows),
            'Riesgo': np.random.randint(1, 6, n_rows),
            'Temp_Max': np.random.uniform(20, 35, n_rows),
            'Temp_Min': np.random.uniform(10, 20, n_rows),
            'Viento': np.random.uniform(0.5, 8.0, n_rows),
            'Humedad': np.random.uniform(40, 95, n_rows),
            'Precipitacion': np.random.uniform(0, 50, n_rows)
        }
        
        df_mock = pd.DataFrame(mock_data)
        os.makedirs(folder_path, exist_ok=True)
        df_mock.to_excel(clima_xlsx, index=False)
        logger.info(f"Created mock climate file: {clima_xlsx}")

    _notify_progress(
        progress_callback,
        stage="climate",
        completed=0,
        total=1,
        message="Preparando datos climáticos...",
    )

    # invert
    dfc = pd.read_excel(clima_xlsx)
    df_inverted = dfc.iloc[::-1]
    clima_inverted_path = os.path.join(folder_path, f"Clima_{indice}_{anio}_O.xlsx")
    df_inverted.to_excel(clima_inverted_path, index=False)

    # reload
    XDB  = pd.read_excel(clima_inverted_path)
    # filter
    if "Fuente de datos" in XDB.columns:
        XDB = XDB[XDB["Fuente de datos"] != "-"]
    # pick columns
    col_idx = [7,8,11,12,13,4]
    # climate
    XD  = XDB.iloc[:, col_idx].copy()
    # rename
    new_cols = ["MaxC","MinC","Viento","Humedad","Precip","NDVI"]
    XD.columns = new_cols

    # also read the same inverted for 'XD2' if your notebook does not filter it
    XDB2 = pd.read_excel(clima_inverted_path)
    XD2  = XDB2.iloc[:, col_idx].copy()
    XD2.columns = new_cols

    # drop any all-NaN
    XD  = XD.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    XD2 = XD2.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if len(XD) < 1:
        logger.error("No climate data => skip HPC.")
        _notify_progress(
            progress_callback,
            stage="error",
            completed=0,
            total=1,
            message="No hay datos climáticos válidos para el pipeline HPC.",
        )
        return None
    _notify_progress(
        progress_callback,
        stage="climate",
        completed=1,
        total=1,
        message="Datos climáticos listos.",
    )

    # 2) Build V => Emision for each var=0..4 (the climate columns). NDVI is index=5
    n_var  = 5
    V = np.zeros((n_var, 12), dtype=int)
    ydpar  = np.zeros((360, n_var), dtype=float)

    for i in range(n_var):
        _notify_progress(
            progress_callback,
            stage="emission",
            completed=i,
            total=n_var,
            message=f"Entrenando modelo de emisión {i + 1}/{n_var}...",
            variable_index=i,
        )
        # e.g. i=0 => 'MaxC', i=1 => 'MinC', ...
        XDe = XD2.iloc[:, i].values
        Vp1, Vp2, yp, _ = Emision(i, XDe, 500, 4)
        if (Vp1 is None) or (Vp2 is None) or (yp is None):
            logger.warning(f"Emision => var i={i} => not enough data => skip.")
            continue
        V[i,:]     = Vp2.flatten()
        ydpar[:,i] = yp.flatten()
        _notify_progress(
            progress_callback,
            stage="emission",
            completed=i + 1,
            total=n_var,
            message=f"Modelo de emisión {i + 1}/{n_var} completado.",
            variable_index=i,
        )

    # build ydmes => shape(12,5)
    ydmes = np.zeros((12, n_var))
    for i in range(12):
        idx = 30*(i+1)-1
        if 0<=idx<len(ydpar):
            ydmes[i,:] = ydpar[idx,:]

    # QGIS - use field-based path structure
    field_name = os.path.basename(base_folder)
    qgis_path = os.path.join("assets", "data", field_name, f"INFORME_{indice}_QGIS_{anio}.xlsx")
    if not os.path.exists(qgis_path):
        # Fallback to old structure
        qgis_path = os.path.join("assets", "data", f"INFORME_{indice}_QGIS_{anio}.xlsx")
    
    # Debug: show what files exist
    logger.info(f"Looking for QGIS file: {qgis_path}")
    assets_dir = os.path.join("assets", "data")
    if os.path.exists(assets_dir):
        if field_name and os.path.exists(os.path.join(assets_dir, field_name)):
            field_files = os.listdir(os.path.join(assets_dir, field_name))
            logger.info(f"Files in assets/data/{field_name}: {field_files}")
        else:
            all_files = os.listdir(assets_dir)
            logger.info(f"Files in assets/data: {all_files}")
    
    if not os.path.exists(qgis_path):
        logger.error(f"QGIS not found => {qgis_path}")
        _notify_progress(
            progress_callback,
            stage="error",
            completed=0,
            total=1,
            message=f"Archivo QGIS no encontrado: {qgis_path}",
        )
        return None

    _notify_progress(
        progress_callback,
        stage="qgis",
        completed=0,
        total=1,
        message="Cargando datos QGIS para simulación prospectiva...",
    )
    XDB3 = pd.read_excel(qgis_path, sheet_name=None)
    sheet_names = list(XDB3.keys())
    array_hojas = [df_.values for df_ in XDB3.values()]
    if len(array_hojas)<1:
        logger.error("No sheets in QGIS => skip HPC.")
        _notify_progress(
            progress_callback,
            stage="error",
            completed=0,
            total=1,
            message="El archivo QGIS no contiene hojas válidas.",
        )
        return None
    _notify_progress(
        progress_callback,
        stage="qgis",
        completed=1,
        total=1,
        message=f"QGIS cargado: {len(array_hojas)} hoja(s).",
    )

    # each => shape(25,5)
    n_sheets = len(array_hojas)
    total_points = 25  # 5x5
    XD3 = np.empty((n_sheets, total_points, 5), dtype=float)
    for iSheet, df_sheet in enumerate(XDB3.values()):
        numeric_sheet = df_sheet.apply(pd.to_numeric, errors="coerce")
        if numeric_sheet.shape[0] < total_points or numeric_sheet.shape[1] < 5:
            logger.error(
                "QGIS sheet %s has invalid shape %s; expected at least (%s, 5).",
                sheet_names[iSheet],
                numeric_sheet.shape,
                total_points,
            )
            _notify_progress(
                progress_callback,
                stage="error",
                completed=0,
                total=1,
                message=f"Hoja QGIS inválida: {sheet_names[iSheet]} tiene forma {numeric_sheet.shape}; se esperan 25 filas y 5 columnas.",
            )
            return None
        sheet_values = numeric_sheet.iloc[:total_points, :5].to_numpy(dtype=float)
        if not np.isfinite(sheet_values).all():
            logger.error("QGIS sheet %s contains non-numeric or missing values in the first 25x5 block.", sheet_names[iSheet])
            _notify_progress(
                progress_callback,
                stage="error",
                completed=0,
                total=1,
                message=f"Hoja QGIS inválida: {sheet_names[iSheet]} contiene valores no numéricos o vacíos.",
            )
            return None
        XD3[iSheet, :, :] = sheet_values

    # HPC => loop
    results = []
    for iPoint in range(total_points):
        _notify_progress(
            progress_callback,
            stage="points",
            completed=iPoint,
            total=total_points,
            message=f"Simulando punto {iPoint + 1}/{total_points}...",
            point_index=iPoint,
        )
        try:
            # build aTr, bEm
            aTr, bEm, XCr, lonp, latp = MatricesTransicion(XD, XD3, n_var, iPoint)
            if aTr is None or bEm is None:
                logger.warning(f"Point={iPoint}, HPC => skip.")
                _notify_progress(
                    progress_callback,
                    stage="points",
                    completed=iPoint + 1,
                    total=total_points,
                    message=f"Punto {iPoint + 1}/{total_points} omitido.",
                    point_index=iPoint,
                )
                continue

            # HPC => Prospectiva
            VC, XInf, XLDA = Prospectiva(iPoint, XD, XCr, V, aTr, bEm, ydmes)

            # in your notebook => scale from col=1.. 
            #   e.g. XLDA[:,k]*=PRef
            #   PRef => user input => * (1-0.7)
            PRef = 100.0
            eff_pref = PRef*(1-0.7)  # => 30
            for k in range(1, V.shape[1]):
                XLDA[:,k]*= eff_pref
            logger.debug("XLDA shape for point %s: %s", iPoint, XLDA.shape)
            # rewrite columns 9..11 => MPerd
            # oldv => XInf[row, col], do => eff_pref / abs(1 - oldv)
            XInf2 = XInf.copy()
            for row_i in range(V.shape[1]):
                for col_j in [9,10,11]:
                    oldv = XInf2[row_i, col_j]
                    val  = 0.0
                    if not np.isnan(oldv) and not np.isinf(oldv):
                        val = eff_pref/abs(1 - oldv)
                    XInf2[row_i, col_j] = val

            info = {
                "point_idx": iPoint,
                "VC": VC,
                "lon": lonp,
                "lat": latp,
                "XInf": XInf2,
                "XLDA": XLDA,
            }
            results.append(info)
        except Exception as exc:
            logger.exception("Point=%s HPC failed.", iPoint)
            _notify_progress(
                progress_callback,
                stage="points",
                completed=iPoint + 1,
                total=total_points,
                message=f"Punto {iPoint + 1}/{total_points} falló: {exc}",
                point_index=iPoint,
            )
            continue
        _notify_progress(
            progress_callback,
            stage="points",
            completed=iPoint + 1,
            total=total_points,
            message=f"Punto {iPoint + 1}/{total_points} completado.",
            point_index=iPoint,
        )

    hpc_data = {
        "V": V,
        "ydmes": ydmes,
        "results": results,
        "using_mock_data": using_mock_data
    }
    _notify_progress(
        progress_callback,
        stage="done",
        completed=total_points,
        total=total_points,
        message=f"Pipeline HPC completado: {len(results)}/{total_points} punto(s) procesado(s).",
        successful=len(results),
    )
    return hpc_data
