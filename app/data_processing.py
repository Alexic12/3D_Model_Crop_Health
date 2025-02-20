import os
import zipfile
import datetime
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import logging
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import streamlit as st

import concurrent.futures  # For parallelism
import multiprocessing     # To get CPU count

logger = logging.getLogger(__name__)


def process_uploaded_file(uploaded_file):
    # [unchanged from before...]
    try:
        df = pd.read_excel(uploaded_file)
        required = ["Longitud", "Latitud", "NDVI", "Riesgo"]
        if not all(c in df.columns for c in required):
            st.error("Excel must contain columns: Longitud, Latitud, NDVI, Riesgo")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading single-sheet file: {e}")
        return None


def load_timeseries_data(uploaded_file):
    # [unchanged from before...]
    try:
        excel_obj = pd.ExcelFile(uploaded_file)
        sheet_names = excel_obj.sheet_names
        data_sheets = {}
        for sname in sheet_names:
            df = pd.read_excel(uploaded_file, sheet_name=sname, header=None)
            if df.shape[0] < 3 or df.shape[1] < 2:
                continue
            lon = df.iloc[1, 1:].to_numpy(float)
            lat = df.iloc[2:, 0].to_numpy(float)
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
    # [unchanged from before...]
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
    Reads the “base NDVI” from `ruta_imagen` (GeoTIFF) + color map from `ruta_color`.
    Returns a DataFrame [UTM-x, UTM-y, longitud, latitud, col, row, NDVI].
    Missing or -9999 NDVI become NaN and are skipped.
    """
    from PIL import Image
    try:
        logger.info(f"[rejilla_indice] Processing base='{ruta_imagen}', color='{ruta_color}'")
        color_img = Image.open(ruta_color)
        pixels = color_img.load()

        xm = []
        ym = []
        latm = []
        lonm = []
        colm = []
        rowm = []
        NDVI = []

        with rasterio.open(ruta_imagen) as src:
            band1 = src.read(1)
            crs = src.crs
            logger.debug(f"[rejilla_indice] CRS={crs}, shape=({src.height}, {src.width})")
            no_data_value = src.nodata if src.nodata is not None else -9999
            band1 = np.where(band1 == no_data_value, np.nan, band1)

            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

            for r_i in range(src.height):
                for c_i in range(src.width):
                    ndvi_val = band1[r_i, c_i]
                    if np.isnan(ndvi_val):
                        continue
                    x, y = src.xy(r_i, c_i)
                    lon, lat = transformer.transform(x, y)
                    xm.append(x)
                    ym.append(y)
                    latm.append(lat)
                    lonm.append(lon)
                    colm.append(c_i)
                    rowm.append(r_i)
                    NDVI.append(float(ndvi_val))

        df = pd.DataFrame({
            "UTM-x": xm,
            "UTM-y": ym,
            "longitud": lonm,
            "latitud": latm,
            "col": colm,
            "row": rowm,
            "NDVI": NDVI
        })
        logger.info(f"[rejilla_indice] Created DataFrame with {len(df)} points.")
        return df

    except Exception as e:
        logger.exception(f"[rejilla_indice] Error: {e}")
        return None

def IDW_Index(df, resolution=5, k_neighbors=10):
    """
    Simple IDW interpolation. Returns (dfidw, dfidw_2).
    dfidw => 2D "spreadsheet" style
    dfidw_2 => long-form [id, long-xm, long-ym, NDVI]
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


def Riesgo(df_idw_2, XC):
    """
    Classifies NDVI in df_idw_2 with a minimal clustering approach.
    Returns (df_idw_2_with_Riesgo, updated_XC).
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
    Try extracting something like '06ene2024' from the filename via regex.
    Adjust if your naming differs.
    """
    import re
    pattern = re.compile(r'(\d{1,2}[a-zA-Z]{3}\d{2,4})')
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None

def IDW_Index(df):
    xm = np.array(df['longitud'])
    ym = np.array(df['latitud'])
    zms = np.array(df['NDVI'])

    xm_min = xm.min(); ym_min = ym.min()
    xm_max = xm.max(); ym_max = ym.max()

    res = 5
    xg = np.linspace(xm_min, xm_max, res)
    yg = np.linspace(ym_min, ym_max, res)

    zidw = np.zeros((res, res))
    zidwd = np.zeros((res, res))

    nKNN = 10
    ds = np.zeros((nKNN, 1))
    dknn = np.zeros((nKNN, 1))
    dknnd = np.zeros((nKNN, 1))

    idm = np.zeros((res*res, 1))
    xm_2 = np.zeros((res*res, 1))
    ym_2 = np.zeros((res*res, 1))
    zms_2 = np.zeros((res*res, 1))

    m4 = -1
    for i in range(res):
        for j in range(res):
            m4 += 1
            idm[m4] = m4
            xm_2[m4] = xg[i]
            ym_2[m4] = yg[j]

            dx = (xg[i] - xm)**2
            dy = (yg[j] - ym)**2
            d = np.sqrt(dx+dy)

            # sort distances
            sorted_d = np.sort(d)
            ds = sorted_d[0:nKNN]

            for k in range(nKNN):
                idx_d = np.where(d == ds[k])[0][0]
                dknn[k, 0] = idx_d
                dknnd[k, 0] = np.sqrt((xg[i] - xm[idx_d])**2 + (yg[j] - ym[idx_d])**2)
                zidwd[i, j] += (1 / dknnd[k, 0]**3)
                zidw[i, j]  += (1 / dknnd[k, 0]**3) * zms[idx_d]

            zidw[i, j] = zidw[i, j] / zidwd[i, j]
            zms_2[m4, ] = zidw[i, j]

    # build the df
    dfidw_2 = pd.DataFrame(np.column_stack((idm, xm_2, ym_2, zms_2)),
                           columns=['id', 'long-xm', 'long-ym', 'NDVI'])

    # also build the 2D array with x, y in first row/col
    zidws = np.zeros((res+1, res+1))
    zidws[0, 1:res+1] = xg.flatten()
    zidws[1:res+1, 0] = yg.flatten()
    zidws[1:res+1, 1:res+1] = zidw

    dfidw = pd.DataFrame(zidws)

    return dfidw, dfidw_2

#------------------ Parallel Code with Debug Prints ------------------ #

def _process_one_k(k_val, folder_path, colorMap_keyword, XC):
    """
    Top-level function for parallel execution.
    Runs 'rejilla_indice + IDW_Index + Riesgo' for a single K = 001, 002, etc.
    Returns (k_val, sheet_name, df_espacial, df_idw, df_qgis).
    """
    from data_processing import (
        extract_date_from_filename,
        rejilla_indice,
        IDW_Index,
        Riesgo
    )

    logger.info(f"[_process_one_k] Worker starts for k={k_val}")
    k_str = str(k_val).zfill(3)
    base_file, color_file = None, None

    # 1) Identify the TIFF files for the given prefix k_str
    for fname in os.listdir(folder_path):
        if fname.startswith(k_str) and fname.lower().endswith('.tiff'):
            if colorMap_keyword in fname:
                color_file = fname
            else:
                base_file = fname

    if not base_file or not color_file:
        logger.warning(f"[_process_one_k] k={k_val} => No valid pair => skipping.")
        return (k_val, None, None, None, None)

    # 2) Build a sheet name from date in the filename (or fallback)
    sheet_name = extract_date_from_filename(base_file)
    if not sheet_name:
        sheet_name = f"{k_str}_data"

    base_path  = os.path.join(folder_path, base_file)
    color_path = os.path.join(folder_path, color_file)
    logger.info(f"[_process_one_k] k={k_val}, base='{base_file}', color='{color_file}', sheet='{sheet_name}'")

    df_esp = rejilla_indice(base_path, color_path)
    if df_esp is None or df_esp.empty:
        logger.warning(f"[_process_one_k] k={k_val} => No data in df_esp => skipping.")
        return (k_val, sheet_name, None, None, None)

    df_idw, df_idw_2 = IDW_Index(df_esp)
    if df_idw is None or df_idw_2 is None:
        logger.warning(f"[_process_one_k] k={k_val} => IDW failed => partial data.")
        return (k_val, sheet_name, df_esp, None, None)

    # Use a copy of XC so each image doesn't mutate a shared global
    df_qgis, _ = Riesgo(df_idw_2, XC.copy())

    logger.info(f"[_process_one_k] k={k_val} => success => returning data (sheet='{sheet_name}')")
    return (k_val, sheet_name, df_esp, df_idw, df_qgis)


def bulk_unzip_and_analyze_new_parallel(
    indice, anio, base_folder="./upload_data", colorMap_keyword="ColorMap"
):
    """
    Parallel version of the bulk analysis. Because Windows requires
    top-level pickleable callables, _process_one_k is now defined above.
    """
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Starting => indice='{indice}', anio='{anio}'")
    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Created folder_path='{folder_path}'")

    # 1) Unzip all .zip
    for file_ in os.listdir(folder_path):
        if file_.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, file_)
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Unzipping => {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder_path)
                prefix = file_[0:5]  # e.g. "001."
                nlist = zf.namelist()
                if len(nlist) >= 2:
                    old1 = os.path.join(folder_path, nlist[0])
                    old2 = os.path.join(folder_path, nlist[1])
                    new1 = os.path.join(folder_path, prefix + nlist[0])
                    new2 = os.path.join(folder_path, prefix + nlist[1])
                    logger.debug(f"[bulk_unzip_and_analyze_new_parallel] rename => {old1} => {new1}")
                    logger.debug(f"[bulk_unzip_and_analyze_new_parallel] rename => {old2} => {new2}")
                    if os.path.exists(new1):
                        try: os.remove(old1)
                        except: pass
                    else:
                        os.rename(old1, new1)
                    if os.path.exists(new2):
                        try: os.remove(old2)
                        except: pass
                    else:
                        os.rename(old2, new2)

    # 2) Gather all colorMap .tiff
    color_files = [
        f for f in os.listdir(folder_path)
        if colorMap_keyword in f and f.lower().endswith('.tiff')
    ]
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Found {len(color_files)} color-map files in {folder_path}")
    if not color_files:
        msg = f"No files containing '{colorMap_keyword}' found."
        logger.error(msg)
        st.error(msg)
        return None, None, None

    # Final Excel filenames
    espacial_xlsx = os.path.join(folder_path, f"INFORME_{indice}_Espacial_{anio}.xlsx")
    idw_xlsx      = os.path.join(folder_path, f"INFORME_{indice}_IDW_{anio}.xlsx")
    qgis_xlsx     = os.path.join(folder_path, f"INFORME_{indice}_QGIS_{anio}.xlsx")

    # Optionally remove older versions
    for path_ in [espacial_xlsx, idw_xlsx, qgis_xlsx]:
        if os.path.exists(path_):
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Removing old {path_}")
            os.remove(path_)

    # 3) Identify max k => "001", "002", ...
    max_k = 0
    for f in color_files:
        try:
            k_ = int(f[0:3])
            if k_ > max_k: max_k = k_
        except:
            pass
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Computed max_k={max_k}")
    if max_k == 0:
        warn_msg = "No numeric prefixes found in colorMap files (like '001')."
        logger.warning(warn_msg)
        st.warning(warn_msg)
        return None, None, None

    # We'll do a shared cluster seed
    XC = np.sort(np.random.uniform(0,1,5))
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Initial cluster seeds (XC)={XC}")

    # 4) Create tasks for k in [1..max_k]
    tasks = range(1, max_k+1)

    # 5) Launch parallel
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Using up to {cpu_count} parallel processes.")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
        future_map = {
            executor.submit(_process_one_k, k_val, folder_path, colorMap_keyword, XC): k_val
            for k_val in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            k_val = future_map[future]
            try:
                res = future.result()
                logger.info(f"[bulk_unzip_and_analyze_new_parallel] Completed k={k_val}")
                results.append(res)
            except Exception as e:
                logger.exception(f"[bulk_unzip_and_analyze_new_parallel] Task for K={k_val} failed: {e}")

    # Sort by k
    results.sort(key=lambda r: r[0])

    # 6) Write to Excel in main thread
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Writing results to Excel => {espacial_xlsx}, {idw_xlsx}, {qgis_xlsx}")
    for (k_val, sheet_name, df_esp, df_idw, df_qgis) in results:
        if sheet_name is None or df_esp is None:
            continue
        save_df_to_excel(espacial_xlsx, df_esp, sheet_name)
        if df_idw is not None:
            save_df_to_excel(idw_xlsx, df_idw, sheet_name)
        if df_qgis is not None:
            save_df_to_excel(qgis_xlsx, df_qgis, sheet_name)

    logger.info("[bulk_unzip_and_analyze_new_parallel] Done.")
    return (espacial_xlsx, idw_xlsx, qgis_xlsx)