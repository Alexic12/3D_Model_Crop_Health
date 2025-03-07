import os
import zipfile
import logging
import datetime
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import skew as skewfunc
import streamlit as st
import random
import concurrent.futures  # For parallelism
import multiprocessing     # To get CPU count
from openpyxl import load_workbook, Workbook
from openpyxl.utils.exceptions import InvalidFileException
from sklearn.cluster import KMeans
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)
n_components=5
n_var=5
titulos=['Máx grado C','Mín grado C','Viento (m/s)','Humedad (%)','Precipitaciones (mm)']


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
            df = pd.read_excel(uploaded_file, sheet_name=sname, header=None)
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
    Tries extracting something like '06ene2024' from the filename using a regex.
    """
    import re
    pattern = re.compile(r'(\d{1,2}[a-zA-Z]{3}\d{2,4})')
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None


def _process_one_k(k_val, folder_path, colorMap_keyword, XC):
    """
    Function for parallel execution: Rejilla + IDW + Riesgo for a single prefix K.
    Returns (k_val, sheet_name, df_espacial, df_idw, df_qgis).
    """
    logger.info(f"[_process_one_k] Worker starts for k={k_val}")
    k_str = str(k_val).zfill(3)
    base_file, color_file = None, None

    # Find the TIFF pair for this k_val
    for fname in os.listdir(folder_path):
        if fname.startswith(k_str) and fname.lower().endswith('.tiff'):
            if colorMap_keyword in fname:
                color_file = fname
            else:
                base_file = fname

    if not base_file or not color_file:
        logger.warning(f"[_process_one_k] k={k_val} => No valid pair => skipping.")
        return (k_val, None, None, None, None)

    sheet_name = extract_date_from_filename(base_file)
    if not sheet_name:
        sheet_name = f"{k_str}_data"

    base_path = os.path.join(folder_path, base_file)
    color_path = os.path.join(folder_path, color_file)
    logger.info(f"[_process_one_k] k={k_val}, base='{base_file}', color='{color_file}', sheet='{sheet_name}'")

    df_esp = rejilla_indice(base_path, color_path)
    if df_esp is None or df_esp.empty:
        logger.warning(f"[_process_one_k] k={k_val} => No data in df_esp => skipping.")
        return (k_val, sheet_name, None, None, None)

    df_idw, df_idw_2 = _idw_index_core(df_esp)  # IDW
    if df_idw is None or df_idw_2 is None:
        logger.warning(f"[_process_one_k] k={k_val} => IDW failed => partial data.")
        return (k_val, sheet_name, df_esp, None, None)

    df_qgis, _ = _riesgo(df_idw_2, XC.copy())

    logger.info(f"[_process_one_k] k={k_val} => success => returning data (sheet='{sheet_name}')")
    return (k_val, sheet_name, df_esp, df_idw, df_qgis)


def bulk_unzip_and_analyze_new_parallel(
    indice, anio, base_folder="./upload_data", colorMap_keyword="ColorMap"
):
    """
    Main bulk analysis pipeline:
      1) Unzip .zip pairs.
      2) Identify (base .tiff) + (ColorMap .tiff) by numeric prefix (001, 002, etc.).
      3) For each pair => Rejilla + IDW + Riesgo => generate Espacial, IDW, QGIS output.
      4) Store final Excel output files into 'assets/data/' instead of the local subfolder.
    Returns (espacial_xlsx, idw_xlsx, qgis_xlsx) or (None, None, None) on error.
    """
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Starting => indice='{indice}', anio='{anio}'")
    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Created folder_path='{folder_path}'")

    # 1) Unzip all .zip in folder_path
    for file_ in os.listdir(folder_path):
        if file_.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, file_)
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Unzipping => {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder_path)
                # optional rename logic if needed
                # (commented out because your naming might differ)
                # prefix = file_[0:5]
                # nlist = zf.namelist()

    # 2) Gather color map .tiff
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

    # We'll store final results in assets/data
    output_dir = os.path.join("assets", "data")
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

    # 3) Identify max K by prefix
    max_k = 0
    for f in color_files:
        try:
            # expecting "001... 002..."
            k_ = int(f[0:3])
            if k_ > max_k:
                max_k = k_
        except:
            pass
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Computed max_k={max_k}")
    if max_k == 0:
        warn_msg = "No numeric prefixes found in colorMap files (like '001')."
        logger.warning(warn_msg)
        st.warning(warn_msg)
        return None, None, None

    # Shared cluster seeds
    XC = np.sort(np.random.uniform(0, 1, 5))
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Initial cluster seeds (XC)={XC}")

    # 4) Create tasks for k in [1..max_k]
    tasks = range(1, max_k+1)

    # 5) Parallel processing
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

    # sort results
    results.sort(key=lambda r: r[0])

    # 6) Write to final Excel files
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Writing results => {espacial_xlsx}, {idw_xlsx}, {qgis_xlsx}")
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

def Emision(i2, XDe, NIT, NR):
    """
    [COPIED FROM NOTEBOOK]
    Keras-based “toy forecast” to produce prospective risk classification
    for a single climate variable.
    Returns (Vp1, Vp2, yp, XC1p).

    i2 : int (variable index)
    XDe: np.array, climate data for a single variable
    NIT: int, #epochs for Keras training
    NR : int, # of monthly lags
    """


    try:
        # 1) Force numeric => avoid object dtype
        XDe = pd.Series(XDe).astype(float).values
        if len(XDe) < 400:  # we need at least 360+some offset
            # If there's not enough data to do a 360-day forecast + lags => skip
            return None, None, None, None

        # 2) scale data 0..1
        XDen = (XDe - XDe.min()) / (XDe.max() - XDe.min())
        if (XDe.max() - XDe.min()) == 0:
            # means data is all the same => can't scale
            return None, None, None, None

        npr = 30  # days per month offset

        # 3) Build the dataset with lags = NR months
        XDst = np.zeros((len(XDe) - 30*(NR-1), NR))
        for k in range(NR):
            for i in range(len(XDe) - 30*(NR-1)):
                XDst[i, k] = XDen[i + k*30]

        # 4) Build “y” vector
        ydst = np.zeros((len(XDe) - 30*(NR-1)))
        for i in range(len(XDe) - (360 + 30*(NR-1))):
            ydst[i] = XDen[i + (360 + 30*(NR-1))]

        # 5) Neural net
        model = Sequential()
        model.add(Dense(100, activation='relu', use_bias=False, input_dim=NR, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(50, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(25, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.01))
        model.add(Dense(1, activation='relu', use_bias=False, kernel_regularizer=l2(0.001)))
        model.compile(optimizer='adam', loss='mse', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='loss', patience=10, mode="min",
                                       restore_best_weights=True, verbose=0)

        # 6) Fit
        #    we skip last 360 rows from XDst, because they are used for “forecast”
        XDn = XDst[:-360, :]
        ydn = ydst[:-360]
        if len(XDn) < 1 or len(ydn) < 1:
            # Not enough data for training
            return None, None, None, None

        model.fit(XDn, ydn, epochs=NIT, batch_size=250, callbacks=[early_stopping], verbose=0)

        # 7) Forecast next 360
        XDpn = XDst[-360:, :]
        yp_scaled = model.predict(XDpn)
        # Revert scale
        rng = (XDe.max() - XDe.min())
        yp = yp_scaled*(rng) + XDe.min()

        # 8) Evaluate quantiles
        XC1p = np.percentile(yp, [10,20,30,40,50,60,70,80,90,100])  # 10 deciles

        # 9) classification => Vp1, Vp2
        Vp1 = np.zeros((12,1))
        Vp2 = np.zeros((12,1))
        for i in range(12):
            idx = (i+1)*30 - 1
            val = yp[idx]
            d2 = np.abs(XC1p - val)
            near_idx = np.argmin(d2)
            Vp1[i] = near_idx
            # Some offset for risk classification
            if near_idx < 5:
                Vp2[i] = 4 - near_idx
            else:
                Vp2[i] = near_idx - 5

        return Vp1, Vp2, yp, XC1p

    except Exception as e:
        logger.exception("Error in Emision:")
        return None, None, None, None

def MatricesTransicion(XD, NDVI_df, n_var, punto, n_components=5):
    """
    [COPIED FROM NOTEBOOK] 
    Build Transition (aTr) and Emission (bEm) matrices for a single point (punto).
    - XD: DataFrame with climate data columns => [MaxC, MinC, Viento, Humedad, Precip, NDVI].
    - NDVI_df: A memory structure that has shape Nx? => row=some QGIS data
    - punto: int index of the point to analyze
    Returns (aTr, bEm, XCr).
    """
    try:
        # LDA => 1 - NDVI
        ##print XD shape
        print(f"***********************Shape of XD: {XD.shape}")
        print(f"***********************Shape of NDVI_df: {NDVI_df.shape}")
        print(f"***********************XD Values: {XD.values}")
        LDA = np.array(XD.iloc[:, n_var])  # NDVI column
        LDA = 1 - LDA
        ##LDA = LDA[~np.isnan(LDA)]

        # KMeans => cluster NDVI into n_components states
        seed = np.random.randint(0, 1000)
        km = KMeans(n_clusters=n_components, random_state=seed)
        km.fit(LDA.reshape(-1,1))

        # cluster centers
        XC = sorted(km.cluster_centers_.flatten(), reverse=False)
        XC = np.array(XC).reshape(1, n_components)

        # cluster assignment
        XCr = np.zeros((len(LDA), 1))
        for i in range(len(LDA)):
            d1 = np.abs(XC - LDA[i])
            c_idx = np.argmin(d1, axis=1)[0]
            XCr[i] = c_idx
            # update center
            XC[0, int(c_idx)] = (LDA[i] + XC[0, int(c_idx)])/2

        # Build transition matrix (aTr)
        MTr = np.zeros((n_components, n_components))
        for i in range(len(XCr)-1):
            fila = int(XCr[i])
            col = int(XCr[i+1])
            MTr[fila, col] += 1
        row_sums = MTr.sum(axis=1)
        for r in range(n_components):
            if row_sums[r] == 0:
                row_sums[r] = 1e-9
        MTr = MTr / row_sums[:, None]

        # Build emission matrices (bEm) => shape [n_var, n_components, n_components]
        # Each climate variable is assigned a 5-level classification
        WD = np.zeros((len(LDA), n_var))
        for j in range(n_var):
            WD[:, j] = XD.iloc[:, j].values

        # random expansions for temperature
        for j in [0,1]:  # MaxC, MinC
            for k2 in range(len(WD)):
                WD[k2,j] = (0.9 + 0.2*np.random.random())*WD[k2,j]

        bEm = np.zeros((n_var, n_components, n_components))
        for j in range(n_var):
            km2 = KMeans(n_clusters=2*n_components, random_state=seed)
            km2.fit(WD[:, j].reshape(-1,1))
            c2 = sorted(km2.cluster_centers_.flatten(), reverse=False)
            c2 = np.array(c2)

            climate_class = np.zeros(len(LDA))
            for i2 in range(len(LDA)):
                dd = np.abs(c2 - WD[i2,j])
                idx = np.argmin(dd)
                if idx < 5:
                    climate_class[i2] = 4 - idx
                else:
                    climate_class[i2] = idx - 5

            for i2 in range(len(LDA)):
                fila = int(XCr[i2,0])
                col = int(climate_class[i2])
                bEm[j, fila, col] += 1

            # normalize
            for s in range(n_components):
                row_sum = bEm[j, s, :].sum()
                if row_sum == 0:
                    row_sum = len(LDA)
                bEm[j, s, :] /= row_sum


        lonp = NDVI_df[0, punto, 1]  # might vary
        latp = NDVI_df[0, punto, 2]

        return MTr, bEm, XCr, lonp, latp
    except Exception as e:
        logger.exception(f"Error in MatricesTransicion for point {punto}:")
        return None, None, None, None, None

def Prospectiva(i1, XD, XCr, V, aTr, bEm, ydmes):
    """
    HPC routine from notebook that simulates monthly risk distribution (XLDA) for next 12 months.
    
    i1: point index
    XD: climate + NDVI data for the entire time series
    XCr: NDVI cluster assignments
    V: (5x12) pattern from Emision for all climate variables (the 'risk states')
    aTr: NDVI risk transition matrix
    bEm: climate emission matrix
    ydmes: shape(12x5) => forecast for next 12 months

    Returns:
      VC   => array of string labels, e.g. ["High 1", "Average 2", ...] 
      XInf => shape (12 x 12) with monthly stats (skew, means, etc.)
      XLDA => shape (1000 x 12), random draws for monthly losses
    """
    import random

    n_components = 5
    n_var = 5

    LDA = np.array(XD.iloc[:, n_var])  # NDVI col => last climate col
    LDA = 1 - LDA

    # cluster proportion
    nC = np.zeros((n_components,1))
    inr = np.zeros((n_components,))
    for j in range(n_components):
        nC[j] = len(np.where(XCr==j)[0])/len(LDA)
        inr[j] = nC[j]

    XLDA = np.zeros((1000, V.shape[1]), dtype=np.float64)
    XInf = np.zeros((V.shape[1], 12), dtype=np.float64)

    # initial stats
    XInf[0,0] = ydmes[0,0]
    XInf[0,1] = ydmes[0,1]
    XInf[0,2] = ydmes[0,2]
    XInf[0,3] = ydmes[0,3]
    XInf[0,4] = ydmes[0,4]
    XInf[0,5] = np.round(skewfunc(LDA),3)
    XInf[0,6] = np.round(nC[0]+nC[1],3)
    XInf[0,7] = np.round(nC[2]+nC[3],3)
    XInf[0,8] = np.round(nC[4],3)
    XInf[0,9] = np.round(np.mean(LDA),3)
    XInf[0,10] = np.round(np.percentile(LDA,75),3)
    XInf[0,11] = np.round(np.percentile(LDA,99),3)

    # build “VC”
    VC = []
    for k in range(V.shape[1]):
        if V[4,k] == 0:
            VC.append(f"High {k+1}")
        elif V[4,k] == 1:
            VC.append(f"Average {k+1}")
        elif V[4,k] == 2:
            VC.append(f"Low {k+1}")
        elif V[4,k] == 3:
            VC.append(f"Very Low {k+1}")
        elif V[4,k] == 4:
            VC.append(f"Dry {k+1}")

    # alpha => row=month, col=component
    alpha = np.zeros((V.shape[1], n_components), dtype=np.float64)

    # initialization
    alpha[0, :] = 0
    for m in range(n_var):
        alpha[0, :] += inr * bEm[m, :, V[m,0]]
    alpha[0, :] /= alpha[0, :].sum()

    # draw ~1000 for month 0
    NDm = np.int32(1000*alpha[0, :])
    m1 = -1
    for k2 in range(n_components):
        filas = np.where(XCr==k2)[0]
        LDAm = LDA[filas]
        um = np.mean(LDAm)
        sigmam = np.sqrt(np.var(LDAm))
        for i2 in range(NDm[k2]):
            m1 += 1
            XLDA[m1, 0] = (0.8 + 0.4*random.random()) * np.random.normal(um, 2*sigmam)

    XInf[1,0] = ydmes[0,0]
    XInf[1,1] = ydmes[0,1]
    XInf[1,2] = ydmes[0,2]
    XInf[1,3] = ydmes[0,3]
    XInf[1,4] = ydmes[0,4]
    XInf[1,5] = round(skewfunc(XLDA[:,0]), 3)
    XInf[1,6] = round(alpha[0,0]+alpha[0,1],3)
    XInf[1,7] = round(alpha[0,2]+alpha[0,3],3)
    XInf[1,8] = round(alpha[0,4],3)
    XInf[1,9] = round(np.mean(XLDA[:,0]),3)
    XInf[1,10] = round(np.percentile(XLDA[:,0],75),3)
    XInf[1,11] = round(np.percentile(XLDA[:,0],99),3)

    # loop months
    for t in range(1, V.shape[1]):
        alpha[t, :] = 0
        for m in range(n_var):
            alpha[t, :] += alpha[t-1].dot(aTr) * bEm[m, :, V[m,t]]
        alpha[t, :] /= alpha[t, :].sum()

        NDm = np.int32(1000*alpha[t, :])
        m1 = -1
        for k2 in range(n_components):
            filas = np.where(XCr==k2)[0]
            LDAm = LDA[filas]
            um = np.mean(LDAm)
            sigmam = np.sqrt(np.var(LDAm))
            for i2 in range(NDm[k2]):
                m1 += 1
                XLDA[m1, t] = (0.9 + 0.2*random.random()) * np.random.normal(um, sigmam)

        XInf[t,0] = ydmes[t,0]
        XInf[t,1] = ydmes[t,1]
        XInf[t,2] = ydmes[t,2]
        XInf[t,3] = ydmes[t,3]
        XInf[t,4] = ydmes[t,4]
        XInf[t,5] = round(skewfunc(XLDA[:,t-1]),3)
        XInf[t,6] = round(alpha[t,0]+alpha[t,1],3)
        XInf[t,7] = round(alpha[t,2]+alpha[t,3],3)
        XInf[t,8] = round(alpha[t,4],3)
        XInf[t,9] = round(np.mean(XLDA[:,t-1]),3)
        XInf[t,10] = round(np.percentile(XLDA[:,t-1],75),3)
        XInf[t,11] = round(np.percentile(XLDA[:,t-1],99),3)

    return np.array(VC), XInf, XLDA



def run_full_hpc_pipeline(indice:str, anio:str, base_folder:str="./upload_data"):
    """
    Replicates the EXACT logic from your Jupyter notebook for HPC:
      1) Read & invert Clima => Emision => build V, ydmes
      2) Read QGIS => build a 3D array XD3 => (#sheets x #points x 5 cols)
      3) For each point => gather NDVI timeseries from XD3 => HPC

    Returns:
       A dictionary => { "V":..., "ydmes":..., "results":[ { "point_idx", "sheet_names", "XInf", ...}, ... ] }
    """

    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found => {folder_path}")

    # 1) Invert climate => same as your HPC code
    clima_path = os.path.join(folder_path, f"Clima_{indice}_{anio}.xlsx")
    if not os.path.exists(clima_path):
        logger.error(f"Climate file not found => {clima_path}")
        return None

    logger.info(f"[HPC-Notebook-Style] Reading & inverting => {clima_path}")
    dfc = pd.read_excel(clima_path)
    df_inverted = dfc.iloc[::-1]
    clima_inverted_path = os.path.join(folder_path, f"Clima_{indice}_{anio}_O.xlsx")
    print(f"***********************Clima inverted Path: {clima_inverted_path}")
    df_inverted.to_excel(clima_inverted_path, index=False)

    # Reload inverted
    XDB = pd.read_excel(clima_inverted_path)
    print(f"***********************Shape of XDB: {XDB.shape}")  
    print(f"***********************Columns of XDB: {XDB.columns}")
    print(f"***********************Values of XDB: {XDB.values}")
    '''
    if "Fuente de datos" in XDB.columns:
        XDB = XDB.dropna(subset=["Fuente de datos"], how="any")'
    '''
    XDB=XDB[XDB['Fuente de datos']!='-']
    print(f"***********************Values of XDB after dropout: {XDB.values}")
    # columns => [7,8,11,12,13,4]
    col_idx = [7,8,11,12,13,4]
    try:
        XD = XDB.iloc[:, col_idx].copy()
    except Exception as e:
        logger.exception(f"Cannot select col {col_idx} from climate => {e}")
        return None

    # Force numeric
    XD = XD.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if len(XD) < 1:
        logger.error("[HPC-Notebook-Style] climate data empty => cannot proceed.")
        return None

    n_components = 5
    n_var = 5

    # 2) Build "V" from Emision => exactly as notebook
    # We do it for each climate var => 5
    # Then build ydmes => shape(12,5)

    V = np.zeros((n_var, 12), dtype=int)
    ydpar = np.zeros((360, n_var), dtype=float)
    for i in range(n_var):
        XDe = XD.iloc[:, i].values
        Vp1, Vp2, yp, XC1p = Emision(i, XDe, 500, 4)
        if (Vp1 is None) or (Vp2 is None) or (yp is None):
            logger.warning(f"[HPC-Notebook-Style] Emision => var {i} => None => skipping.")
            continue
        V[i, :] = Vp2.flatten()
        ydpar[:, i] = yp.flatten()

    ydmes = np.zeros((12, n_var), dtype=float)
    for i in range(12):
        idx = 30*(i+1) - 1
        if idx >= 0 and idx < len(ydpar):
            ydmes[i,:] = ydpar[idx,:]

    # 3) Read QGIS => create XD3 => #sheets x #points x 5
    qgis_path = os.path.join("assets","data", f"INFORME_{indice}_QGIS_{anio}.xlsx")
    if not os.path.exists(qgis_path):
        logger.error(f"[HPC-Notebook-Style] QGIS not found => {qgis_path}")
        return None

    XDB3 = pd.read_excel(qgis_path, sheet_name=None)
    # Build a list of sheets => array_hojas = list(XDB3.values())
    sheet_names = list(XDB3.keys())
    array_hojas = [df.values for df in XDB3.values()]

    # Suppose each sheet => shape(25, 5) => 5x5=25 points
    n_sheets = len(array_hojas)
    if n_sheets < 1:
        logger.error("[HPC-Notebook-Style] no sheets found in QGIS => skipping")
        return None

    # shape => (n_sheets, 25, 5)
    # careful: your code => res=5 => #points=25
    res = 5
    total_points = res*res
    XD3 = np.empty((n_sheets, total_points, 5), dtype=float)
    for iSheet, hoja_np in enumerate(array_hojas):
        # hoja_np => shape(25,5)
        XD3[iSheet, :, :] = hoja_np

    # HPC results
    results = []

    # 4) For each point => gather NDVI time-series => HPC
    # exactly like your notebook
    for iPoint in range(total_points):
        # build NDVI from XD3[ iSheet, iPoint, 3 ] for iSheet in [0..n_sheets-1]
        # LDA = NDVI timeseries
        LDA = []
        for iSheet in range(n_sheets):
            LDA.append( XD3[iSheet, iPoint, 3] )
        LDA = np.array(LDA)

        # Overwrite NDVI in your climate DF => we want the same shape as XD
        # "XD" was the climate timeseries => we basically do a copy, then replace NDVI col with LDA
        # but your notebook does "MatricesTransicion(XD, XD3, n_var, iPoint)"
        # which means it reads "XD3[iSheet, iPoint, 3]" inside the function, ignoring "LDA"?
        # We'll replicate your *exact* approach:

        aTr, bEm, XCr, lonp, latp = MatricesTransicion(XD, XD3, n_var, iPoint)
        if aTr is None or bEm is None:
            logger.warning(f"[HPC-Notebook-Style] point={iPoint} => no HPC => skip.")
            continue

        # HPC => Prospectiva
        VC, XInf2, XLDA = Prospectiva(iPoint, XD, XCr, V, aTr, bEm, ydmes)
        info = {
            "point_idx": iPoint,
            "sheet_names": sheet_names,
            "VC": VC,
            "XInf": XInf2,
            "XLDA": XLDA
        }
        results.append(info)

    # Return final
    hpc_data = {
        "V": V,
        "ydmes": ydmes,
        "results": results
    }
    return hpc_data
