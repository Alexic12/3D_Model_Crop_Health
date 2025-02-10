import pandas as pd
import numpy as np
import streamlit as st
import logging
from scipy.interpolate import griddata
import rasterio
from pyproj import Transformer
import logging

logger = logging.getLogger(__name__)

import os
import zipfile
import datetime
from PIL import Image
import shutil  # if you need to move or remove directories


def process_uploaded_file(uploaded_file):
    """
    Reads a single-sheet Excel file with columns [Longitud, Latitud, NDVI, Riesgo].
    Ensures NDVI is in [-1.5, 1.5], Riesgo in [1..5].
    Returns a DataFrame if valid, otherwise None.
    """
    try:
        logger.info("Processing uploaded file")
        
        # Read the Excel file into a DataFrame (first sheet by default)
        df = pd.read_excel(uploaded_file)
        
        # Verify required columns
        required_columns = ["Longitud", "Latitud", "NDVI", "Riesgo"]
        if not all(col in df.columns for col in required_columns):
            error_message = (
                "The uploaded file must contain the following columns: "
                f"{', '.join(required_columns)}"
            )
            st.error(error_message)
            logger.error(error_message)
            return None

        # Extract the required columns
        data = df[required_columns].copy()
        
        # Handle missing values
        if data.isnull().values.any():
            st.warning("Missing values detected. Rows with missing values will be dropped.")
            logger.warning("Missing values in data")
            data.dropna(subset=required_columns, inplace=True)
        
        # Validate NDVI range
        if not ((data["NDVI"] >= -1.5) & (data["NDVI"] <= 1.5)).all():
            error_message = "NDVI values must be between -1.5 and 1.5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Validate Riesgo range
        if not ((data["Riesgo"] >= 1) & (data["Riesgo"] <= 5)).all():
            error_message = "Riesgo values must be between 1 and 5"
            st.error(error_message)
            logger.error(error_message)
            return None

        # Convert Riesgo to integer
        data["Riesgo"] = data["Riesgo"].astype(int)
        
        logger.info("File processed successfully")
        return data

    except Exception as e:
        error_message = f"An error occurred while processing the file: {e}"
        st.error(error_message)
        logger.exception("Error processing file")
        return None


def load_timeseries_data(uploaded_file):
    """
    Reads an Excel file with multiple sheets, each containing a matrix:
      - First row is discarded.
      - Second row (index=1) has longitude in columns [1..end].
      - First column in rows [2..end] has latitude.
      - Intersection is NDVI data in 2D form.

    Returns a dict { sheet_name: {"lon":..., "lat":..., "ndvi":...} } or None on error.
    """
    try:
        logger.info("Loading time-series data from multiple sheets...")

        excel_obj = pd.ExcelFile(uploaded_file)
        sheet_names = excel_obj.sheet_names
        data_sheets = {}

        for sname in sheet_names:
            logger.info(f"Reading sheet: {sname}")
            df = pd.read_excel(uploaded_file, sheet_name=sname, header=None)
            
            # Basic check
            if df.shape[0] < 3 or df.shape[1] < 2:
                logger.warning(f"Sheet {sname} is too small. Skipping.")
                continue
            
            # The 1D array of longitude is from row=1, columns=1..end
            lon = df.iloc[1, 1:].to_numpy(dtype=float)
            # The 1D array of latitude is from rows=2..end, col=0
            lat = df.iloc[2:, 0].to_numpy(dtype=float)
            # The 2D NDVI matrix is from rows=2..end, columns=1..end
            ndvi_matrix = df.iloc[2:, 1:].to_numpy(dtype=float)
            
            data_sheets[sname] = {
                "lon": lon,
                "lat": lat,
                "ndvi": ndvi_matrix
            }

        if not data_sheets:
            st.error("No valid sheets found or parsing failed in the uploaded file.")
            return None

        logger.info("Successfully loaded all sheets for time-series data.")
        return data_sheets

    except Exception as e:
        error_message = f"An error occurred while loading time-series data: {e}"
        st.error(error_message)
        logger.exception("Error in load_timeseries_data")
        return None


def griddata_points_to_grid(x, y, values, xi, yi):
    """
    Interpolates scattered (x, y, values) onto a regular grid (xi, yi).
    Replaces NaN with 0.0 after interpolation.
    """
    points = np.column_stack((x, y))
    grid_z = griddata(points, values, (xi, yi), method='linear')
    grid_z = np.nan_to_num(grid_z, nan=0.0)
    return grid_z


def rejilla_indice(ruta_imagen, ruta_imagen_2):
    # 1. Load the colorized image using PIL
    imagen = Image.open(ruta_imagen_2)
    pixels = imagen.load()  # for color info

    xm = []; ym = []
    latm = []; lonm = []
    colm = []; rowm = []
    NDVI = []

    # 2. Open the NDVI (grayscale) with rasterio
    with rasterio.open(ruta_imagen) as src:
        banda1 = src.read(1)
        crs = src.crs
        print("CRS de la imagen:", crs)

        # For debugging, let's check first pixel
        x_0, y_0 = src.xy(0, 0)
        print("Coordenadas del píxel (0,0):", x_0, y_0)

        print("el ancho es", src.width)
        print("el alto es", src.height)

        # Convert no_data to something if needed
        no_data_value = src.nodata if src.nodata is not None else -9999
        banda1 = np.where(banda1 == no_data_value, np.nan, banda1)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        # 3. Loop over each pixel
        for row_i in range(src.height):
            for col_i in range(src.width):
                val = banda1[row_i, col_i]
                if val is not np.nan and val != -9999.0:
                    x, y = src.xy(row_i, col_i)
                    # Retrieve color pixel from PIL (note the order is (col, row) for PIL)
                    color_vals = pixels[col_i, row_i]  # might be RGBA or RGB

                    # Convert them to [0..1] floats
                    if isinstance(color_vals, int):
                        # Single-band? 
                        color_vals = (color_vals / 255.0,)
                    else:
                        color_vals = tuple(c / 255.0 for c in color_vals)

                    # Transform to lat/lon
                    lon, lat = transformer.transform(x, y)

                    xm.append(x);    ym.append(y)
                    lonm.append(lon); latm.append(lat)
                    colm.append(col_i); rowm.append(row_i)
                    NDVI.append(val)

    df = pd.DataFrame({
        'UTM-x': xm,
        'UTM-y': ym,
        'longitud': lonm,
        'latitud': latm,
        'col': colm,
        'row': rowm,
        'NDVI': NDVI
    })
    return df


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



def Riesgo(dfm, XC, df_2):
    """
    This replicates your code that used `df_2[1]['NDVI']` as XLDA, etc.
    We must pass in the second output from IDW_Index (df_2) so we can get 'NDVI'.
    """
    nXC = np.zeros((25, 1))
    # The second item from IDW_Index is df_2[1], but here we just do:
    XLDA = np.array(df_2['NDVI'])  # using that logic

    for k1 in range(2):
        XC = np.sort(XC)[::-1]
        for k in range(25):
            if k >= len(XLDA):
                break  # if your data is smaller than 25
            dist_sq = (XC - XLDA[k])**2
            ncj = np.argmin(dist_sq)
            XC[ncj] = (XC[ncj] + XLDA[k]) / 2
            nXC[k, 0] = ncj

    dfm['Riesgo'] = nXC[:len(dfm), 0] + 1  # just to match your code shape
    return dfm, XC


def invert_climate_file_rows(file_buffer, output_filename=None):
    """
    Reads an Excel climate file from an uploaded buffer,
    inverts its rows, and optionally returns or saves the result.
    """
    try:
        df = pd.read_excel(file_buffer)
        # invert
        df_inverted = df.iloc[::-1]

        if output_filename:
            df_inverted.to_excel(output_filename, index=False)
            return None
        else:
            return df_inverted
    except Exception as e:
        logger.exception(f"Error in invert_climate_file_rows: {e}")
        return None
    

def bulk_unzip_and_analyze(indice, anio, base_folder="./upload_data"):
    """
    This is the original main routine that:
      1) Looks in base_folder/<indice>_<anio>/ for .zip files named like "001. perimetro__prev_NDVI_..."
      2) Unzips them, renames them
      3) Identifies each colorMap pair
      4) Runs rejilla_indice, IDW_Index, Riesgo
      5) Writes the results to an Excel with NDVI_RESULTS_<YYYY-MM-DD_HHMM>.xlsx
    """

    import os
    import time
    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # 1) Unzip all .zip
    for archivo in os.listdir(folder_path):
        if archivo.lower().endswith('.zip'):
            ruta_zip = os.path.join(folder_path, archivo)
            print(f"Procesando: {ruta_zip}")

            with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
                print(f"Archivos extraídos en: {folder_path}")
                print(zip_ref.namelist())

                prefix = archivo[0:5]  # e.g. "001. "
                name_list = zip_ref.namelist()
                if len(name_list) >= 2:
                    fn1 = name_list[0]
                    fn2 = name_list[1]
                    old_path1 = os.path.join(folder_path, fn1)
                    old_path2 = os.path.join(folder_path, fn2)

                    new_name1 = prefix + fn1
                    new_name2 = prefix + fn2
                    new_path1 = os.path.join(folder_path, new_name1)
                    new_path2 = os.path.join(folder_path, new_name2)

                    if os.path.exists(new_path1):
                        os.remove(old_path1)
                    else:
                        os.rename(old_path1, new_path1)

                    if os.path.exists(new_path2):
                        os.remove(old_path2)
                    else:
                        os.rename(old_path2, new_path2)

    nombre = "ColorMap"
    files_with_colormap = [f for f in os.listdir(folder_path)
                           if nombre in f and f.lower().endswith('.tiff')]
    print("Cantidad de archivos con ColorMap:", len(files_with_colormap))

    XC = np.sort(np.random.uniform(0,1,5))
    print("Inicial XC:", XC)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    results_filename = os.path.join(folder_path, f"NDVI_RESULTS_{date_str}.xlsx")

    max_k = 0
    for f in files_with_colormap:
        prefix_str = f[0:3]  # "001"
        if prefix_str.isdigit():
            k_val = int(prefix_str)
            if k_val > max_k:
                max_k = k_val

    print("Tenemos un max K:", max_k)

    for k in range(1, max_k+1):
        k_str = str(k).zfill(3)
        colormap_file = None
        base_file = None
        for f in os.listdir(folder_path):
            if f.startswith(k_str):
                if nombre in f:  # "ColorMap"
                    colormap_file = f
                else:
                    if f.lower().endswith('.tiff'):
                        base_file = f

        if colormap_file is None or base_file is None:
            continue

        print(f"Processing K={k}: base={base_file}, color={colormap_file}")

        ruta_base = os.path.join(folder_path, base_file)
        ruta_color = os.path.join(folder_path, colormap_file)

        df = rejilla_indice(ruta_base, ruta_color)

        dfidw, dfidw_2 = IDW_Index(df)

        df2x, XC = Riesgo(dfidw_2, XC, dfidw_2)

        from openpyxl import Workbook, load_workbook
        from openpyxl.utils.exceptions import InvalidFileException

        def append_df_to_excel(filename, dfdata, sheet_name):
            try:
                with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                    dfdata.to_excel(writer, index=False, sheet_name=sheet_name)
            except FileNotFoundError:
                with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                    dfdata.to_excel(writer, index=False, sheet_name=sheet_name)
            except InvalidFileException:
                with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                    dfdata.to_excel(writer, index=False, sheet_name=sheet_name)

        append_df_to_excel(results_filename, df,        f"{k_str}_Espacial")
        append_df_to_excel(results_filename, dfidw,     f"{k_str}_IDW0")
        append_df_to_excel(results_filename, dfidw_2,   f"{k_str}_IDW1")
        append_df_to_excel(results_filename, df2x,      f"{k_str}_QGIS")

    print("ALL DONE. Final clusters:", XC)
    print(f"Results written to: {results_filename}")

    return results_filename


# ----------------------------------------------------------------------------
# New Parallel Implementation
# ----------------------------------------------------------------------------

import concurrent.futures

def _process_one_k(args):
    """
    A helper function for parallel processing of each K.
    We'll do basically what the loop in bulk_unzip_and_analyze does, but for one K.
    """
    (k, folder_path, XC, nombre) = args

    k_str = str(k).zfill(3)
    colormap_file = None
    base_file = None
    for f in os.listdir(folder_path):
        if f.startswith(k_str):
            if nombre in f:  # "ColorMap" in file
                if f.lower().endswith('.tiff'):
                    colormap_file = f
            else:
                if f.lower().endswith('.tiff'):
                    base_file = f

    if colormap_file is None or base_file is None:
        return (k_str, None, XC)  # no data => skip

    from data_processing import rejilla_indice, IDW_Index, Riesgo

    ruta_base = os.path.join(folder_path, base_file)
    ruta_color = os.path.join(folder_path, colormap_file)

    print(f"[Worker PID] Processing K={k_str}: base={base_file}, color={colormap_file}")
    df = rejilla_indice(ruta_base, ruta_color)
    if df is None:
        return (k_str, None, XC)

    dfidw, dfidw_2 = IDW_Index(df)
    df2x, XC_updated = Riesgo(dfidw_2, XC, dfidw_2)

    # Return the data for writing to Excel in main process
    return (k_str, (df, dfidw, dfidw_2, df2x), XC_updated)


def bulk_unzip_and_analyze_parallel(indice, anio, base_folder="./upload_data"):
    """
    Parallel version of bulk_unzip_and_analyze using concurrent.futures to
    process each K in a separate process. This can speed up if you have multiple cores.
    """

    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # 1) Unzip all .zip (same as before)
    for archivo in os.listdir(folder_path):
        if archivo.lower().endswith('.zip'):
            ruta_zip = os.path.join(folder_path, archivo)
            print(f"Procesando: {ruta_zip}")

            with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
                print(f"Archivos extraídos en: {folder_path}")
                print(zip_ref.namelist())

                prefix = archivo[0:5]
                name_list = zip_ref.namelist()
                if len(name_list) >= 2:
                    fn1 = name_list[0]
                    fn2 = name_list[1]
                    old_path1 = os.path.join(folder_path, fn1)
                    old_path2 = os.path.join(folder_path, fn2)

                    new_name1 = prefix + fn1
                    new_name2 = prefix + fn2
                    new_path1 = os.path.join(folder_path, new_name1)
                    new_path2 = os.path.join(folder_path, new_name2)

                    if os.path.exists(new_path1):
                        os.remove(old_path1)
                    else:
                        os.rename(old_path1, new_path1)

                    if os.path.exists(new_path2):
                        os.remove(old_path2)
                    else:
                        os.rename(old_path2, new_path2)

    nombre = "ColorMap"
    files_with_colormap = [f for f in os.listdir(folder_path)
                           if nombre in f and f.lower().endswith('.tiff')]
    print("Cantidad de archivos con ColorMap:", len(files_with_colormap))

    import numpy as np
    XC = np.sort(np.random.uniform(0,1,5))
    print("Inicial XC:", XC)

    import datetime
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    results_filename = os.path.join(folder_path, f"NDVI_RESULTS_{date_str}.xlsx")

    max_k = 0
    for f in files_with_colormap:
        prefix_str = f[0:3]
        if prefix_str.isdigit():
            k_val = int(prefix_str)
            if k_val > max_k:
                max_k = k_val

    print("Tenemos un max K:", max_k)

    # Build tasks
    tasks = []
    for k in range(1, max_k+1):
        tasks.append((k, folder_path, XC, nombre))

    # Parallel
    from openpyxl.utils.exceptions import InvalidFileException
    from openpyxl import Workbook, load_workbook
    import pandas as pd

    def append_df_to_excel(filename, dfdata, sheet_name):
        try:
            with pd.ExcelWriter(filename, mode='a', engine='openpyxl', if_sheet_exists='new') as writer:
                dfdata.to_excel(writer, index=False, sheet_name=sheet_name)
        except FileNotFoundError:
            with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                dfdata.to_excel(writer, index=False, sheet_name=sheet_name)
        except InvalidFileException:
            with pd.ExcelWriter(filename, mode='w', engine='openpyxl') as writer:
                dfdata.to_excel(writer, index=False, sheet_name=sheet_name)

    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers = 4) as executor:
        future_to_k = {
            executor.submit(_process_one_k, arg): arg[0] for arg in tasks
        }
        for future in concurrent.futures.as_completed(future_to_k):
            k_val = future_to_k[future]
            try:
                k_str, df_tuple, updated_XC = future.result()
            except Exception as e:
                print(f"K={k_val} generated an exception: {e}")
                continue

            if df_tuple is None:
                continue

            df, dfidw, dfidw_2, df2x = df_tuple
            # Overwrite global XC with last updated if you want sequential updates
            XC = updated_XC

            append_df_to_excel(results_filename, df,     f"{k_str}_Espacial")
            append_df_to_excel(results_filename, dfidw,  f"{k_str}_IDW0")
            append_df_to_excel(results_filename, dfidw_2,f"{k_str}_IDW1")
            append_df_to_excel(results_filename, df2x,   f"{k_str}_QGIS")

    print("ALL DONE (parallel). Final clusters:", XC)
    print(f"Results written to: {results_filename}")
    return results_filename
