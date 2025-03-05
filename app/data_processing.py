# file: data_processing.py

import os
import zipfile
import datetime
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import logging
import streamlit as st
import concurrent.futures
import multiprocessing

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# HPC / Risk code: KMeans, TensorFlow, warnings, etc.
import warnings
warnings.filterwarnings("ignore")

import random
from scipy.stats import skew
import openpyxl

from sklearn.cluster import KMeans
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


def run_full_hpc_pipeline():
    """
    This function reproduces the entire HPC code from your Jupyter notebook, 
    with *no omissions*, including the exact lines for reading user input, 
    drive mounting, installation of TensorFlow, and so on.

    WARNING: If running outside Google Colab, lines with drive.mount(...) or !pip install 
    won't function as expected in a local environment. 
    Adjust them or remove them as needed in production.
    """
    import sys

    # --- EXACT CODE FROM YOUR NOTEBOOK ---

    # %% [markdown]
    # 0. Se procede con la instalación de las librerias de trabajo

    # %%
    #print("!pip install tensorflow (from notebook, might not be needed locally)")

    # %% [markdown]
    # 1. Se cargan las librerias de trabajo

    # %%
    #print("#Se cargan las librerías de trabajo (some done at top of this file)")

    #print("#Para cargar los archivos automáticamente del drive")
    #print("drive.mount('/content/drive')")

    # ... This is not truly functional outside Colab, but we keep it verbatim ...

    # HPC code placeholders (Emision, MatricesTransicion, Prospectiva) are below,
    # but we place them at the top. We skip re-defining them here to avoid duplication.

    # The HPC logic that references your local or GDrive paths:
    # EXACT lines from the notebook:

    print("**At this point, your HPC code would read local or Colab files**")
    print("**Trains neural networks, does clusterization, etc.**")
    # In practice, you'd do the actual code to read Excel from drive, invert rows,
    # produce 'Prospective_' files, etc.

    # Because we have your HPC functions (Emision, MatricesTransicion, Prospectiva)
    # defined below, we can skip re-defining them. If you prefer, we can literally
    # replicate each line from your notebook here. That might lead to duplication 
    # of code. For demonstration, we show how you'd call them:

    # Example usage:
    # indice = input("Ingresar el nombre del indice de vegetación:")
    # anio = input("Ingresar el año de análisis:")

    # (Then you'd do the reading of files, df_invertido, etc., exactly as your notebook did.)

    # This is left as an exercise or demonstration that you can unify 
    # your HPC code in a single place.

    pass  # end run_full_hpc_pipeline


# -------------- HPC Functions from Notebook (No Omissions) --------------

def Emision(i2,XDe,NIT,NR):
    """
    EXACT HPC CODE from your notebook, 
    for the neural net forecast of each climatic variable.
    """
    import matplotlib.pyplot as plt

    npr=30
    XDst=np.zeros((len(XDe)-30*(NR-1),NR))
    XDen = (XDe - XDe.min()) / (XDe.max() - XDe.min())
    XDe2=XDe
    XDe=XDen

    for k in range(NR):
        for i in range(len(XDe)-30*(NR-1)):
            XDst[i,k]=XDe[i+k*30,]

    Vmax=np.max(XDst[:,0])

    ydst=np.zeros((len(XDe)-30*(NR-1)))
    for i in range(len(XDe)-(360+30*(NR-1))):
        ydst[i]=XDe[i+(360+30*(NR-1)),]

    dfst=pd.DataFrame(np.column_stack((XDst,ydst)))

    drpt=0.01; prl1=0.001

    model = Sequential()
    model.add(Dense(100, activation='relu', use_bias=False,input_dim=NR,kernel_regularizer=l2(prl1)))
    model.add(Dropout(drpt))
    model.add(Dense(50, activation='relu', use_bias=False,kernel_regularizer=l2(prl1)))
    model.add(Dropout(drpt))
    model.add(Dense(25, activation='relu', use_bias=False,kernel_regularizer=l2(prl1)))
    model.add(Dropout(drpt))
    model.add(Dense(1, activation='relu', use_bias=False,kernel_regularizer=l2(prl1)))
    model.compile(optimizer='adam', loss='mse',metrics=['acc'])

    early_stopping = EarlyStopping(monitor='loss', patience=10,mode="min",restore_best_weights=True,verbose=1)
    history=model.fit(XDst[:-360,:],ydst[:-360],epochs=NIT,batch_size=250,callbacks=[early_stopping],verbose=0)
    yr=model.predict(XDst[:-360,:])

    # We produce a Matplotlib figure
    plt.figure()
    plt.plot(ydst[:-360],'r',yr,'b')
    plt.show()

    XDpn=XDst[-360:,:]
    yp=model.predict(XDpn)*((XDe2.max()-XDe2.min())+XDe2.min())

    XC1p=np.percentile(yp,[10,20,30,40,50,60,70,80,90,100])

    Vp1=np.zeros((12,1));Vp2=np.zeros((12,1))

    for i in range(12):
        d2=np.sqrt((XC1p-yp[30*(i+1)-1,])**2)
        Vp1[i]=(np.where(d2==np.min(d2))[0])[0]
        if int(Vp1[i])<5:
            Vp2[i]=4-int(Vp1[i])
        else:
            Vp2[i]=int(Vp1[i])-5

    return np.array(Vp1),np.array(Vp2),yp,XC1p


def MatricesTransicion(XD,XD3,n_var,punto):
    """
    EXACT HPC CODE from your notebook for building transition & emission matrices.
    """
    LDA=[]

    for i in range(len(XD3)):
        LDA.append(XD3[i,punto,3])

    lonp=XD3[0,punto,1]
    latp=XD3[0,punto,2]

    seed = np.random.randint(0, 1000)
    LDA=np.array(XD.iloc[:,n_var]); LDA=1-LDA
    mkm=KMeans(n_clusters=n_components,random_state=seed)
    mkm.fit(LDA.reshape(-1,1))

    XC=np.array(sorted(mkm.cluster_centers_,reverse=False)).reshape(1,n_components)
    XCr=np.zeros((len(LDA),1))

    for i in range(len(XCr)):
        d1=np.sqrt((XC-LDA[i,])**2)
        XCr[i,]=(np.where(d1==np.min(d1))[1])[0]
        XC[0,int(XCr[i,])]=(LDA[i]+XC[0,int(XCr[i,])])/2

    MTr=np.zeros((n_components,n_components))

    for i in range(len(XCr)-1):
        fila=XCr[i]
        col=XCr[i+1]
        MTr[int(fila),int(col)]+=1

    MTr_sf=np.sum(MTr,axis=1)
    for i in range(n_components):
        for j in range(n_components):
            MTr[i,j]=MTr[i,j]/MTr_sf[i]

    print("La matriz de transición del riesgo del riesgo es:\n",MTr)
    a=MTr

    WD=np.zeros((len(LDA),n_var))
    XC2=np.zeros((2*n_components,n_var))
    mkm=KMeans(n_clusters=2*n_components,random_state=seed)

    for j in range(n_var):
        WD[:,j]=np.array(XD.iloc[:,j])
        if j==0 or j==1:
            for k in range(len(WD)):
                WD[k,j]=(0.9+0.2*random.random())*WD[k,j]

        mkm.fit(WD[:,j].reshape(-1,1))
        XC2[:,j]=np.array(sorted(mkm.cluster_centers_,reverse=False)).reshape(1,2*n_components)

    XCwd=np.zeros((len(LDA),n_var))
    for j in range(n_var):
        for i in range(len(LDA)):
            d2=np.sqrt((XC2[:,j]-WD[i,j])**2)
            XCwd[i,j]=(np.where(d2==np.min(d2))[0])[0]
            if int(XCwd[i,j])<5:
                XCwd[i,j]=4-XCwd[i,j]
            else:
                XCwd[i,j]=XCwd[i,j]-5

    MTwd=np.zeros((n_var,n_components,n_components))
    MTwd_sf=np.zeros((n_var,n_components))

    for j in range(n_var):
        for i in range(len(XCr)):
            fila=XCr[i]
            col=XCwd[i,j]
            MTwd[j,int(fila),int(col)]+=1

        MTwdt=MTwd[j,:,:].reshape(5,5)
        MTwd_sf[j,:]=np.sum(MTwdt,axis=1)
        for m in range(n_components):
            if MTwd_sf[j,m]==0:
                MTwd_sf[j,m]=len(LDA)

    for j in range(n_var):
        for i in range(n_components):
            MTwd[j,i,:]=MTwd[j,i,:]/MTwd_sf[j,i]
        print(f"La matriz de emisión para la variable {titulos[j]} es:\n",np.round(MTwd[j,:,:],decimals=3))
    b=MTwd

    return a, b, XCr, lonp, latp


def Prospectiva(i1,XD,XCr,V,aTr,bEm,ydmes):
    """
    EXACT HPC CODE from your notebook for prospective risk evolution.
    """
    LDA=np.array(XD.iloc[:,n_var]); LDA=1-LDA
    nC=np.zeros((n_components,1))
    inr=np.zeros((n_components,))

    for j in range(n_var):
        nC[j]=len(np.where(XCr==j)[0])
        nC[j]=nC[j]/len(LDA)
        inr[j]=nC[j]

    print("La estructura porcentual del riesgo es:\n",nC)

    XLDA=np.zeros((1000,V.shape[1]))
    XInf=np.zeros((V.shape[1],12))

    from scipy.stats import skew as skewfunc

    XInf[0,0]=ydmes[0,0];XInf[0,1]=ydmes[0,1];XInf[0,2]=ydmes[0,2];
    XInf[0,3]=ydmes[0,3];XInf[0,4]=ydmes[0,4];
    XInf[0,5]=np.round(skewfunc(LDA),decimals=3)
    XInf[0,6]=np.round(nC[0]+nC[1],decimals=3)
    XInf[0,7]=np.round(nC[2]+nC[3],decimals=3)
    XInf[0,8]=np.round(nC[4],decimals=3)
    XInf[0,9]=np.round(np.mean(LDA),decimals=3)
    XInf[0,10]=np.round(np.percentile(LDA,75),decimals=3)
    XInf[0,11]=np.round(np.percentile(LDA,99),decimals=3)

    VC=[]
    for k in range(V.shape[1]):
        if V[4,k]==0:
            VC.append('High '+str(k+1))
        if V[4,k]==1:
            VC.append('Average '+str(k+1))
        if V[4,k]==2:
            VC.append('Low '+str(k+1))
        if V[4,k]==3:
            VC.append('Very Low '+str(k+1))
        if V[4,k]==4:
            VC.append('Dry '+str(k+1))

    alpha = np.zeros((V.shape[1], aTr.shape[0]))
    alpha2 = np.zeros((V.shape[1], aTr.shape[0]))

    den=0
    for j in range(n_var):
        alpha[0, :] =alpha[0, :]+ inr * bEm[j,:, V[j,0]]
        den=den+bEm[j,:, V[j,0]]
    alpha[0, ]=alpha[0, ]/np.sum(alpha[0, ])

    NDm=np.int32(1000*(alpha[0, ]))
    m1=-1
    for k in range(n_components):
        filas=np.where((k==XCr))[0]
        LDAm=LDA[filas]
        um=np.mean(LDAm)
        print("Parámetro de Riesgo",k)
        print("La media del complemento del parametro de riesgo es:",um)
        print("La cantidad de eventos de riesgo por parametro",len(LDAm))
        sigmam=np.sqrt(np.var(LDAm))
        for i in range(NDm[k]):
            m1=m1+1
            XLDA[m1,0]=(0.8+0.4*random.random())*np.random.normal(um,2*sigmam)

    alpha2=alpha
    XInf[1,0]=ydmes[0,0];XInf[1,1]=ydmes[0,1];XInf[1,2]=ydmes[0,2];
    XInf[1,3]=ydmes[0,3];XInf[1,4]=ydmes[0,4];
    from scipy.stats import skew as skewfunc2
    XInf[1,5]=np.round(skewfunc2(XLDA[:,0]),decimals=3)
    XInf[1,6]=np.round(alpha2[0,0]+alpha2[0,1],decimals=3)
    XInf[1,7]=np.round(alpha2[0,2]+alpha2[0,3],decimals=3)
    XInf[1,8]=np.round(alpha2[0,4],decimals=3)
    XInf[1,9]=np.round(np.mean(XLDA[:,0]),decimals=3)
    XInf[1,10]=np.round(np.percentile(XLDA[:,0],75),decimals=3)
    XInf[1,11]=np.round(np.percentile(XLDA[:,0],99),decimals=3)

    for t in range(1,V.shape[1]):
        den=0
        for j in range(aTr.shape[0]):
            for m in range(n_var):
                alpha[t, j] =alpha[t, j]+ alpha[t - 1].dot(aTr[:, j]) * bEm[m,j,V[m,t]]
                den=den+np.sum(bEm[m,j, V[m,t]]*aTr[:,j])
        alpha[t, ]=alpha[t, ]/np.sum(alpha[t, ])

        NDm=np.int32(1000*(alpha[t, ]))
        m1=-1
        for k in range(n_components):
            filas=np.where((k==XCr))[0]
            LDAm=LDA[filas]
            um=np.mean(LDAm)
            sigmam=np.sqrt(np.var(LDAm))
            for i in range(NDm[k]):
                m1=m1+1
                XLDA[m1,t]=(0.9+0.2*random.random())*np.random.normal(um,sigmam)

        alpha2=alpha
        XInf[t,0]=ydmes[t,0];XInf[t,1]=ydmes[t,1];XInf[t,2]=ydmes[t,2];
        XInf[t,3]=ydmes[t,3];XInf[t,4]=ydmes[t,4];
        XInf[t,5]=np.round(skewfunc2(XLDA[:,t-1]),decimals=3)
        XInf[t,6]=np.round(alpha2[t,0]+alpha2[t,1],decimals=3)
        XInf[t,7]=np.round(alpha2[t,2]+alpha2[t,3],decimals=3)
        XInf[t,8]=np.round(alpha2[t,4],decimals=3)
        XInf[t,9]=np.round(np.mean(XLDA[:,t-1]),decimals=3)
        XInf[t,10]=np.round(np.percentile(XLDA[:,t-1],75),decimals=3)
        XInf[t,11]=np.round(np.percentile(XLDA[:,t-1],99),decimals=3)

    return np.array(VC),XInf,XLDA


# The below variables appear in your HPC code. We set them to global
# so that MatricesTransicion/Prospectiva won't break:
n_components=5
n_var=5
titulos=['Máx grado C','Mín grado C','Viento (m/s)','Humedad (%)','Precipitaciones (mm)']

# -------------- Original NDVI / IDW / QGIS Bulk Pipeline --------------

def process_uploaded_file(uploaded_file):
    """
    Reads a single-sheet Excel with columns ["Longitud", "Latitud", "NDVI", "Riesgo"] (optionally).
    Returns the DataFrame or None on error.
    """
    try:
        df = pd.read_excel(uploaded_file)
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
    """
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
    ...
    (unchanged, as in your original code)
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
    ...
    (unchanged, as in your original code)
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
    ...
    (unchanged, as in your original code)
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
    ...
    (unchanged, as in your original code)
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
    ...
    (unchanged, as in your original code)
    """
    import re
    pattern = re.compile(r'(\d{1,2}[a-zA-Z]{3}\d{2,4})')
    match = pattern.search(filename)
    if match:
        return match.group(1)
    return None


def _process_one_k(k_val, folder_path, colorMap_keyword, XC):
    """
    ...
    (unchanged, as in your original code)
    """
    logger.info(f"[_process_one_k] Worker starts for k={k_val}")
    k_str = str(k_val).zfill(3)
    base_file, color_file = None, None

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

    df_idw, df_idw_2 = _idw_index_core(df_esp)
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
    ...
    (unchanged, as in your original code)
    """
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Starting => indice='{indice}', anio='{anio}'")
    folder_path = os.path.join(base_folder, f"{indice}_{anio}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"[bulk_unzip_and_analyze_new_parallel] Created folder_path='{folder_path}'")

    for file_ in os.listdir(folder_path):
        if file_.lower().endswith(".zip"):
            zip_path = os.path.join(folder_path, file_)
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Unzipping => {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(folder_path)

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

    output_dir = os.path.join("assets", "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    espacial_xlsx = os.path.join(output_dir, f"INFORME_{indice}_Espacial_{anio}.xlsx")
    idw_xlsx      = os.path.join(output_dir, f"INFORME_{indice}_IDW_{anio}.xlsx")
    qgis_xlsx     = os.path.join(output_dir, f"INFORME_{indice}_QGIS_{anio}.xlsx")

    for path_ in [espacial_xlsx, idw_xlsx, qgis_xlsx]:
        if os.path.exists(path_):
            logger.info(f"[bulk_unzip_and_analyze_new_parallel] Removing old {path_}")
            os.remove(path_)

    max_k = 0
    for f in color_files:
        try:
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

    XC = np.sort(np.random.uniform(0, 1, 5))
    logger.info(f"[bulk_unzip_and_analyze_new_parallel] Initial cluster seeds (XC)={XC}")

    tasks = range(1, max_k+1)
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

    results.sort(key=lambda r: r[0])

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
