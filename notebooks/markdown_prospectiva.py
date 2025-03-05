# %% [markdown]
# 0. Se procede con la instalación de las librerias de trabajo

# %%
!pip install tensorflow

# %% [markdown]
# 1. Se cargan las librerias de trabajo

# %%
#Se cargan las librerías de trabajo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import skew
import openpyxl
import random

#Para cargar los archivos automáticamente del drive
from google.colab import drive
drive.mount('/content/drive')

#Se procede con el proceso de clusterización
from sklearn.cluster import KMeans

#Se procede con el pronóstico de las variables meteorológicas
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2,L1,L1L2
from keras.callbacks import EarlyStopping

#Se eliminan los warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# **Prospectiva Paramétros de Riesgo**
# 
# 1. Se procede con el pronóstico del patrón prospectivo de las variables agroclimáticas
# *   Modelo neuronal del riesgo
# *   Aquí se hace el pronóstico de cada una de las variables
# *  Para el caso de las variables climáticas de entrada se les calcula los paramétros de riesgo mes a mes (solo 12 datos).
# *  Para el caso de la variable de salida se hace un total de 360 datos (año siguiente)
# 

# %%
def Emision(i2,XDe,NIT,NR):

  #i2: Indica la variable climática (parámetro) a analizar.
  #XDe: Datos de Entrada - Aquí van solo las variables climáticas
  #NIT: Número de Iteraciones
  #NR: Número de Retardos Mensuales

  #''''''''''''''''''''''''''''''''''''''''''''''
  #1. Se procede con la construcción de los Datos
  #''''''''''''''''''''''''''''''''''''''''''''''
  npr=30 #Número de valores de pronóstico adelante
  XDst=np.zeros((len(XDe)-30*(NR-1),NR))
  XDen = (XDe - XDe.min()) / (XDe.max() - XDe.min())
  XDe2=XDe
  XDe=XDen

  for k in range(NR):
    for i in range(len(XDe)-30*(NR-1)):
      XDst[i,k]=XDe[i+k*30,]

  Vmax=np.max(XDst[:,0])

  #Se procede con la construcción dle vector de salida
  ydst=np.zeros((len(XDe)-30*(NR-1)))

  for i in range(len(XDe)-(360+30*(NR-1))):
    ydst[i]=XDe[i+(360+30*(NR-1)),]

  dfst=pd.DataFrame(np.column_stack((XDst,ydst)))

  #''''''''''''''''''''''''''''''''''''''''''''''''''''''
  #2. Se procede con la configuración del modelo neuronal
  #''''''''''''''''''''''''''''''''''''''''''''''''''''''
  XDn=XDst[:-360,:]
  ydn=ydst[:-360]

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
  history=model.fit(XDn,ydn,epochs=NIT,batch_size=250,callbacks=[early_stopping],verbose=0)
  yr=model.predict(XDn)

  plt.figure()
  plt.plot(ydn,'r',yr,'b')
  plt.show()

  #'''''''''''''''''''''''''''''''''''''''''''''''''''
  #3. Se procede con el pronóstico para el próximo año
  #'''''''''''''''''''''''''''''''''''''''''''''''''''
  XDpn=XDst[-360:,:]
  yp=model.predict(XDpn)*((XDe2.max()-XDe2.min())+XDe2.min())

  #3.1 Se buscan los percentiles para los datos de pronóstico (yp) por percentil
  XC1p=np.percentile(yp,[10,20,30,40,50,60,70,80,90,100])

  #3.2 Se hace la clasificación del pronóstico (yp) por nivel de riesgo (1,2,3,4,5)
  Vp1=np.zeros((12,1));Vp2=np.zeros((12,1))

  for i in range(12):
    d2=np.sqrt((XC1p-yp[30*(i+1)-1,])**2)
    Vp1[i]=(np.where(d2==np.min(d2))[0])[0]

    if int(Vp1[i])<5:
       Vp2[i]=4-int(Vp1[i])
    else:
       Vp2[i]=int(Vp1[i])-5

  return np.array(Vp1),np.array(Vp2),yp,XC1p

# %% [markdown]
# 3. Se procede con la construcción de las matrices de emisión y transición del riesgo

# %%
def MatricesTransicion(XD,XD3,n_var,punto):

  #Esto permite hacer la prospectiva frente a la variable de salida
  #XCr: Matriz de los niveles de riesgo
  #XD: Matriz con los datos climático
  #n_var: Indica el número de variables climáticas
  #n_components: Indica los clusters de agrupación

  #'''''''''''''''''''''''''''''''''''''''''''''
  #0. Aquí vamos a leer el punto que se necesita
  #'''''''''''''''''''''''''''''''''''''''''''''
  LDA=[]

  for i in range(len(XD3)):
      LDA.append(XD3[i,punto,3])

  lonp=XD3[0,punto,1]
  latp=XD3[0,punto,2]
  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  #1. Se procede con la construccción de la matriz de transición del riesgo
  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  seed = np.random.randint(0, 1000)  #Generar una semilla aleatoria
  LDA=np.array(XD.iloc[:,n_var]); LDA=1-LDA
  mkm=KMeans(n_clusters=n_components,random_state=seed)
  mkm.fit(LDA.reshape(-1,1))

  #Se pro
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
    MTr[int(fila),int(col)]=MTr[int(fila),int(col)]+1

  MTr_sf=np.sum(MTr,axis=1)

  for i in range(n_components):
    for j in range(n_components):
      MTr[i,j]=MTr[i,j]/MTr_sf[i]

  print("La matriz de transición del riesgo del riesgo es:\n",MTr)
  a=MTr

  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  #2. Se procede con la construcción de las matrices de emisión
  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  WD=np.zeros((len(LDA),n_var))
  XC2=np.zeros((2*n_components,n_var))
  mkm=KMeans(n_clusters=2*n_components,random_state=seed)

  #2.1 Se procede a incrementar la variedad de los datos para la temperatura
  for j in range(n_var):
    WD[:,j]=np.array(XD.iloc[:,j])

    if j==0 or j==1:
      for k in range(len(WD)):
        WD[k,j]=(0.9+0.2*random.random())*WD[k,j]

    mkm.fit(WD[:,j].reshape(-1,1))
    XC2[:,j]=np.array(sorted(mkm.cluster_centers_,reverse=False)).reshape(1,2*n_components)

  #2.2 Se determinan los niveles de clasificación or variable
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

# %% [markdown]
# 4. Se procede con la prospectiva del riesgo

# %%
def Prospectiva(i1,XD,XCr,V,aTr,bEm,ydmes):

  #i1: Indica el punto de análisis para la rejilla
  #V: Indica el vector de prospectiva del riesgo
  #aTr: Matriz de transición del riesgo
  #bEm: Matriz de emisión dle riesgo
  #XCr: Es vector de sanidad vegetal para un punto específico

  #'''''''''''''''''''''''''''''''''''''''''''''''''''
  #1. Se determina la estructura porcentual del riesgo
  #'''''''''''''''''''''''''''''''''''''''''''''''''''
  #Se toma la matriz de las pérdidas
  LDA=np.array(XD.iloc[:,n_var]); LDA=1-LDA

  #Inercia de las categorías del riesgo
  nC=np.zeros((n_components,1))
  inr=np.zeros((n_components,))

  for j in range(n_var):
    nC[j]=len(np.where(XCr==j)[0])
    nC[j]=nC[j]/len(LDA)
    inr[j]=nC[j]

  print("La estructura porcentual del riesgo es:\n",nC)

  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''
  #2. Patrón de evolución de las variables observables
  #'''''''''''''''''''''''''''''''''''''''''''''''''''''''
  XLDA=np.zeros((1000,V.shape[1]))
  XInf=np.zeros((V.shape[1],12))

  #2.1 Estado Actual
  LDA = LDA.astype(np.float64)
  XInf[0,0]=ydmes[0,0];XInf[0,1]=ydmes[0,1];XInf[0,2]=ydmes[0,2];
  XInf[0,3]=ydmes[0,3];XInf[0,4]=ydmes[0,4];
  XInf[0,5]=np.round(skew(LDA),decimals=3)
  XInf[0,6]=np.round(nC[0]+nC[1],decimals=3)  #Pérdidas Esperadas
  XInf[0,7]=np.round(nC[2]+nC[3],decimals=3)  #Pérdidas No Esperadas
  XInf[0,8]=np.round(nC[4],decimals=3)
  XInf[0,9]=np.round(np.mean(LDA),decimals=3)
  XInf[0,10]=np.round(np.percentile(LDA,75),decimals=3)
  XInf[0,11]=np.round(np.percentile(LDA,99),decimals=3)

  #Patrón cualitativo - Caracterización Variables Observables
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
  XInf[1,5]=np.round(skew(XLDA[:,0]),decimals=3)
  XInf[1,6]=np.round(alpha2[0,0]+alpha2[0,1],decimals=3)
  XInf[1,7]=np.round(alpha2[0,2]+alpha2[0,3],decimals=3)
  XInf[1,8]=np.round(alpha2[0,4],decimals=3)
  XInf[1,9]=np.round(np.mean(XLDA[:,0]),decimals=3)
  XInf[1,10]=np.round(np.percentile(XLDA[:,0],75),decimals=3)
  XInf[1,11]=np.round(np.percentile(XLDA[:,0],99),decimals=3)

  #''''''''''''''''''''''''''''''''''''''''''''''
  #3. Evolución del Riesgo Prospectiva del Riesgo
  #''''''''''''''''''''''''''''''''''''''''''''''
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
    XInf[t,5]=np.round(skew(XLDA[:,t-1]),decimals=3)
    XInf[t,6]=np.round(alpha2[t,0]+alpha2[t,1],decimals=3)
    XInf[t,7]=np.round(alpha2[t,2]+alpha2[t,3],decimals=3);
    XInf[t,8]=np.round(alpha2[t,4],decimals=3)
    XInf[t,9]=np.round(np.mean(XLDA[:,t-1]),decimals=3)
    XInf[t,10]=np.round(np.percentile(XLDA[:,t-1],75),decimals=3)
    XInf[t,11]=np.round(np.percentile(XLDA[:,t-1],99),decimals=3)

  VC=np.array(VC)

  return VC,XInf,XLDA

# %% [markdown]
# #**Evolución Espacial del Riesgo**
# 0. Se procede con la organización del archivo del clima
# * Este procedimiento permite la organización temporal del archivo climático.

# %%
#1. Se procede con la selección de la carpeta de trabajo
indice=input("Ingresar el nombre del indice de vegetación:")
año=input("Ingresar el año de análisis:")

nxl='/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Clima_'+indice+'_'+año+'.xlsx'
print(nxl)

#2. Se procede con la lectura de los datos climáticos
dfc = pd.read_excel(nxl, sheet_name=0)

#2.1 Invertir el orden de las filas
df_invertido = dfc.iloc[::-1]

ruta_salida = 'Clima_'+indice+'_'+año+'_O.xlsx'  # Cambia esto por la ruta de salida
nxl2='/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/'+ruta_salida
df_invertido.to_excel(nxl2, index=False, sheet_name='Sheet 1')

print(f"El archivo con las filas invertidas se guardó en: {nxl2}")
df_invertido.to_excel(ruta_salida, index=False, sheet_name='Sheet 1')

print(f"El archivo con las filas invertidas se guardó en: {ruta_salida}")

# %% [markdown]
# 1. Se procede con la construcción del patrón prospectivo para las variables agroclimáticas
# 

# %%
#1. Se procede con la selección del archivo de trabajo
indice=input("Ingresar el nombre del indice de vegetación: ")
año=input("Ingresar el año de análisis: ")

nxl='/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/'+'Clima_'+indice+'_'+año+'_O.xlsx'
ruta_destino=nxl

XDB = pd.read_excel(nxl, sheet_name=0)
XDB2 = pd.read_excel(nxl, sheet_name=0)
XDB = XDB.dropna()

#2. Se determinan las filas que poseen datos
XDB=XDB[XDB['Fuente de datos']!='-']
columnas_interes=[7,8,11,12,13,4]
XD = XDB.iloc[:, columnas_interes]
XD2 = XDB2.iloc[:, columnas_interes]
titulos=['Máx grado C','Mín grado C','Viento (m/s)','Humedad (%)','Precipitaciones (mm)']

#3. Se crea el informe de salida solo con las fechas en las cuales se tomaron las imagenes satelitales
impath2='/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/'
XDB.to_excel(impath2+"Clima_"+indice+"_"+año+"_HMM"+".xlsx")

#4. Parámetros de configuración
n_components=5  #Niveles de clasificación del riesgo
n_var=5         #Variables agroclimáticas de entrada
print("El número de parametros de riesgo es:",n_components)
print("El número de variables agroclimáticas es:",n_var)

# %% [markdown]
# 3. Se crea el patrón de evolución para el año siguiente

# %%
#1. Se configuran los valores de referencia para cada variable (de deciles a quintiles)
Vref=np.array([0,1,2,3,4,5,6,7,8,9])
XCpar = np.zeros((5, 10))
Vnref=np.array([4,3,2,1,0,0,1,2,3,4])

#2. Se crean los niveles climáticos para la evolución prospectiva del riesgo
V = np.random.randint(0, 5, size=(n_var, 12))
Vn = np.random.randint(0, 5, size=(n_var, 12))
mes=np.array([1,2,3,4,5,6,7,8,9,10,11,12])

#3. Se presenta la construcción de los patrones climáticos
ydpar=np.zeros((360,n_var))

for i in range(n_var):
  XDe=np.array(XD2.iloc[:,i])
  NNEm=(Emision(i,XDe,500,4))
  print(f"La Evolución de la Variable {titulos[i]} es:")
  print("Los niveles  del riesgo son:")
  print(Vref)
  print(f"Los cluster de referencia para la Variable {titulos[i]} son:")
  XCpar[i,:]=NNEm[3]
  print(XCpar[i,:])
  print("Los niveles absolutos del riesgo son:")
  Vn[i,:]=NNEm[0].reshape(1,12)
  print(Vn[i,:])
  print("Los niveles relativos del riesgo son:")
  V[i,:]=NNEm[1].reshape(1,12)
  print(V[i,:])

  #4. Se procede con el pronóstico de
  ydpar[:,i]=NNEm[2].reshape(360,)
  #print(ydpar[:,i])

#5. Se procde con al configuraciónde los informes del riesgo
dfpar=pd.DataFrame(np.column_stack((Vref,Vnref,XCpar.transpose())))
dfpar.columns=['Niveles','Parametros','Max C','Min  C','Viento (m/s)','Humedad (%)','Precip. (mm)']
dfpar.head(10)

#Se procede con la presentación de los datos de pronóstico mes a mes
ydmes=np.zeros((12,5))

for i in range(12):
  ydmes[i,:]=ydpar[30*(i)-1,:]
  print(ydmes[i,:])

dfevol=pd.DataFrame(np.column_stack((mes,ydmes)))
dfevol.columns=['Mes'+str(int(año)+1),'Max C','Min  C','Viento (m/s)','Humedad (%)','Precip. (mm)']
dfevol.head(10)

#Se crea el archivo con la prospectiva
ruta_excel = '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Prospective_'+indice+'_'+str(int(año)+1)+'.xlsx'

with pd.ExcelWriter(ruta_excel, engine='openpyxl') as writer:
    dfpar.to_excel(writer, index=False, sheet_name=str('Clusters'))

# Ruta del archivo Excel
ruta_excel = '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Prospective_'+indice+'_'+str(int(año)+1)+'.xlsx'

with pd.ExcelWriter(ruta_excel, mode='a', engine='openpyxl') as writer:
   dfevol.to_excel(writer, index=False, sheet_name=str('Prospectiva'+str(int(año)+1)))


# %% [markdown]
# 2. Se procede con la selección de cada uno de los puntos espaciales para el análisis

# %%
#Se procede con la carga de la información
res=5 #Resolución de la rejilla de trabajo
nxl3='/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Informe '+indice+' QGIS '+año+'.xlsx'
libro = openpyxl.load_workbook(nxl3)
hojas = libro.sheetnames

XDB3 = pd.read_excel(nxl3, sheet_name=None)
array_hojas = list(XDB3.values())
XD3=np.empty((len(XDB3),5*5,5))

for i, hoja in enumerate(array_hojas):
      XD3[i,:,:]=hoja

#Es importante determinar la resolución espacial de análisis
res=5
MTr2=np.zeros((res**2,n_components,n_components))  #Matriz transición del riesgo
MTwd2=np.zeros((res**2,n_var,n_components,n_components)) #Matrices de Emisión

#Se procede con el ingreso de la pérdida por hectarea por cultivo
PRef=np.float128(input("Ingresar productividad de referencia (USD/Ha):"))
Ef=0.70
PRef=PRef*(1-Ef)

for i1 in range(res*res):
  print(i1)

  #Se crea el vector de riesgo
  punto=i1

  #Se procede con la obtención de las matrices de Transición y Emisión
  MTrr=MatricesTransicion(XD,XD3,n_var,punto)
  MTr2[i1,:,:]=MTrr[0]
  MTwd2[i1,:,:,:]=MTrr[1]
  XCr=MTrr[2]
  lonp=XD3[0,i1,1];latp=XD3[0,i1,2]

  #Se procede a mostrar la información de la zona de estudio
  infm=[i1,lonp,latp]
  dfm2=pd.DataFrame(np.column_stack((infm)))
  dfm2.columns=['Punto','Longitud','Latitud']
  display(dfm2)

  #Se procede con la impresión de las matrices de riesgo
  print(f"La matriz de transición del riesgo del punto {i1} es:\n",MTrr[0])

  for j1 in range(n_var):
    print(f"La matriz de emisión para la variable {titulos[j1]} es:\n",np.round(MTwd2[i1,j1,:,:],decimals=3))

  #Se procede con la evolución prospectiva del riesgo
  aTr=MTr2[i1,:,:];bEm=MTwd2[i1,:,:,:]; NDVI=np.zeros((12,1))

  dfpr=Prospectiva(i1,XD,XCr,V,aTr,bEm,ydmes)
  VC=dfpr[0]
  XInf2=dfpr[1]
  XLDA=dfpr[2]

  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  #4. Distribuciones de Pérdidas - Periodo Prospectiva (12 Meses)
  #''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
  plt.figure()

  for k in range(1,V.shape[1]):
      XLDA[:,k]=XLDA[:,k]*PRef
      sns.kdeplot(XLDA[:,k], shade=True,bw=0.5)

  #Se procede con la gráfica de la distribución de pérdidas
  plt.legend(labels=VC[-10:,])
  plt.title('Monthly Risk Evolution Punto '+str(i1))
  plt.xlabel('Losses (USD/Month-Zone)')
  plt.ylabel('Density')
  plt.grid(True)
  plt.show()


  #Para la configuración del informe
  NDVI=XInf2[:,11].copy()
  MPerd=XInf2[:,9:12]

  for i2 in range(len(MPerd)):
    for j2 in range(MPerd.shape[1]):
      MPerd[i2,j2]=PRef/np.abs(1-MPerd[i2,j2])

  dfm=pd.DataFrame(np.column_stack((VC,XInf2[:,0:5],NDVI,XInf2[:,5],XInf2[:,6:9],MPerd[:,:])))
  dfm.columns=['WD','Max C','Min  C','Viento (m/s)','Humedad (%)','Precip. (mm)',str(indice),'Skewness','%C1','%C2','%C3','Mean (USD)','75% (USD)','OpVar-99.9% (USD)']
  dfm['Max C']=dfm['Max C'].astype(float).map('{:.3f}'.format)
  dfm['Min  C']=dfm['Min  C'].astype(float).map('{:.3f}'.format)
  dfm['Viento (m/s)']=dfm['Viento (m/s)'].astype(float).map('{:.3f}'.format)
  dfm['Humedad (%)']=dfm['Humedad (%)'].astype(float).map('{:.3f}'.format)
  dfm['Precip. (mm)']=dfm['Precip. (mm)'].astype(float).map('{:.3f}'.format)
  dfm['Skewness']=dfm['Skewness'].astype(float).map('{:.4f}'.format)
  dfm[str(indice)]=dfm[str(indice)].astype(float).map('{:.4f}'.format)
  dfm['%C1']=dfm['%C1'].astype(float).map('{:.3f}'.format)
  dfm['%C2']=dfm['%C2'].astype(float).map('{:.3f}'.format)
  dfm['%C3']=dfm['%C3'].astype(float).map('{:.3f}'.format)
  dfm['Mean (USD)']=dfm['Mean (USD)'].astype(float).map('{:.2f}'.format)
  dfm['75% (USD)']=dfm['75% (USD)'].astype(float).map('{:.2f}'.format)
  dfm['OpVar-99.9% (USD)']=dfm['OpVar-99.9% (USD)'].astype(float).map('{:.2f}'.format)

  display(dfm)

  # Se almacenan los informes de evolución del riesgo
  ruta_excel = '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Prospective_'+indice+'_'+str(int(año)+1)+'.xlsx'

  with pd.ExcelWriter(ruta_excel, mode='a', engine='openpyxl',if_sheet_exists="replace") as writer:
          dfm.to_excel(writer, index=False, sheet_name=str('Point_'+str(i1)))

  #Se procede con el almacenamiento de las distribuciones de pérdidas
  ruta_excel_2 = '/content/drive/MyDrive/Software-EAFIT-DMU/Software_Puerta/'+indice+'_'+año+'/Prospective_LDA_'+indice+'_'+str(int(año)+1)+'.xlsx'
  titulos_LDA=[]

  if i1==0:

    titulos_LDA=[]

    for i2 in range(12):
      titulos_LDA.append('Mes' + str(i2))

    titulos_LDA2=titulos_LDA
    dfXLDA=pd.DataFrame(XLDA)
    dfXLDA.columns=titulos_LDA

    with pd.ExcelWriter(ruta_excel_2, engine='openpyxl') as writer:
      dfXLDA.to_excel(writer, index=False, sheet_name=str('LDA Point_'+str(int(i1))))

  else:

    dfXLDA=pd.DataFrame(XLDA)
    dfXLDA.columns=titulos_LDA2

    with pd.ExcelWriter(ruta_excel_2, mode='a', engine='openpyxl') as writer:
      dfXLDA.to_excel(writer, index=False, sheet_name=str('LDA Point_'+str(int(i1))))


