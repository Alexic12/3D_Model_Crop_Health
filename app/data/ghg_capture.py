import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import skew
import os
import random
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import requests


def get_usd_cop_rate():
    """Get current USD to COP exchange rate"""
    try:
        # Using a free API for exchange rates
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = response.json()
        return data['rates'].get('COP', 4200)  # Default to 4200 if API fails
    except:
        return 4200  # Default exchange rate

def ClusteringX(XDi, NCi):
    """Characterize the random variable of frequency"""
    mkm = KMeans(n_clusters=NCi, random_state=42)
    mkm.fit(XDi.reshape(-1, 1))
    XCx = np.array(sorted(mkm.cluster_centers_.flatten()))
    
    # Cluster diameters
    sigmax = np.zeros((NCi, 1))
    for j in range(NCi):
        sigmax[j,] = np.sum(np.abs(XCx - XCx[j])) / 4
    
    return XCx, sigmax.T


def MatriX(MORi, lbfi, lbsi, namei):
    """Create risk matrix visualization"""
    MORi_ordenada = MORi[::-1, :]  # Vertical inversion
    lbf_ordenadax = lbfi[:, ::-1]
    lbs_ordenadax = lbsi[::-1]
    
    return MORi_ordenada, lbf_ordenadax, lbs_ordenadax


def Fuzzyx(Xf2, sigmaf, Xs2, sigmas, XCf, XCs, ME, MI, MG, MP):
    """Fuzzy logic calculation for risk assessment"""
    # Simplified implementation - you may need to adjust based on your specific logic
    LDA = []
    for k in range(len(Xf2)):
        VPf = np.exp(-0.5 * ((XCf - Xf2[k]) / sigmaf) ** 2)
        nf = np.argmax(VPf, axis=1)
        VPs = np.exp(-0.5 * ((XCs - Xs2[k]) / sigmas) ** 2)
        ns = np.argmax(VPs, axis=1)
        
        # Calculate loss distribution aggregate
        lda_value = (ME[nf, ns] * MP[nf, ns] * MI[nf, ns]) / MG[nf, ns] if MG[nf, ns] != 0 else 0
        LDA.append(lda_value)
    
    return np.array(LDA)


def process_ghg_data(indice, anio, base_folder="./upload_data", field_name=None):
    """Main GHG capture processing function"""
    try:
        # Extract field name from base_folder if not provided
        if field_name is None:
            field_name = os.path.basename(base_folder)
        
        # File path construction - prioritize field-based structure
        possible_paths = [
            os.path.join("assets", "data", field_name, f"INFORME_{indice}_Espacial_{anio}.xlsx"),
            os.path.join(base_folder, field_name, indice, anio, f"INFORME_{indice}_Espacial_{anio}.xlsx"),
            os.path.join("assets", "data", f"INFORME_{indice}_Espacial_{anio}.xlsx"),
            os.path.join(base_folder, indice, anio, f"INFORME_{indice}_Espacial_{anio}.xlsx")
        ]
        
        nxl = None
        for path in possible_paths:
            if os.path.exists(path):
                nxl = path
                break
        
        if nxl is None:
            st.error(f"File not found in any of these locations: {possible_paths}")
            # Debug: show what files actually exist
            assets_dir = os.path.join("assets", "data")
            if os.path.exists(assets_dir):
                files = os.listdir(assets_dir)
                st.info(f"Files in assets/data: {files}")
            return None
        
        st.success(f"Found file: {nxl}")
            
        # Read Excel file
        excel = pd.ExcelFile(nxl)
        hojas = pd.read_excel(nxl, sheet_name=None)
        
        # Productivity parameters
        Ing = 888    # USD/ha-month
        NHa = 8.7    # Number of hectares
        CNG = np.array([2, 5, 9, 14, 20])  # Cost Ha-Month - Management Level
        
        NIx = []
        # Find maximum and minimum values
        for nombre, df in hojas.items():
            XD = df.iloc[:, 6]
            # Remove NaN values
            XD_clean = XD.dropna()
            if len(XD_clean) > 0:
                NIx = np.concatenate((NIx, XD_clean))
        
        NIxm = np.mean(NIx)
        NIxstd = np.std(NIx)
        NIx_max = np.max(NIx)
        NIx_min = np.min(NIx)
        
        # Clustering - ensure no NaN values
        NIx = NIx[~np.isnan(NIx)]  # Remove any remaining NaN
        if len(NIx) == 0:
            st.error("No valid data found after removing NaN values")
            return None
            
        NC = 5
        mkm = KMeans(n_clusters=NC, random_state=42)
        mkm.fit(NIx.reshape(-1, 1))
        XC = mkm.cluster_centers_
        XCo = sorted(XC)
        
        # Process vegetation index clustering
        m1 = 0
        XDOR = np.zeros((1, 3))
        
        for nombre, df in hojas.items():
            df['Risk'] = None
            XD = df.iloc[:, 6]
            
            # Clean data - remove NaN values
            valid_indices = ~pd.isna(XD)
            XD_clean = XD[valid_indices]
            
            if len(XD_clean) == 0:
                continue  # Skip sheets with no valid data
                
            NPuntos = int(len(XD_clean))
            Ingp = (Ing * NHa) / NPuntos
            CNGp = np.array(CNG * NHa) / NPuntos
            
            # Estimate labels only for valid data
            for idx in XD_clean.index:
                df.loc[idx, 'Risk'] = np.argmin(np.abs(XCo - XD_clean[idx]))
            
            # Count data - only use valid entries
            risk_col = df['Risk'].dropna()
            for j in range(5):
                m1 += 1
                matching_risks = risk_col[risk_col == j]
                Xf = len(matching_risks)
                
                if Xf > 0:
                    # Get corresponding XD values for this risk level
                    risk_indices = risk_col[risk_col == j].index
                    Xs = np.mean(XD_clean[risk_indices])
                else:
                    Xs = 0  # Default value when no data
                
                if indice == 'RECI':
                    Xs = -Xs * Ingp
                else:
                    Xs = (1 - Xs) * Ingp
                
                XDOR = np.vstack((XDOR, [m1, Xf, Xs]))
        
        XDOR = np.delete(XDOR, 0, axis=0)
        
        # Create clustering for frequency and severity
        Xf = XDOR[:, 1]
        Xs = XDOR[:, 2]
        
        XCf, sigmaf = ClusteringX(Xf, NC)
        XCs, sigmas = ClusteringX(Xs, NC)
        
        # Labels
        lbf = np.array([['Muy Pocos', 'Pocos', 'Más o Menos', 'Muchos', 'Bastantes']])
        lbs = np.array([['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']])
        
        # Create matrices
        ME = np.zeros((5, 5))  # Events matrix
        MP = np.zeros((5, 5))  # Losses matrix
        
        for k in range(len(Xf)):
            VPf = np.exp(-0.5 * ((XCf - Xf[k]) / sigmaf) ** 2)
            nf = np.argmax(VPf, axis=1)
            VPs = np.exp(-0.5 * ((XCs - Xs[k]) / sigmas) ** 2)
            ns = np.argmax(VPs, axis=1)
            #ME[nf, ns] += Xf[k]
            ME[nf, ns] = (ME[nf, ns] + Xf[k]) / 2
            MP[nf, ns] = (MP[nf, ns] + Xs[k]) / 2
        
        ME[ME[:, :] == 0] = 1
        
        # Impact matrix
        MI = np.array([[1, 1, 1, 2, 2],
                       [1, 2, 2, 3, 3],
                       [1, 2, 3, 3, 4],
                       [2, 3, 3, 4, 4],
                       [2, 3, 4, 4, 5]])
        
        # Management cost matrix
        MCG = MI.astype(float).copy()
        for i in range(5):
            for j in range(5):
                if MI[i, j] == 1:
                    MCG[i, j] = (0.9 + 0.2 * random.random()) * CNG[0]
                elif MI[i, j] == 2:
                    MCG[i, j] = (0.9 + 0.2 * random.random()) * CNG[1]
                elif MI[i, j] == 3:
                    MCG[i, j] = (0.9 + 0.2 * random.random()) * CNG[2]
                elif MI[i, j] == 4:
                    MCG[i, j] = (0.9 + 0.2 * random.random()) * CNG[3]
                elif MI[i, j] == 5:
                    MCG[i, j] = (0.9 + 0.2 * random.random()) * CNG[4]
        
        # Generate management scenarios with proper sampling (from Jupyter notebook)
        nh = len(hojas.items())  # Number of satellite images
        LDAT = []
        NDm = len(Xf)  # Number of sampling data
        
        # Sample frequency and severity data
        Xfm = pd.DataFrame(Xf).sample(n=NDm, replace=True)
        Xf2 = np.array(Xfm)
        Xsm = pd.DataFrame(Xs).sample(n=NDm, replace=True) 
        Xs2 = np.array(Xsm)
        LDAo = Fuzzyx(Xf2, sigmaf, Xs2, sigmas, XCf, XCs, ME, MI, MI, MP)
        
        # Determine risk levels using KMeans clustering
        np.random.seed(42)
        k = 5
        mkm = KMeans(n_clusters=5, init='random', random_state=42)
        mkm.fit(LDAo.reshape(-1, 1))
        XC = sorted(mkm.cluster_centers_.flatten())
        MIo = np.zeros((len(LDAo), 1))
        MIo_G = np.zeros((len(LDAo), 1))
        
        for k in range(len(LDAo)):
            MIo[k, ] = np.argmin(np.abs(XC - LDAo[k])) + 1
        
        # Baseline LDAo characterization
        CNG = np.array([2, 5, 9, 14, 20])  # Management costs by risk level
        muo = np.mean(LDAo)
        OpVaro = np.percentile(LDAo, 99.9)
        cas_o = skew(LDAo.ravel(), bias=False)
        PNEo = np.mean(LDAo[(LDAo >= muo) & (LDAo < OpVaro)])
        LDAT.append(LDAo)
        LDAT = np.array(LDAT)
        LDAT = np.array(LDAT.flatten())
        
        NEProm = np.mean(Xsm)  # Average loss cost per event
        filaso = np.where(LDAo <= muo)[0]
        NEo = int((np.sum(Xf2[filaso, ]) / nh) * 12)
        filass = np.where(LDAo >= OpVaro)[0]
        NSo = int((np.sum(Xf2[filass, ]) / nh) * 12)
        filasnne = np.where((LDAo >= muo) & (LDAo < OpVaro))[0]
        NNEo = int((np.sum(Xf2[filasnne, ]) / nh) * 12)
        NVCo = (NNEo - NNEo)
        L75o = np.percentile(LDAo, 75)
        CG = NVCo * np.percentile(Xs2, 25)  # Average management cost
        CP = NVCo * np.percentile(Xs2, 99.9)  # Average loss cost
        VCap = CP - CG
        TCO2 = 0
        
        # Financial costs and income for field management activities
        MCT = (MCG * ME) / len(hojas.items())  # Total annual cost matrix
        CTA = np.sum(MCT)  # Total annual cost - Risk management
        XIng = ((np.sum((0.9 + 0.2 * random.random()) * ME * Ingp) / len(hojas.items()))) * 12
        IngOp = (XIng + VCap) - CTA
        
        MRm = []
        MGR = []
        MGR.append('Baseline')
        MRm.append([muo, PNEo, L75o, OpVaro, cas_o, NEo, NNEo, NSo, NVCo, CG, CP, VCap, CTA, XIng, IngOp, TCO2])
        MRm = np.array(MRm)
        NGL = [1.5, 2, 3, 4]
        
        # Management scenarios with proper fuzzy calculations
        for NG in NGL:
            LDAm = np.zeros((len(LDAo), 1))
            
            Xfm = pd.DataFrame(Xf).sample(n=NDm, replace=True)
            Xf2 = np.array(Xfm)
            Xsm = pd.DataFrame(Xs).sample(n=NDm, replace=True)
            Xs2 = np.array(Xsm)
            LDAo_temp = Fuzzyx(Xf2, sigmaf, Xs2, sigmas, XCf, XCs, ME, MI, MI, MP)
            
            # Modify manageable values
            MG = MI.astype(float).copy()
            MG[(MG > 1) & (MG < 5)] *= NG
            
            name = 'Matriz de Gestión ' + str(NG)
            MG_ordenada, lbf_ordenada, lbs_ordenada = MatriX(MG, lbf, lbs, name)
            
            for k in range(len(LDAo)):
                if 1 < MIo[k, ] < 5:
                    MIo_G[k, ] = NG * MIo[k]
                else:
                    MIo_G[k, ] = MIo[k, ]
                
                LDAm[k, ] = LDAo[k, ] * (MIo[k, ] / MIo_G[k, ])
            
            # Stack LDA data for visualization
            LDAT = np.column_stack((LDAT, LDAm.flatten()))
            
            mu = np.mean(LDAm)
            OpVar = np.percentile(LDAm, 99.9)
            cas_m = skew(LDAm.ravel(), bias=False)
            PNE = np.mean(LDAm[(LDAm >= mu) & (LDAm < OpVar)])
            
            # Determine loss structure
            filaso = np.where(LDAm <= muo)[0]
            NEm = int((np.sum(Xf2[filaso, ]) / nh) * 12)
            filass = np.where(LDAm >= OpVar)[0]
            NSm = int((np.sum(Xf2[filass, ]) / nh) * 12)
            filasnne = np.where((LDAm >= muo) & (LDAm < OpVar))[0]
            NNEm = int((np.sum(Xf2[filasnne, ]) / nh) * 12)
            NVCm = NEm - NEo
            L75 = np.percentile(LDAm, 75)
            CG = NVCm * np.percentile(Xsm, 25)  # Management value cost
            CP = NVCm * np.percentile(Xsm, 99.9)  # Loss value cost
            VCap = CP - CG
            TCO2 = VCap / 250  # CO2 Ton cost
            
            MGR.append('Level ' + str(NG))
            
            # Financial costs and income for field management activities
            MCT = (MCG * MG * ME) / (MI * len(hojas.items()))  # Total management cost matrix
            CTA = np.sum(MCT)
            XIng = (np.sum((0.9 + 0.2 * random.random()) * ME * Ingp) / len(hojas.items())) * 12
            IngOp = (XIng + VCap) - CTA
            
            MRm = np.vstack((MRm, np.array([mu, PNE, L75, OpVar, cas_m, NEm, NNEm, NSm, NVCm, CG, CP, VCap, CTA, XIng, IngOp, TCO2])))
        
        # Store reference values for visualization lines
        media_val_o = MRm[0, 0]  # Baseline mean
        opvar_val_o = MRm[0, 3]  # Baseline OpVar
        media_val_g = MRm[-1, 0]  # Last management level mean
        opvar_val_g = MRm[-1, 3]  # Last management level OpVar
        
        # Create results dataframe
        df_results = pd.DataFrame(MRm)
        df_results.columns = ['Media (USD)', 'PNE (USD)', 'L75%', 'OpVar (USD)', 'C.As.', 'NE', 'NNE', 'NS', 'NVC', 'CG (USD)', 'CP (USD)', 'VCap. (USD)', 'CGT (USD)', 'Ing. (USD)', 'IngOp.(USD)', 'TCO2(Ton.)']
        df_results.index = MGR
        df_results = df_results.round(2)
        
        # Get exchange rate
        usd_cop_rate = get_usd_cop_rate()
        
        # Create cluster info dataframe with COP conversion
        df_clusters = pd.DataFrame(np.vstack((
            lbf, 
            np.round(XCf.reshape(1, 5), 0), 
            np.round(sigmaf, 2), 
            lbs, 
            np.round(XCs.reshape(1, 5) * usd_cop_rate, 0),  # Convert to COP
            np.round(sigmas * usd_cop_rate, 0)  # Convert to COP
        )))
        df_clusters.columns = ['0', '1', '2', '3', '4']
        df_clusters.index = ['Labels', 'Frecuencia', 'Diametros', 'Labels', 'Severidad (COP)', 'Diametros (COP)']
        
        # Store management matrices for visualization
        management_matrices = {}
        for i, NG in enumerate([None] + NGL):  # Include baseline (None) + management levels
            if NG is None:
                MG_viz = MI.copy()
                name = 'Baseline'
            else:
                MG_viz = MI.astype(float).copy()
                MG_viz[(MG_viz > 1) & (MG_viz < 5)] *= NG
                name = f'Level {NG}'
            
            management_matrices[name] = MG_viz
        
        return {
            'clusters': df_clusters,
            'events_matrix': ME,
            'losses_matrix': MP,
            'impact_matrix': MI,
            'management_matrices': management_matrices,
            'results': df_results,
            'lda_data': LDAT,  # Already properly formatted
            'mgr_labels': MGR,
            'frequency_labels': lbf,
            'severity_labels': lbs,
            'usd_cop_rate': usd_cop_rate,
            'visualization_lines': {
                'media_val_o': media_val_o,
                'opvar_val_o': opvar_val_o,
                'media_val_g': media_val_g,
                'opvar_val_g': opvar_val_g
            },
            'raw_data': {
                'Xf': Xf,
                'Xs': Xs,
                'XCf': XCf,
                'XCs': XCs,
                'sigmaf': sigmaf,
                'sigmas': sigmas
            }
        }
        
    except Exception as e:
        st.error(f"Error processing GHG data: {e}")
        return None


def create_risk_matrix_heatmap(matrix, title, freq_labels, sev_labels):
    """Create interactive heatmap for risk matrices"""
    fig = go.Figure(data=go.Heatmap(
        z=matrix[::-1],  # Flip vertically
        x=sev_labels.flatten(),
        y=freq_labels[:, ::-1].flatten(),
        colorscale='RdYlBu_r',
        showscale=True,
        hoverongaps=False,
        text=matrix[::-1],  # Show numeric values on the matrix
        texttemplate="%{text:.1f}",  # Format numbers to 1 decimal place
        textfont={"size": 10}  # Set text size for readability
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Severidad",
        yaxis_title="Frecuencia",
        font=dict(size=12),
        height=400
    )
    
    return fig


def create_lda_distribution_plot(lda_data, mgr_labels, visualization_lines=None):
    """Create LDA distribution plot matching Jupyter notebook style with reference lines"""
    fig = go.Figure()
    
    colors = ['red', 'orange', 'yellow', 'blue', 'green']
    
    # Handle both 1D and 2D LDAT arrays
    if lda_data.ndim == 1:
        lda_data = lda_data.reshape(-1, 1)
    
    for i in range(min(lda_data.shape[1], len(mgr_labels))):
        data = lda_data[:, i]
        label = mgr_labels[i]
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Histogram(
            x=data,
            name=label,
            opacity=0.7,
            marker_color=color,
            histnorm='probability density',
            nbinsx=50
        ))
    
    # Add vertical reference lines if provided
    if visualization_lines:
        # Baseline lines (red)
        fig.add_vline(x=visualization_lines['media_val_o'], 
                     line_dash="dash", line_color="red", line_width=1.5,
                     annotation_text="Media_O")
        fig.add_vline(x=visualization_lines['opvar_val_o'], 
                     line_dash="dot", line_color="red", line_width=1.5,
                     annotation_text="OpVar_O")
        
        # Management lines (green)
        fig.add_vline(x=visualization_lines['media_val_g'], 
                     line_dash="dash", line_color="green", line_width=1.5,
                     annotation_text="Media_G")
        fig.add_vline(x=visualization_lines['opvar_val_g'], 
                     line_dash="dot", line_color="green", line_width=1.5,
                     annotation_text="OpVar_G")
    
    fig.update_layout(
        title="Risk Profile (LDA)",
        xaxis_title="USD",
        yaxis_title="Density",
        barmode='overlay',
        height=500,
        showlegend=True
    )
    
    return fig

def create_management_matrix_heatmap(matrix, title, freq_labels, sev_labels):
    """Create management matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=matrix[::-1],
        x=sev_labels.flatten(),
        y=freq_labels[:, ::-1].flatten(),
        colorscale='Viridis',
        showscale=True,
        text=matrix[::-1],
        texttemplate="%{text:.1f}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Severidad",
        yaxis_title="Frecuencia",
        height=400
    )
    
    return fig

def create_cost_benefit_chart(results_df):
    """Create cost-benefit analysis chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=results_df['CG (USD)'],
        y=results_df['VCap. (USD)'],  # Updated to match new column name
        mode='markers+text',
        text=results_df.index,
        textposition="top center",
        marker=dict(
            size=12,
            color=results_df['Media (USD)'],
            colorscale='RdYlGn_r',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Cost-Benefit Analysis",
        xaxis_title="Management Cost (USD)",
        yaxis_title="Value Captured (USD)",
        height=500
    )
    
    return fig