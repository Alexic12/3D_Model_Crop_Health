
import numpy as np
import pandas as pd

class HCPCalc():
    def __init__(self):
        pass

    @staticmethod
    def compute_risk_results_via_hpc(indice, anio):
        """
        Example function that calls the HPC pipeline code (or stubs HPC).
        Then it reads final HPC files to build a 2D map of points + monthly distributions.

        Args:
            indice (str): The vegetation index (e.g., NDVI).
            anio (str): The year of analysis.

        Returns:
            tuple: (df_map, risk_info)
                df_map is a pandas DataFrame with location-based NDVI info.
                risk_info is a dict keyed by point_id with HPC risk data.
        """
        # (A) If you want to run HPC every time:
        # hpc_data = run_full_hpc_pipeline(indice, anio)
        # parse hpc_data as needed

        # (B) For demonstration, create mock data
        df_map = pd.DataFrame({
            "point_id": [0,1,2,3,4],
            "Lon": [-74.05, -74.02, -74.00, -73.98, -73.95],
            "Lat": [4.70, 4.72, 4.69, 4.73, 4.71],
            "NDVI": [0.66, 0.72, 0.60, 0.55, 0.80]
        })

        risk_info = {}
        np.random.seed(42)
        for pid in df_map["point_id"]:
            monthly_samples = np.random.normal(loc=(pid+1)*100, scale=15, size=(1000, 12))
            dfm = pd.DataFrame({
                "WD": [f"WD_{m}" for m in range(12)],
                "Max C": np.random.uniform(10, 30, 12),
                "Min  C": np.random.uniform(5, 15, 12),
                "Viento (m/s)": np.random.uniform(0.5, 5.0, 12),
                "Humedad (%)": np.random.uniform(40, 90, 12),
                "Precip. (mm)": np.random.uniform(0, 30, 12),
                indice: np.random.uniform(0.4, 0.9, 12),
                "Skewness": np.random.uniform(0, 1, 12),
                "%C1": np.random.uniform(0, 1, 12),
                "%C2": np.random.uniform(0, 1, 12),
                "%C3": np.random.uniform(0, 1, 12),
                "Mean (USD)": np.random.uniform(500, 2000, 12),
                "75% (USD)": np.random.uniform(2000, 3000, 12),
                "OpVar-99.9% (USD)": np.random.uniform(3000, 4000, 12)
            })
            dfm["Max C"] = dfm["Max C"].map("{:.3f}".format)
            dfm["Min  C"] = dfm["Min  C"].map("{:.3f}".format)

            risk_info[pid] = {
                "monthly_distribution": monthly_samples,
                "df_table": dfm
            }

        return df_map, risk_info