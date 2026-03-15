import pandas as pd
import numpy as np
from src.config import ANGLO_COUNTRIES

def apply_base_features(df):
    df = df.copy()
    
    if 'tunnel_pct' in df.columns:
        df['tunnel_pct'] = df['tunnel_pct'].apply(lambda x: x/100 if x > 1 else x)
        
    if 'num_stations' in df.columns and 'length_km' in df.columns:
        df['station_density'] = df['num_stations'] / (df['length_km'] + 0.1)
        
    if 'length_km' in df.columns:
        df['log_length'] = np.log(df['length_km'])
        
    if 'start_year' in df.columns and 'end_year' in df.columns:
        df['mid_year'] = (df['start_year'] + df['end_year']) / 2

    if 'is_regional_rail' in df.columns:
        df['is_regional_rail'] = df['is_regional_rail'].fillna(0.0)
        
    return df