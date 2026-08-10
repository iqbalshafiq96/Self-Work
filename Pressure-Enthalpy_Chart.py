import numpy as np
import pandas as pd

# Standard industrial red-flag color palette
COLOR_RED = "#D9534F"     # Degradation / Penalty
COLOR_GREEN = "#5CB85C"   # Optimal performance
COLOR_NEUTRAL = "#6C757D" # Baseline / Informational

def calculate_compressor_metrics(
    p1_bar: float, 
    t1_k: float, 
    p2_bar: float, 
    t2_k: float, 
    k_ratio: float, 
    t_sat_suct_k: float, 
    t_sat_disch_k: float, 
    power_kw: float, 
    enthalpy_diff_superheat_kw: float
) -> dict:
    """
    Calculates compressor performance profile metrics and applies threshold color coding.
    
    Parameters:
     p1_bar, p2_bar : Suction and Discharge pressures (bara)
     t1_k, t2_k     : Suction and Discharge temperatures (K)
     k_ratio        : Ratio of specific heats (Cp/Cv)
    """
    # 1. Compression Ratio (Neutral)
    comp_ratio = p2_bar / p1_bar
    
    # 2. Isentropic Efficiency (%) - Lower is Red
    # Formula: T1 / (T2 - T1) * [(P2/P1)^((k-1)/k) - 1]
    exponent = (k_ratio - 1.0) / k_ratio
    eta_isen = (t1_k / (t2_k - t1_k)) * ((comp_ratio ** exponent) - 1.0) * 100.0
    
    # 3. Suction Superheat (K) - Higher is Red
    suct_superheat_k = t1_k - t_sat_suct_k
    
    # 4. Discharge Superheat (K) - Higher is Red
    disch_superheat_k = t2_k - t_sat_disch_k
    
    # 5. Discharge Superheat (kW) - Higher is Red
    disch_superheat_kw = enthalpy_diff_superheat_kw
    
    return {
        "Compression_Ratio": comp_ratio,
        "Isentropic_Efficiency_pct": eta_isen,
        "Suction_Superheat_K": suct_superheat_k,
        "Discharge_Superheat_K": disch_superheat_k,
        "Discharge_Superheat_kW": disch_superheat_kw,
        "Compressor_Power_kW": power_kw
    }

def apply_color_logic(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Applies conditional color logic based on defined operational thresholds.
    """
    color_df = pd.DataFrame(index=df.index)
    
    # Compression Ratio: Neutral
    color_df["Color_Compression_Ratio"] = COLOR_NEUTRAL
    
    # Isentropic Efficiency: Lower than target is Red
    color_df["Color_Isentropic_Efficiency"] = np.where(
        df["Isentropic_Efficiency_pct"] < thresholds.get("eta_isen_min", 75.0),
        COLOR_RED, COLOR_GREEN
    )
    
    # Suction Superheat: Higher is Red
    color_df["Color_Suction_Superheat"] = np.where(
        df["Suction_Superheat_K"] > thresholds.get("suct_sh_max", 10.0),
        COLOR_RED, COLOR_GREEN
    )
    
    # Discharge Superheat (K): Higher is Red
    color_df["Color_Discharge_Superheat_K"] = np.where(
        df["Discharge_Superheat_K"] > thresholds.get("disch_sh_max", 30.0),
        COLOR_RED, COLOR_GREEN
    )
    
    # Discharge Superheat (kW): Higher is Red
    color_df["Color_Discharge_Superheat_kW"] = np.where(
        df["Discharge_Superheat_kW"] > thresholds.get("disch_sh_kw_max", 15.0),
        COLOR_RED, COLOR_GREEN
    )
    
    # Compressor Power: Higher than limit is Red
    color_df["Color_Compressor_Power"] = np.where(
        df["Compressor_Power_kW"] > thresholds.get("power_max_kw", 250.0),
        COLOR_RED, COLOR_GREEN
    )
    
    return color_df

# Example Usage
if __name__ == "__main__":
    sample_data = {
        "p1_bar": 2.5, "t1_k": 268.15,
        "p2_bar": 12.0, "t2_k": 365.15,
        "k_ratio": 1.31, # e.g., Ammonia
        "t_sat_suct_k": 263.15, "t_sat_disch_k": 305.15,
        "power_kw": 280.0, "enthalpy_diff_superheat_kw": 18.5
    }
    
    # 1. Compute single point metrics
    metrics = calculate_compressor_metrics(**sample_data)
    df = pd.DataFrame([metrics])
    
    # 2. Assign operational thresholds for triggering 'Red' status
    threshold_limits = {
        "eta_isen_min": 70.0,   # Below 70% efficiency triggers RED
        "suct_sh_max": 8.0,     # Above 8K superheat triggers RED
        "disch_sh_max": 25.0,   # Above 25K superheat triggers RED
        "disch_sh_kw_max": 10.0,# Above 10 kW heat penalty triggers RED
        "power_max_kw": 250.0   # Power consumption limit
    }
    
    colors = apply_color_logic(df, threshold_limits)
    result_df = pd.concat([df, colors], axis=1)
    
    print(result_df.T)
