import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# 1. Capture ALL Power BI Data
df_pbi = dataset.copy()

# Extract arrays - Clean NaNs to prevent griddata from failing
actual_igv_array = df_pbi['IGVdegcalc'].fillna(0).values
actual_pressure_array = df_pbi['Discharge Pressure'].fillna(0).values

# 2. Reference Map Data
data = {
    "X": [50000,53300,60000,65900,70000,73700,80000, 53300,60000,65900,70000,72000,76000,77000, 
          67100,70000,80000,82300,90000,92900,96800,97700, 80000,90000,100000,106400,110000,113600,115900,116800, 
          80000,90000,94900,100000,110000,118500,120000,121700,122300,122301, 77000,97700,116800,121800],
    "Y": [2.64,2.65,2.66,2.7,2.76,2.8,2.85, 2.65,2.605,2.52,2.45,2.4,2.2,2.15, 
          2.72,2.71,2.63,2.6,2.48,2.4,2.2,2.11, 2.86,2.81,2.7,2.6,2.54,2.4,2.2,2.05, 
          2.86,2.84,2.8,2.75,2.61,2.4,2.33,2.2,2,1.81, 2.15,2.11,2.05,1.81],
    "Curve": ["Surge"]*7 + ["-70°"]*7 + ["-48°"]*8 + ["0°"]*8 + ["15°"]*10 + ["Stonewall"]*4
}
df_map = pd.DataFrame(data)

# 3. IGV Interpolation
igv_map = {"-70°": -70, "-48°": -48, "0°": 0, "15°": 15}
df_interp_ref = df_map[df_map["Curve"].isin(igv_map.keys())].copy()
df_interp_ref["Angle"] = df_interp_ref["Curve"].map(igv_map)

points = df_interp_ref[['Angle', 'Y']].values
values = df_interp_ref['X'].values

# --- REPAIRED INTERPOLATION LOGIC ---
try:
    # Linear method for precision within the map
    x_predicted_array = griddata(points, values, (actual_igv_array, actual_pressure_array), method='linear')
    
    # Fill any NaNs (points on the very edge or outside) using 'nearest'
    nan_mask = np.isnan(x_predicted_array)
    if np.any(nan_mask):
        x_predicted_array[nan_mask] = griddata(points, values, (actual_igv_array[nan_mask], actual_pressure_array[nan_mask]), method='nearest')
except:
    x_predicted_array = np.full(len(df_pbi), np.nan)

# 4. Plotting - SHARPNESS ENHANCED (DPI 150)
fig, ax = plt.subplots(figsize=(6.73, 5.17), dpi=150)

# --- AXIS LIMITS AND FONT DEFINITION ---
target_font = 'Segoe UI'
ax.set_xlim(45000, 125000)
ax.set_ylim(1.75, 2.9)

# --- REMOVE TOP AND RIGHT OUTLINES ---
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# A. Reference Curves
curve_order = ["Surge", "-70°", "-48°", "0°", "15°", "Stonewall"]
for curve in curve_order:
    grp = df_map[df_map["Curve"] == curve]
    
    if curve in ["Surge", "Stonewall"]:
        c = 'red' if curve == "Surge" else 'black'
        ls = '--'
    else:
        c = None
        ls = '-'
    
    ax.plot(grp["X"], grp["Y"], label=curve, 
            color=c, 
            linestyle=ls, 
            linewidth=1.2, 
            alpha=0.8, 
            zorder=2)

# B. Measured Operating Points
ax.scatter(df_pbi['Flow (m3/h)'], 
            df_pbi['Discharge Pressure'], 
            color='#0047AB', 
            s=10, 
            edgecolors='white', 
            linewidth=0.2, 
            alpha=0.5, 
            label='Measured flow (F1100_AI1)', 
            zorder=4)

# C. Predicted Operating Points (Using repaired x_predicted_array)
ax.scatter(x_predicted_array, 
            actual_pressure_array, 
            color='red', 
            s=12, 
            marker='x', 
            linewidth=0.8, 
            alpha=0.7, 
            label='Predicted flow (Actual IGV)', 
            zorder=5)

# 5. Formatting: Segoe UI
ax.set_xlabel("Flow (m3/hr)", fontname=target_font, fontsize=10)
ax.set_ylabel("Discharge Pressure (bara)", fontname=target_font, fontsize=10)

# Tick labels
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname(target_font)
    label.set_fontsize(8)

# Legend
ax.legend(loc='lower left', frameon=False, prop={'family': target_font, 'size': 7})

# Grid color kept light as per your design
ax.grid(True, which='major', linestyle='-', linewidth=0.3, color='#F0F0F0')

plt.tight_layout()
plt.show()
