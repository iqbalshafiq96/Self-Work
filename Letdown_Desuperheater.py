# @title 💨 Desuperheater Letdown Mass and Energy Balance Calculator { display-mode: "form" }
# @markdown Fill in the process parameters on the right, then press **Ctrl + F9** or click the **Play** button on the left to execute the calculation.
# @markdown ---

# @markdown ### Pressure Unit Type Selection
Pressure_Unit_Type = "Bar Gauge (barG)"  # @param ["Bar Gauge (barG)", "Bar Absolute (barA)", "Megapascals Gauge (MPaG)", "Megapascals Absolute (MPaA)"]

# @markdown ---
# @markdown ### 1. High-Pressure Inlet Steam Parameters
High_Pressure_Inlet_Steam_Pressure = 50  # @param {type:"number"}
High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius = 419  # @param {type:"number"}

# @markdown ---
# @markdown ### 2. Desuperheater Outlet Parameters & Calculation Mode
Desuperheater_Outlet_Steam_Pressure = 4.6  # @param {type:"number"}
Outlet_Temperature_Calculation_Mode = "INPUT - Specify Target Outlet Temperature"  # @param ["INPUT - Specify Target Outlet Temperature", "CALC - Calculate Outlet Temperature from Spray Flow"]

# @markdown **[ Used ONLY when mode is set to INPUT ]**
Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius = 160  # @param {type:"number"}

# @markdown ---
# @markdown ### 3. Spray Feedwater Inlet Parameters
Spray_Feedwater_Inlet_Pressure = 70  # @param {type:"number"}
Spray_Feedwater_Inlet_Temperature_Degrees_Celsius = 90  # @param {type:"number"}

# @markdown **[ Used ONLY when mode is set to CALC ]**
Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour = 2.35  # @param {type:"number"}

# @markdown ---
# @markdown ### 4. Mass Flow Rate Basis and Specification
Mass_Flow_Rate_Basis = "Inlet Steam Flow Rate"  # @param ["Inlet Steam Flow Rate", "Outlet Target Steam Flow Rate"]
Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour = 107  # @param {type:"number"}

# --- AUTOMATIC CALCULATION EXECUTION ---
import sys

from iapws import IAPWS97

ATMOSPHERIC_PRESSURE_MEGAPASCALS = 0.101325
ATMOSPHERIC_PRESSURE_BAR = 1.01325


# --- PLAIN TEXT HELPER FUNCTION (NO ANSI CODES) ---
def digital_style(text, *args, **kwargs):
    return str(text)


# Standardize inputs to Absolute Pressure in Megapascals (MPaA) for IAPWS-IF97 library
if Pressure_Unit_Type == "Bar Gauge (barG)":
    pressure_steam_inlet_absolute_mpaa = (
        High_Pressure_Inlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0
    pressure_steam_outlet_absolute_mpaa = (
        Desuperheater_Outlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0
    pressure_feedwater_inlet_absolute_mpaa = (
        Spray_Feedwater_Inlet_Pressure + ATMOSPHERIC_PRESSURE_BAR
    ) / 10.0

elif Pressure_Unit_Type == "Bar Absolute (barA)":
    pressure_steam_inlet_absolute_mpaa = (
        High_Pressure_Inlet_Steam_Pressure / 10.0
    )
    pressure_steam_outlet_absolute_mpaa = (
        Desuperheater_Outlet_Steam_Pressure / 10.0
    )
    pressure_feedwater_inlet_absolute_mpaa = (
        Spray_Feedwater_Inlet_Pressure / 10.0
    )

elif Pressure_Unit_Type == "Megapascals Gauge (MPaG)":
    pressure_steam_inlet_absolute_mpaa = (
        High_Pressure_Inlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )
    pressure_steam_outlet_absolute_mpaa = (
        Desuperheater_Outlet_Steam_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )
    pressure_feedwater_inlet_absolute_mpaa = (
        Spray_Feedwater_Inlet_Pressure + ATMOSPHERIC_PRESSURE_MEGAPASCALS
    )

else:  # Megapascals Absolute (MPaA)
    pressure_steam_inlet_absolute_mpaa = High_Pressure_Inlet_Steam_Pressure
    pressure_steam_outlet_absolute_mpaa = Desuperheater_Outlet_Steam_Pressure
    pressure_feedwater_inlet_absolute_mpaa = Spray_Feedwater_Inlet_Pressure

# Calculate display values across all pressure units
p_in_mpaa = pressure_steam_inlet_absolute_mpaa
p_in_mpag = p_in_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS
p_in_bara = p_in_mpaa * 10.0
p_in_barg = p_in_bara - ATMOSPHERIC_PRESSURE_BAR

p_out_mpaa = pressure_steam_outlet_absolute_mpaa
p_out_mpag = p_out_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS
p_out_bara = p_out_mpaa * 10.0
p_out_barg = p_out_bara - ATMOSPHERIC_PRESSURE_BAR

p_fw_mpaa = pressure_feedwater_inlet_absolute_mpaa
p_fw_mpag = p_fw_mpaa - ATMOSPHERIC_PRESSURE_MEGAPASCALS
p_fw_bara = p_fw_mpaa * 10.0
p_fw_barg = p_fw_bara - ATMOSPHERIC_PRESSURE_BAR

temperature_steam_inlet = High_Pressure_Inlet_Steam_Temperature_Degrees_Celsius
temperature_feedwater_inlet = (
    Spray_Feedwater_Inlet_Temperature_Degrees_Celsius
)

# Inlet Specific Enthalpies
enthalpy_steam_inlet = IAPWS97(
    P=p_in_mpaa, T=temperature_steam_inlet + 273.15
).h
enthalpy_feedwater_inlet = IAPWS97(
    P=p_fw_mpaa, T=temperature_feedwater_inlet + 273.15
).h

# --- CALCULATION LOGIC BY MODE ---
is_calc_mode = (
    Outlet_Temperature_Calculation_Mode
    == "CALC - Calculate Outlet Temperature from Spray Flow"
)

if is_calc_mode:
    # CALC MODE: Spray feedwater flow is specified, calculate outlet steam temperature
    mass_flow_feedwater_inlet = (
        Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour
    )

    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        mass_flow_steam_inlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_outlet = (
            mass_flow_steam_inlet + mass_flow_feedwater_inlet
        )
    else:  # Outlet Target Steam Flow Rate
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = (
            mass_flow_steam_outlet - mass_flow_feedwater_inlet
        )

    # Energy Balance: Mixed Outlet Specific Enthalpy (kJ/kg)
    enthalpy_steam_outlet = (
        (mass_flow_steam_inlet * enthalpy_steam_inlet)
        + (mass_flow_feedwater_inlet * enthalpy_feedwater_inlet)
    ) / mass_flow_steam_outlet

    # Determine Outlet Temperature from calculated pressure and specific enthalpy
    outlet_state = IAPWS97(P=p_out_mpaa, h=enthalpy_steam_outlet)
    temperature_steam_outlet = outlet_state.T - 273.15

else:
    # INPUT MODE: Target outlet steam temperature is specified, calculate required spray flow
    temperature_steam_outlet = (
        Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius
    )
    enthalpy_steam_outlet = IAPWS97(
        P=p_out_mpaa, T=temperature_steam_outlet + 273.15
    ).h

    if Mass_Flow_Rate_Basis == "Inlet Steam Flow Rate":
        mass_flow_steam_inlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_feedwater_inlet = (
            mass_flow_steam_inlet
            * (enthalpy_steam_outlet - enthalpy_steam_inlet)
            / (enthalpy_feedwater_inlet - enthalpy_steam_outlet)
        )
        mass_flow_steam_outlet = (
            mass_flow_steam_inlet + mass_flow_feedwater_inlet
        )
    else:  # Outlet Target Steam Flow Rate
        mass_flow_steam_outlet = Specified_Steam_Mass_Flow_Rate_Tons_Per_Hour
        mass_flow_steam_inlet = (
            mass_flow_steam_outlet
            * (enthalpy_steam_outlet - enthalpy_feedwater_inlet)
            / (enthalpy_steam_inlet - enthalpy_feedwater_inlet)
        )
        mass_flow_feedwater_inlet = (
            mass_flow_steam_outlet - mass_flow_steam_inlet
        )

# Saturation Properties at Outlet Pressure
saturated_liquid = IAPWS97(P=p_out_mpaa, x=0)
saturation_temperature_outlet_degrees_celsius = saturated_liquid.T - 273.15
superheat_margin_degrees_celsius = (
    temperature_steam_outlet - saturation_temperature_outlet_degrees_celsius
)

# Determine Physical Phase
if superheat_margin_degrees_celsius > 0.1:
    outlet_steam_condition = "SUPERHEATED STEAM"
elif abs(superheat_margin_degrees_celsius) <= 0.1:
    outlet_steam_condition = "SATURATED STEAM (Dry Saturated)"
else:
    outlet_steam_condition = "WET STEAM (Two-Phase Liquid and Vapor)"

pressure_drop_megapascals = p_in_mpaa - p_out_mpaa
pressure_drop_bar = pressure_drop_megapascals * 10.0

# Pressure unit display string formatting
if Pressure_Unit_Type == "Bar Gauge (barG)":
    p_in_str, p_out_str, p_fw_str = (
        f"{p_in_barg:.2f} barG",
        f"{p_out_barg:.2f} barG",
        f"{p_fw_barg:.2f} barG",
    )
elif Pressure_Unit_Type == "Bar Absolute (barA)":
    p_in_str, p_out_str, p_fw_str = (
        f"{p_in_bara:.2f} barA",
        f"{p_out_bara:.2f} barA",
        f"{p_fw_bara:.2f} barA",
    )
elif Pressure_Unit_Type == "Megapascals Gauge (MPaG)":
    p_in_str, p_out_str, p_fw_str = (
        f"{p_in_mpag:.2f} MPaG",
        f"{p_out_mpag:.2f} MPaG",
        f"{p_fw_mpag:.2f} MPaG",
    )
else:
    p_in_str, p_out_str, p_fw_str = (
        f"{p_in_mpaa:.2f} MPaA",
        f"{p_out_mpaa:.2f} MPaA",
        f"{p_fw_mpaa:.2f} MPaA",
    )

# --- OUTPUT DISPLAY ---
header_line = "=" * 80

print("\n")
print(digital_style(header_line))
print(
    digital_style(
        "            DESUPERHEATER PROCESS FLOW SCHEME AND MASS BALANCE            ".center(
            80
        )
    )
)
print(digital_style(header_line))

flow_line_1 = f"Inlet Steam ({p_in_str}, {temperature_steam_inlet:.1f} °C, {mass_flow_steam_inlet:.2f} t/h) ----| "
flow_line_2 = (
    "                                                                             |----> Outlet Steam ("
    f"{p_out_str}, {temperature_steam_outlet:.1f} °C, {mass_flow_steam_outlet:.2f} t/h)"
)
flow_line_3 = f"Feedwater   ({p_fw_str}, {temperature_feedwater_inlet:.1f} °C, {mass_flow_feedwater_inlet:.2f} t/h) ----| (Desuperheater)"

print(digital_style(flow_line_1))
print(digital_style(flow_line_2))
print(digital_style(flow_line_3))
print(digital_style(header_line))

print("\n")
print(
    digital_style(
        "------------------------- DETAILED PROCESS RESULTS -------------------------"
    )
)

print(digital_style(f"Selected Input Unit Basis                 : {Pressure_Unit_Type}"))
print(digital_style(f"Calculation Mode                          : {Outlet_Temperature_Calculation_Mode}"))

if is_calc_mode:
    target_temp_str = "[IGNORED IN CALC MODE]"
    spray_flow_str = f"{Specified_Spray_Feedwater_Mass_Flow_Rate_Tons_Per_Hour:.2f} t/h"
else:
    target_temp_str = f"{Desuperheater_Outlet_Steam_Target_Temperature_Degrees_Celsius:.2f} °C"
    spray_flow_str = "[CALCULATED FROM TARGET TEMP]"

print(digital_style(f"Specified Target Outlet Temperature       : {target_temp_str}"))
print(digital_style(f"Specified Spray Feedwater Flow            : {spray_flow_str}"))

# Included Spray Feedwater Mass Flow Rate in Detailed Results list
results_body = [
    f"High-Pressure Inlet Steam Pressure        : {p_in_barg:.2f} barG | {p_in_bara:.2f} barA | {p_in_mpag:.3f} MPaG | {p_in_mpaa:.3f} MPaA",
    f"Desuperheater Outlet Steam Pressure       : {p_out_barg:.2f} barG | {p_out_bara:.2f} barA | {p_out_mpag:.3f} MPaG | {p_out_mpaa:.3f} MPaA",
    f"Spray Feedwater Inlet Pressure            : {p_fw_barg:.2f} barG | {p_fw_bara:.2f} barA | {p_fw_mpag:.3f} MPaG | {p_fw_mpaa:.3f} MPaA",
    f"Inlet Steam Pressure Drop                 : {pressure_drop_bar:.2f} Bar ({pressure_drop_megapascals:.3f} Megapascals)",
    "----------------------------------------------------------------------------",
    f"Outlet Steam Physical Condition          : {outlet_steam_condition}",
    f"Resulting Outlet Steam Temperature       : {temperature_steam_outlet:.2f} °C",
    f"Outlet Saturation Temperature            : {saturation_temperature_outlet_degrees_celsius:.2f} °C",
    f"Outlet Superheat Margin                  : {superheat_margin_degrees_celsius:.2f} °C",
    "----------------------------------------------------------------------------",
    f"Inlet Steam Mass Flow Rate               : {mass_flow_steam_inlet:.2f} t/h",
    f"Spray Feedwater Mass Flow Rate           : {mass_flow_feedwater_inlet:.2f} t/h",
    f"Outlet Steam Mass Flow Rate              : {mass_flow_steam_outlet:.2f} t/h",
    "----------------------------------------------------------------------------",
    f"Specific Enthalpy of Inlet Steam         : {enthalpy_steam_inlet:.2f} kJ/kg",
    f"Specific Enthalpy of Spray Feedwater     : {enthalpy_feedwater_inlet:.2f} kJ/kg",
    f"Specific Enthalpy of Outlet Steam        : {enthalpy_steam_outlet:.2f} kJ/kg",
]

for line in results_body:
    print(digital_style(line))

print(digital_style(header_line))

# Safety & Operational Alerts
if superheat_margin_degrees_celsius < 0:
    alert_1 = (
        "\n[!] CRITICAL ALERT: Outlet temperature is below the saturation"
        " temperature!"
    )
    alert_2 = (
        "    Condensed liquid water droplets will be present in the steam line."
    )
    print(digital_style(alert_1))
    print(digital_style(alert_2))
elif 0 <= superheat_margin_degrees_celsius < 2.0:
    alert_1 = (
        "\n[!] WARNING: Low superheat margin (less than 2 Degrees Celsius)!"
    )
    alert_2 = (
        "    There is a high risk of incomplete vaporization and water droplet"
        " carryover downstream."
    )
    print(digital_style(alert_1))
    print(digital_style(alert_2))
