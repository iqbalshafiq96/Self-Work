import pandas as pd


def calculate_gearbox_vibration(
    driver_rpm: float,
    teeth_stage1_gear: int,
    teeth_stage1_pinion: int,
    teeth_stage2_gear: int = 0,
    teeth_stage2_pinion: int = 0,
    is_reducer: bool = False,
    num_stages: int = 1,
) -> dict:
    """Calculates fundamental frequencies and fault harmonics for a 1-stage or 2-stage gearbox.

    Args:
        driver_rpm: Input rotational speed in RPM.
        teeth_stage1_gear: Number of teeth on Stage 1 Gear.
        teeth_stage1_pinion: Number of teeth on Stage 1 Pinion.
        teeth_stage2_gear: Number of teeth on Stage 2 Gear (optional).
        teeth_stage2_pinion: Number of teeth on Stage 2 Pinion (optional).
        is_reducer: True if Speed Reducer (Pinion drives Gear),
                    False if Speed Increaser (Gear drives Pinion).
        num_stages: Number of gear stages (1 or 2).

    Returns:
        dict containing 'stage_results' and 'frequencies' DataFrames.
    """
    input_hz = driver_rpm / 60.0
    stage_results = []

    # -------------------------------------------------------------------------
    # STAGE 1 CALCULATIONS
    # -------------------------------------------------------------------------
    ratio_s1 = (
        teeth_stage1_gear / teeth_stage1_pinion
        if is_reducer
        else teeth_stage1_pinion / teeth_stage1_gear
    )

    if is_reducer:
        # Driver = Pinion (Input), Driven = Gear (Output)
        f_pinion_s1_hz = input_hz
        f_gear_s1_hz = input_hz / (teeth_stage1_gear / teeth_stage1_pinion)
    else:
        # Driver = Gear (Input), Driven = Pinion (Output)
        f_gear_s1_hz = input_hz
        f_pinion_s1_hz = input_hz * (teeth_stage1_gear / teeth_stage1_pinion)

    output_s1_hz = f_gear_s1_hz if is_reducer else f_pinion_s1_hz
    gmfo_s1 = f_gear_s1_hz * teeth_stage1_gear  # GMFO = Gear Speed * Gear Teeth

    stage_results.append(
        {
            "Stage": 1,
            "Input Speed (Hz)": input_hz,
            "Output Speed (Hz)": output_s1_hz,
            "Gear Teeth": teeth_stage1_gear,
            "Pinion Teeth": teeth_stage1_pinion,
            "Gear Speed (Hz)": f_gear_s1_hz,
            "Pinion Speed (Hz)": f_pinion_s1_hz,
            "GMF (Hz)": gmfo_s1,
        }
    )

    # -------------------------------------------------------------------------
    # STAGE 2 CALCULATIONS (IF APPLICABLE)
    # -------------------------------------------------------------------------
    if num_stages == 2 and teeth_stage2_gear > 0 and teeth_stage2_pinion > 0:
        input_s2_hz = output_s1_hz

        if is_reducer:
            f_pinion_s2_hz = input_s2_hz
            f_gear_s2_hz = input_s2_hz / (
                teeth_stage2_gear / teeth_stage2_pinion
            )
        else:
            f_gear_s2_hz = input_s2_hz
            f_pinion_s2_hz = input_s2_hz * (
                teeth_stage2_gear / teeth_stage2_pinion
            )

        output_s2_hz = f_gear_s2_hz if is_reducer else f_pinion_s2_hz
        gmfo_s2 = f_gear_s2_hz * teeth_stage2_gear

        stage_results.append(
            {
                "Stage": 2,
                "Input Speed (Hz)": input_s2_hz,
                "Output Speed (Hz)": output_s2_hz,
                "Gear Teeth": teeth_stage2_gear,
                "Pinion Teeth": teeth_stage2_pinion,
                "Gear Speed (Hz)": f_gear_s2_hz,
                "Pinion Speed (Hz)": f_pinion_s2_hz,
                "GMF (Hz)": gmfo_s2,
            }
        )

    # -------------------------------------------------------------------------
    # HARMONICS & SIDEBANDS COMPILATION
    # -------------------------------------------------------------------------
    freq_data = []

    for stage_info in stage_results:
        stg = stage_info["Stage"]
        gmf = stage_info["GMF (Hz)"]
        fg = stage_info["Gear Speed (Hz)"]
        fp = stage_info["Pinion Speed (Hz)"]

        # Fundamental Speeds
        freq_data.append(
            {
                "Category": f"Stage {stg} Shaft",
                "Component": f"Gear Speed 1X (S{stg})",
                "Frequency (Hz)": fg,
                "Order (x Input)": fg / input_hz,
            }
        )
        freq_data.append(
            {
                "Category": f"Stage {stg} Shaft",
                "Component": f"Pinion Speed 1X (S{stg})",
                "Frequency (Hz)": fp,
                "Order (x Input)": fp / input_hz,
            }
        )

        # GMF Harmonics (1X, 2X, 3X)
        for h in range(1, 4):
            freq_data.append(
                {
                    "Category": f"Stage {stg} GMF",
                    "Component": f"GMF {h}X (S{stg})",
                    "Frequency (Hz)": gmf * h,
                    "Order (x Input)": (gmf * h) / input_hz,
                }
            )

            # Sidebands around 1X GMF
            if h == 1:
                freq_data.append(
                    {
                        "Category": f"Stage {stg} Sidebands",
                        "Component": f"GMF - 1X Gear (S{stg})",
                        "Frequency (Hz)": gmf - fg,
                        "Order (x Input)": (gmf - fg) / input_hz,
                    }
                )
                freq_data.append(
                    {
                        "Category": f"Stage {stg} Sidebands",
                        "Component": f"GMF + 1X Gear (S{stg})",
                        "Frequency (Hz)": gmf + fg,
                        "Order (x Input)": (gmf + fg) / input_hz,
                    }
                )
                freq_data.append(
                    {
                        "Category": f"Stage {stg} Sidebands",
                        "Component": f"GMF - 1X Pinion (S{stg})",
                        "Frequency (Hz)": gmf - fp,
                        "Order (x Input)": (gmf - fp) / input_hz,
                    }
                )
                freq_data.append(
                    {
                        "Category": f"Stage {stg} Sidebands",
                        "Component": f"GMF + 1X Pinion (S{stg})",
                        "Frequency (Hz)": gmf + fp,
                        "Order (x Input)": (gmf + fp) / input_hz,
                    }
                )

    return {
        "stage_summary": pd.DataFrame(stage_results),
        "frequencies": pd.DataFrame(freq_data),
    }


# =============================================================================
# EXAMPLE RUN (Speed Increaser: 1500 RPM, Gear=57T, Pinion=19T)
# =============================================================================
if __name__ == "__main__":
    results = calculate_gearbox_vibration(
        driver_rpm=1500.0,
        teeth_stage1_gear=57,
        teeth_stage1_pinion=19,
        is_reducer=False,  # Speed Increaser: Gear drives Pinion
        num_stages=1,
    )

    print("--- STAGE SUMMARY ---")
    print(results["stage_summary"].to_string(index=False))

    print("\n--- VIBRATION FREQUENCIES ---")
    print(results["frequencies"].to_string(index=False))
