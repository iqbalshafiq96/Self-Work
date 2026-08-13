# Calculate required spectral resolution parameters
req_delta_f = fr_driver_hz / 2.5  # Rule of thumb for sideband resolution
min_fft_lines = int(np.ceil(fmax_hz / req_delta_f))

# Standard commercial analyzer line options
standard_lines = [400, 800, 1600, 3200, 6400, 12800]
recommended_lines = next(
    (l for l in standard_lines if l >= min_fft_lines), 12800
)
actual_delta_f = fmax_hz / recommended_lines

# UI Highlight Box
st.info(
    f"**Required Spectral Resolution Guidance:**\n"
    f"* **Driver Running Speed ($f_r$):** `{fr_driver_hz:.2f} Hz`\n"
    f"* **Max Sideband Spacing Requirement ($\Delta f_{{max}}$):** `{req_delta_f:.2f} Hz`\n"
    f"* **Recommended Analyzer Lines:** `{recommended_lines:,} Lines` at $F_{{max}} = {fmax_hz:.0f}\\text{{ Hz}}$\n"
    f"* **Achieved Resolution:** `{actual_delta_f:.3f} Hz/line` (Sufficient to separate $1x$ sidebands)"
)
