"""
app.py
======
Streamlit front-end for the Letdown Steam System process graphic.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder (app.py, diagram.py, requirements.txt) to a GitHub repo.
    2. On share.streamlit.io, point a new app at app.py in that repo.
"""

import streamlit as st
from diagram import build_figure, BG_COLOR

st.set_page_config(
    page_title="Letdown Steam System",
    page_icon="♨️",
    layout="wide",
)

# Match the app background to the figure background so the graphic
# sits flush with the page instead of showing a white card edge.
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {BG_COLOR}; }}
        [data-testid="stHeader"] {{ background-color: rgba(0,0,0,0); }}
    </style>
    """,
    unsafe_allow_html=True,
)

fig = build_figure()
st.pyplot(fig, use_container_width=True)

with st.expander("About this system"):
    st.markdown(
        """
        This graphic shows a typical **HP-to-LP steam letdown station**:

        - **High Pressure Steam Line** — supplies steam from the upstream
          high pressure header.
        - **Pressure Control Valve (PCV)** — reduces steam pressure from
          HP to the LP setpoint.
        - **Spray Desuperheater (venturi type)** — feedwater is injected
          at the throat of a venturi, where high steam velocity promotes
          rapid atomisation and mixing, cooling the steam toward
          saturation.
        - **Low Pressure Steam Line** — carries the reduced-pressure,
          desuperheated steam onward to the LP distribution header.
        """
    )
