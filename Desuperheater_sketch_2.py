import streamlit as st


# ==============================================================
# ANIMATED PROCESS SVG
# ==============================================================


def build_animated_process_svg(
    p_in,
    t_in,
    m_in,
    p_fw,
    t_fw,
    m_fw,
    p_out,
    t_out,
    m_out,
    p_unit,
):

    # Dynamic color palette definitions
    STEAM_COLOR = "#00D2FF"
    STEAM_GLOW = "#80E5FF"
    FW_COLOR = "#0055FF"
    FW_GLOW = "#3388FF"
    EQUIP_COLOR = "#E0E6ED"
    EQUIP_GLOW = "#00D2FF"
    TEXT_COLOR = "#FFFFFF"

    # Dynamic particle speed calculation based on mass flow rates (safeguarded against zero flow)
    steam_speed = max(0.8, min(6.0, 10.0 / (m_in if m_in > 0 else 1.0)))
    fw_speed = max(0.5, min(4.0, 5.0 / (m_fw if m_fw > 0 else 1.0)))

    # ----------------------------------------------------------
    # Main SVG
    # ----------------------------------------------------------

    svg = f"""
    <svg
        width="100%"
        height="390"
        viewBox="0 0 1500 390"
        xmlns="http://www.w3.org/2000/svg"
        preserveAspectRatio="xMidYMid meet"
    >

    <defs>

        <style>
            @keyframes dashFlow {{
                from {{ stroke-dashoffset: 40; }}
                to {{ stroke-dashoffset: 0; }}
            }}
            .animated-pipe {{
                stroke-dasharray: 8 12;
                animation: dashFlow 1s linear infinite;
            }}
            .fw-pipe {{
                stroke-dasharray: 6 10;
                animation: dashFlow {fw_speed:.2f}s linear infinite;
            }}
        </style>

        <!-- ==================================================
             STEAM GLOW
             ================================================== -->

        <filter id="steamGlow"
                x="-100%"
                y="-100%"
                width="300%"
                height="300%">

            <feGaussianBlur
                stdDeviation="4"
                result="blur"/>

            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>

        </filter>


        <!-- ==================================================
             FEEDWATER GLOW
             ================================================== -->

        <filter id="waterGlow"
                x="-100%"
                y="-100%"
                width="300%"
                height="300%">

            <feGaussianBlur
                stdDeviation="4"
                result="blur"/>

            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>

        </filter>


        <!-- ==================================================
             EQUIPMENT GLOW
             ================================================== -->

        <filter id="equipmentGlow"
                x="-100%"
                y="-100%"
                width="300%"
                height="300%">

            <feGaussianBlur
                stdDeviation="5"
                result="blur"/>

            <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
            </feMerge>

        </filter>


        <!-- ==================================================
             STEAM PARTICLE GRADIENT
             ================================================== -->

        <radialGradient id="steamParticle">

            <stop
                offset="0%"
                stop-color="#FFFFFF"
                stop-opacity="1"/>

            <stop
                offset="40%"
                stop-color="{STEAM_GLOW}"
                stop-opacity="0.95"/>

            <stop
                offset="100%"
                stop-color="{STEAM_COLOR}"
                stop-opacity="0"/>

        </radialGradient>


        <!-- ==================================================
             WATER PARTICLE GRADIENT
             ================================================== -->

        <radialGradient id="waterParticle">

            <stop
                offset="0%"
                stop-color="#FFFFFF"
                stop-opacity="1"/>

            <stop
                offset="40%"
                stop-color="{FW_GLOW}"
                stop-opacity="1"/>

            <stop
                offset="100%"
                stop-color="{FW_COLOR}"
                stop-opacity="0"/>

        </radialGradient>


        <!-- ==================================================
             ARROW MARKERS
             ================================================== -->

        <marker
            id="steamArrow"
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="5"
            orient="auto">

            <path
                d="M0,0 L10,5 L0,10 Z"
                fill="{STEAM_COLOR}"/>

        </marker>


        <marker
            id="waterArrow"
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="5"
            orient="auto">

            <path
                d="M0,0 L10,5 L0,10 Z"
                fill="{FW_COLOR}"/>

        </marker>


        <!-- ==================================================
             PROCESS LINE GLOW
             ================================================== -->

        <filter id="lineGlow"
                x="-20%"
                y="-100%"
                width="140%"
                height="300%">

            <feGaussianBlur
                stdDeviation="3"
                result="blur"/>

            <feMerge>
                <feMergeNode
                    in="blur"/>

                <feMergeNode
                    in="SourceGraphic"/>
            </feMerge>

        </filter>

    </defs>


    <!-- ======================================================
         STEAM PIPE GLOW BACKGROUND
         ====================================================== -->

    <path
        d="M60 220
           H370
           M410 220
           H550
           M610 220
           H700
           M900 220
           H1450"

        stroke="{STEAM_COLOR}"
        stroke-width="9"
        stroke-linecap="round"
        opacity="0.16"
        filter="url(#lineGlow)"
    />


    <!-- ======================================================
         MAIN STEAM PIPE (DYNAMIC DASH ANIMATION ADDED)
         ====================================================== -->

    <path
        d="M60 220 H370"
        stroke="{STEAM_COLOR}"
        stroke-width="5"
        stroke-linecap="round"
        class="animated-pipe"
        style="animation-duration: {steam_speed:.2f}s;"
    />

    <path
        d="M410 220 H550"
        stroke="{STEAM_COLOR}"
        stroke-width="5"
        stroke-linecap="round"
        class="animated-pipe"
        style="animation-duration: {steam_speed * 0.8:.2f}s;"
    />

    <path
        d="M610 220 H700"
        stroke="{STEAM_COLOR}"
        stroke-width="5"
        stroke-linecap="round"
        class="animated-pipe"
        style="animation-duration: {steam_speed * 0.9:.2f}s;"
    />

    <path
        d="M900 220 H1450"
        stroke="{STEAM_COLOR}"
        stroke-width="5"
        stroke-linecap="round"
        class="animated-pipe"
        style="animation-duration: {steam_speed * 1.2:.2f}s;"
    />


    <!-- ======================================================
         STEAM FLOW ARROWS
         ====================================================== -->

    <path
        d="M210 220 H275"
        stroke="{STEAM_COLOR}"
        stroke-width="2"
        marker-end="url(#steamArrow)"
        opacity="0.8"
    />

    <path
        d="M470 220 H520"
        stroke="{STEAM_COLOR}"
        stroke-width="2"
        marker-end="url(#steamArrow)"
        opacity="0.8"
    />

    <path
        d="M1120 220 H1190"
        stroke="{STEAM_COLOR}"
        stroke-width="2"
        marker-end="url(#steamArrow)"
        opacity="0.8"
    />


    <!-- ======================================================
         STEAM ANIMATION PATH
         ====================================================== -->

    <path
        id="steamPath"
        d="M60 220
           H370
           M410 220
           H550
           M610 220
           H700
           M900 220
           H1450"
        fill="none"
        stroke="none"
    />


    <!-- ======================================================
         STEAM PARTICLES
         ====================================================== -->

    <g filter="url(#steamGlow)">

        <circle
            r="7"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{steam_speed:.2f}s"
                repeatCount="indefinite"
                path="M60 220 H370"/>

        </circle>


        <circle
            r="5"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{steam_speed:.2f}s"
                begin="-{(steam_speed * 0.3):.2f}s"
                repeatCount="indefinite"
                path="M60 220 H370"/>

        </circle>


        <circle
            r="4"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{steam_speed:.2f}s"
                begin="-{(steam_speed * 0.65):.2f}s"
                repeatCount="indefinite"
                path="M60 220 H370"/>

        </circle>


        <circle
            r="7"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 0.65):.2f}s"
                repeatCount="indefinite"
                path="M410 220 H550"/>

        </circle>


        <circle
            r="5"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 0.65):.2f}s"
                begin="-{(steam_speed * 0.25):.2f}s"
                repeatCount="indefinite"
                path="M410 220 H550"/>

        </circle>


        <circle
            r="6"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 0.8):.2f}s"
                repeatCount="indefinite"
                path="M610 220 H700"/>

        </circle>


        <circle
            r="5"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 1.2):.2f}s"
                repeatCount="indefinite"
                path="M900 220 H1450"/>

        </circle>


        <circle
            r="4"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 1.2):.2f}s"
                begin="-{(steam_speed * 0.4):.2f}s"
                repeatCount="indefinite"
                path="M900 220 H1450"/>

        </circle>


        <circle
            r="6"
            fill="url(#steamParticle)">

            <animateMotion
                dur="{(steam_speed * 1.2):.2f}s"
                begin="-{(steam_speed * 0.8):.2f}s"
                repeatCount="indefinite"
                path="M900 220 H1450"/>

        </circle>

    </g>


    <!-- ======================================================
         PRESSURE CONTROL VALVE
         ====================================================== -->

    <g
        transform="translate(490,220)"
        filter="url(#equipmentGlow)">

        <!-- Valve body -->

        <polygon
            points="-35,-30 0,0 -35,30"
            fill="none"
            stroke="{EQUIP_COLOR}"
            stroke-width="3"/>

        <polygon
            points="35,-30 0,0 35,30"
            fill="none"
            stroke="{EQUIP_COLOR}"
            stroke-width="3"/>


        <!-- Valve stem (Animated stroke/position) -->

        <line
            x1="0"
            y1="-30"
            x2="0"
            y2="-65"
            stroke="{EQUIP_COLOR}"
            stroke-width="3">
            <animateTransform
                attributeName="transform"
                type="translate"
                values="0,0; 0,-3; 0,0"
                dur="2s"
                repeatCount="indefinite"/>
        </line>


        <!-- Actuator -->

        <circle
            cx="0"
            cy="-82"
            r="17"
            fill="none"
            stroke="{EQUIP_COLOR}"
            stroke-width="3"/>


        <!-- Animated actuator pulse -->

        <circle
            cx="0"
            cy="-82"
            r="20"
            fill="none"
            stroke="{EQUIP_GLOW}"
            stroke-width="2"
            opacity="0">

            <animate
                attributeName="r"
                values="18;28;18"
                dur="2.5s"
                repeatCount="indefinite"/>

            <animate
                attributeName="opacity"
                values="0.7;0;0.7"
                dur="2.5s"
                repeatCount="indefinite"/>

        </circle>

    </g>


    <!-- ======================================================
         DESUPERHEATER BODY
         ====================================================== -->

    <g
        transform="translate(800,220)"
        filter="url(#equipmentGlow)">

        <!-- Main body -->

        <path
            d="
            M-105 -42
            L-35 -16
            L35 -16
            L105 -42

            L105 42
            L35 16
            L-35 16
            L-105 42
            Z"

            fill="none"
            stroke="{EQUIP_COLOR}"
            stroke-width="3"
            stroke-linejoin="round"
        />


        <!-- Throat -->

        <line
            x1="-35"
            y1="-16"
            x2="35"
            y2="-16"
            stroke="{EQUIP_COLOR}"
            stroke-width="2"
        />

        <line
            x1="-35"
            y1="16"
            x2="35"
            y2="16"
            stroke="{EQUIP_COLOR}"
            stroke-width="2"
        />


        <!-- Internal spray line -->

        <line
            x1="0"
            y1="-75"
            x2="0"
            y2="-5"
            stroke="{FW_COLOR}"
            stroke-width="3"
        />


        <!-- Spray nozzle -->

        <circle
            cx="0"
            cy="-5"
            r="6"
            fill="{FW_COLOR}"
            filter="url(#waterGlow)"
        />

        <!-- Spray plume mist effect -->
        <path d="M0 -5 L35 -15 L45 0 L35 15 Z" fill="{FW_GLOW}" opacity="0.25">
            <animate attributeName="opacity" values="0.1;0.4;0.1" dur="0.8s" repeatCount="indefinite"/>
        </path>


        <!-- ==================================================
             SPRAY PARTICLES
             ================================================== -->

        <g filter="url(#waterGlow)">

            <circle
                r="5"
                fill="url(#waterParticle)">

                <animateMotion
                    dur="1.0s"
                    repeatCount="indefinite"
                    path="M0 -5 L35 5"/>

            </circle>


            <circle
                r="4"
                fill="url(#waterParticle)">

                <animateMotion
                    dur="1.2s"
                    begin="-0.4s"
                    repeatCount="indefinite"
                    path="M0 -5 L45 -8"/>

            </circle>


            <circle
                r="3"
                fill="url(#waterParticle)">

                <animateMotion
                    dur="0.9s"
                    begin="-0.2s"
                    repeatCount="indefinite"
                    path="M0 -5 L40 12"/>

            </circle>

        </g>

    </g>


    <!-- ======================================================
         FEEDWATER PIPE
         ====================================================== -->

    <path
        d="M800 75 V215"
        stroke="{FW_COLOR}"
        stroke-width="5"
        stroke-linecap="round"
        class="fw-pipe"
    />


    <!-- Feedwater pipe glow -->

    <path
        d="M800 75 V215"
        stroke="{FW_COLOR}"
        stroke-width="10"
        stroke-linecap="round"
        opacity="0.15"
        filter="url(#lineGlow)"
    />


    <!-- Feedwater arrow -->

    <path
        d="M800 110 V160"
        stroke="{FW_COLOR}"
        stroke-width="2"
        marker-end="url(#waterArrow)"
    />


    <!-- ======================================================
         FEEDWATER PARTICLES
         ====================================================== -->

    <g filter="url(#waterGlow)">

        <circle
            r="7"
            fill="url(#waterParticle)">

            <animateMotion
                dur="{fw_speed:.2f}s"
                repeatCount="indefinite"
                path="M800 75 V215"/>

        </circle>


        <circle
            r="5"
            fill="url(#waterParticle)">

            <animateMotion
                dur="{fw_speed:.2f}s"
                begin="-{(fw_speed * 0.33):.2f}s"
                repeatCount="indefinite"
                path="M800 75 V215"/>

        </circle>


        <circle
            r="4"
            fill="url(#waterParticle)">

            <animateMotion
                dur="{fw_speed:.2f}s"
                begin="-{(fw_speed * 0.66):.2f}s"
                repeatCount="indefinite"
                path="M800 75 V215"/>

        </circle>

    </g>


    <!-- ======================================================
         LABELS
         ====================================================== -->

    <!-- Inlet -->

    <text
        x="60"
        y="105"
        fill="{TEXT_COLOR}"
        font-size="18"
        font-weight="600">

        High Pressure Steam

    </text>

    <text
        x="60"
        y="132"
        fill="{TEXT_COLOR}"
        font-size="14">

        Flow: {m_in:.2f} t/h

    </text>

    <text
        x="60"
        y="153"
        fill="{TEXT_COLOR}"
        font-size="14">

        Pressure: {p_in:.2f} {p_unit}

    </text>

    <text
        x="60"
        y="174"
        fill="{TEXT_COLOR}"
        font-size="14">

        Temperature: {t_in:.1f} °C

    </text>


    <!-- PCV label -->

    <text
        x="490"
        y="330"
        text-anchor="middle"
        fill="{TEXT_COLOR}"
        font-size="15"
        font-weight="600">

        Pressure Control Valve

    </text>

    <text
        x="490"
        y="350"
        text-anchor="middle"
        fill="{EQUIP_COLOR}"
        font-size="13">

        Isenthalpic Expansion

    </text>


    <!-- Feedwater -->

    <text
        x="825"
        y="60"
        fill="{FW_COLOR}"
        font-size="17"
        font-weight="600">

        Feedwater Spray

    </text>

    <text
        x="825"
        y="82"
        fill="{FW_COLOR}"
        font-size="13">

        {m_fw:.2f} t/h

    </text>


    <!-- Desuperheater -->

    <text
        x="800"
        y="300"
        text-anchor="middle"
        fill="{TEXT_COLOR}"
        font-size="16"
        font-weight="600">

        DESUPERHEATER

    </text>

    <text
        x="800"
        y="322"
        text-anchor="middle"
        fill="{FW_COLOR}"
        font-size="13">

        Spray Cooling

    </text>


    <!-- Outlet -->

    <text
        x="1100"
        y="105"
        fill="{TEXT_COLOR}"
        font-size="18"
        font-weight="600">

        Low Pressure Steam

    </text>

    <text
        x="1100"
        y="132"
        fill="{TEXT_COLOR}"
        font-size="14">

        Flow: {m_out:.2f} t/h

    </text>

    <text
        x="1100"
        y="153"
        fill="{TEXT_COLOR}"
        font-size="14">

        Pressure: {p_out:.2f} {p_unit}

    </text>

    <text
        x="1100"
        y="174"
        fill="{TEXT_COLOR}"
        font-size="14">

        Temperature: {t_out:.1f} °C

    </text>


    <!-- ======================================================
         DIGITAL FLOW INDICATORS
         ====================================================== -->

    <text
        x="220"
        y="255"
        fill="{STEAM_GLOW}"
        font-size="11"
        opacity="0.8">

        ● FLOW

        <animate
            attributeName="opacity"
            values="0.3;1;0.3"
            dur="1.5s"
            repeatCount="indefinite"/>

    </text>


    <text
        x="815"
        y="120"
        fill="{FW_GLOW}"
        font-size="10"
        opacity="0.8">

        ● INJECTION

        <animate
            attributeName="opacity"
            values="0.3;1;0.3"
            dur="1.2s"
            repeatCount="indefinite"/>

    </text>


    <!-- ======================================================
         SMALL DIGITAL DOTS
         ====================================================== -->

    <circle
        cx="380"
        cy="220"
        r="3"
        fill="{EQUIP_COLOR}">

        <animate
            attributeName="r"
            values="2;5;2"
            dur="1.5s"
            repeatCount="indefinite"/>

        <animate
            attributeName="opacity"
            values="0.4;1;0.4"
            dur="1.5s"
            repeatCount="indefinite"/>

    </circle>


    <circle
        cx="910"
        cy="220"
        r="3"
        fill="{EQUIP_COLOR}">

        <animate
            attributeName="r"
            values="2;5;2"
            dur="1.5s"
            begin="0.5s"
            repeatCount="indefinite"/>

    </circle>


    </svg>
    """

    return svg


# ==============================================================
# APPLICATION TITLE
# ==============================================================

st.title("💨 Desuperheater Letdown Mass & Energy Balance")

st.caption(
    "Developed by Iqbal SHERPA 20260708. "
    "Contact me for further information @iqbalshafiq96@gmail.com"
)


# ==============================================================
# SIDEBAR
# ==============================================================

st.sidebar.header("Configuration")


Pressure_Unit_Type = st.sidebar.selectbox(
    "Pressure Unit Type",
    [
        "Bar Gauge (barG)",
        "Bar Absolute (barA)",
        "Megapascals Gauge (MPaG)",
        "Megapascals Absolute (MPaA)",
    ],
)
