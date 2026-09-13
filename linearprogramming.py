"""
app.py

Two-page LP optimizer Streamlit app — single file.
Page 1: worked refinery crude-oil purchasing example (readable variable names).
Page 2: user builds their own LP using plain-word variable names (e.g. Utility,
RawMaterial) — declared variables update live as the user types.

Both pages share the exact same solving pipeline (solve_lp) and the exact
same result-display logic, so behavior is consistent between them.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
from scipy.optimize import linprog

st.set_page_config(page_title="LP Optimizer", page_icon="📈", layout="centered")

IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# ========================================================================
# LP ENGINE — parsing, linearity check, solving
# ========================================================================
def extract_identifiers(text):
    """Find every word-like token in a string (candidate variable names)."""
    return set(IDENTIFIER_RE.findall(text))


def build_symbol_locals(*texts):
    """
    Build a dict of plain sympy Symbols for every word found across the given
