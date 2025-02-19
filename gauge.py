import streamlit as st
import streamviz

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


streamviz.gauge(
    gVal=.70,
)
gauge_value = st.slider("Set Gauge Value", min_value=-1.0, max_value=1.0, step=0.01, value=0.30)

def get_color(value):
    # Map the value (-1 to 1) to a range (0 to 255) for red and green
    red = int((1 - value) * 127.5)
    green = int((value + 1) * 127.5)
    return f"rgb({red},{green},0)"

# Get dynamic color for the bar
bar_color = get_color(gauge_value)

# Create gauge figure
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=gauge_value,
    gauge={
        "axis": {"range": [-1, 1]},  # Range of the gauge
        "bar": {"color": bar_color},  # Bar color
        "steps": [
            {"range": [-1, 1], "color": "lightgray"}
            # {"range": [0, 1], "color": "green"},
        ],
    },
    number={"suffix": ""}  # Optional suffix
))

# Display the gauge
st.plotly_chart(fig)


