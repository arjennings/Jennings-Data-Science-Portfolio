import streamlit as st
st.title("Jennings Streamlit App")
st.write("This app displays stats about Palmer's Penguins")

import pandas as pd
st.write("Here's a basic table")
penguins_df = pd.read_csv("basic streamlit/data/penguins.csv")
st.dataframe(penguins_df)
island = st.selectbox("Island", penguins_df["Island"].unique())
st.slider("flipper length", float(penguins_df["Flipper Length (mm)"].min()), float(penguins_df["Flipper Length (mm)"].max()))
