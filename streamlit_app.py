import streamlit as st
from streamlit_folium import folium_static

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from folium.plugins import HeatMapWithTime
import folium


@st.cache
def load_data():
    df = pd.read_csv("data/aus_clean_data.csv")
    return df


@st.cache
def load_map_data():
    df = pd.read_csv("data/aus_clean_map_data.csv")
    return df


def get_corr_heatmap(df):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr,
                mask=mask,
                cmap=cmap,
                vmax=.3,
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    return plt


def get_max_temp_bar_chart(df):
    loc_max_temp = df[["Location", "MaxTemp"]].groupby("Location").mean()
    loc_max_temp.sort_values(by="MaxTemp", inplace=True)
    labels = {"MaxTemp": "Maximum Temperature °C", "Location": "Location Names"}
    fig = px.bar(loc_max_temp,
                 labels=labels,
                 color_discrete_sequence=["lightcoral"])
    return fig


def get_min_temp_bar_chart(df):
    loc_min_temp = df[["Location", "MinTemp"]].groupby("Location").mean()
    loc_min_temp.sort_values(by="MinTemp", inplace=True)
    labels = {"MinTemp": "Minumum Temperature °C", "Location": "Location Names"}
    fig = px.bar(loc_min_temp,
                 labels=labels,
                 color_discrete_sequence=["lightslategray"])
    return fig


def get_rain_bar_chart(df):
    loc_rain = df[["Location", "Rainfall"]].groupby("Location").mean()
    loc_rain.sort_values(by="Rainfall", inplace=True)
    labels = {"Rainfall": "Amount of Rainfall", "Location": "Location Names"}
    fig = px.bar(loc_rain, labels=labels, color_discrete_sequence=["skyblue"])
    return fig

def get_weather_map(map_data):
    date = map_data["Date"]
    date_choice = st.selectbox("Select date:", date)
    date_df = map_data.loc[map_data.Date == date_choice]
    cols = ["MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
            "WindSpeed9am", "Humidity9am", "Pressure9am", "Cloud9am", "Temp9am"]
    col_option = st.multiselect("Select Features", cols, cols[0])
    aus_plot = folium.Map(location=[-28.0, 135],
                          control_scale=True,
                          zoom_start=4.3,
                          tiles="CartoDB positron")
    for i in range(0, len(date_df)):
        tool_tip = ""
        if "MinTemp" in col_option:
            tool_tip += f"MinTemp: {date_df.iloc[i]['MinTemp']}<br/>"
        if "MaxTemp" in col_option:
            tool_tip += f"MaxTemp: {date_df.iloc[i]['MaxTemp']}<br/>"
        if "Rainfall" in col_option:
            tool_tip += f"Rainfall: {date_df.iloc[i]['Rainfall']}<br/>"
        if "Evaporation" in col_option:
            tool_tip += f"Evaporation: {date_df.iloc[i]['Evaporation']}<br/>"
        if "Sunshine" in col_option:
            tool_tip += f"Sunshine: {date_df.iloc[i]['Sunshine']}<br/>"
        if "WindGustSpeed" in col_option:
            tool_tip += f"WindGustSpeed: {date_df.iloc[i]['WindGustSpeed']}<br/>"
        if "WindSpeed9am" in col_option:
            tool_tip += f"WindSpeed9am: {date_df.iloc[i]['WindSpeed9am']}<br/>"
        if "Humidity9am" in col_option:
            tool_tip += f"Humidity9am: {date_df.iloc[i]['Humidity9am']}<br/>"
        if "Pressure9am" in col_option:
            tool_tip += f"Pressure9am: {date_df.iloc[i]['Pressure9am']}<br/>"
        if "Cloud9am" in col_option:
            tool_tip += f"Cloud9am: {date_df.iloc[i]['Cloud9am']}<br/>"
        if "Temp9am" in col_option:
            tool_tip += f"Temp9am: {date_df.iloc[i]['Temp9am']}<br/>"
        # "MinTemp: " + str(date_df.iloc[i]["MinTemp"]) +
        #             "<br/> MaxTemp: " + str(date_df.iloc[i]["MaxTemp"])
        folium.Marker(
            location=[date_df.iloc[i]["lat"], date_df.iloc[i]["lng"]],
            tooltip=tool_tip,
            icon=folium.Icon(color="blue",
                             prefix="fa fas fa-cloud")).add_to(aus_plot)
    return aus_plot

# Main
st.title("Australia Rain Prediction")
image = Image.open("data/aus_climate.jpg")
st.image(image)

data = load_data()
map_data = load_map_data()

if st.sidebar.checkbox("Show Data"):
    st.write("## Dataset:")
    st.write(data.head())
if st.sidebar.checkbox("Show Feature Correlations"):
    st.write("## Features Correlation:")
    st.pyplot(get_corr_heatmap(data))
st.sidebar.header("Features:")
if st.sidebar.checkbox("Show Max Tempreature"):
    st.write("## Cities with High Tempreature:")
    st.plotly_chart(get_max_temp_bar_chart(data))
if st.sidebar.checkbox("Show Min Tempreature"):
    st.write("## Cities with Minimum Tempreature:")
    st.plotly_chart(get_min_temp_bar_chart(data))
if st.sidebar.checkbox("Show Rainfall"):
    st.write("## Cities with Rainfall:")
    st.plotly_chart(get_rain_bar_chart(data))
st.sidebar.header("Maps:")
if st.sidebar.checkbox("Show Temp Map"):
    st.write("## Cities Map with Weather Markers:")
    aus_plot = get_weather_map(map_data)
    folium_static(aus_plot)