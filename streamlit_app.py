import streamlit as st
from streamlit_folium import folium_static

from pycaret.classification import *

from folium.plugins import HeatMapWithTime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

import numpy as np
import pandas as pd

from PIL import Image

st.markdown(
    """
    <style>
    .reportview-container {
        background-image: url("https://d8st7idcnjoas.cloudfront.net/galfull/DPF-204.jpg");
        background-position: bottom;
        background-repeat: no-repeat;
        background-size: cover;
        width: 100%;
    }
   .sidebar .sidebar-content {
        background-color: black;
    }
    h1,h2,h3,h4,p {
        color: rgb(0, 0, 0);
    }
    .css-1l02zno {
    background-color: #cce6ff;
    }
    </style>
    """, unsafe_allow_html=True)


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


def get_evaporation_scatter_chart(df):
    loc_evaporation = df[["Location", "Evaporation"]].groupby("Location").mean()
    loc_evaporation.sort_values(by="Evaporation", inplace=True)
    labels = {"Evaporation": "mm/h", "Location": "Location Names"}
    fig = px.scatter(loc_evaporation, labels=labels,
                     color_discrete_sequence=['lightcoral'])
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


def get_rainfall_timeseries_map(map_df):
    dfmap = map_df[['Date', 'lat', 'lng', 'Rainfall']]
    df_day_list = []

    for day in dfmap["Date"].sort_values().unique():
        data = dfmap.loc[
            dfmap["Date"] == day,
            ['Date', 'lat', 'lng', 'Rainfall']].groupby(
            ['lat', 'lng']).sum().reset_index().values.tolist()
        df_day_list.append(data)

    ts_rain_map = folium.Map([-28.0, 135],
                             zoom_start=4.3,
                             tiles='CartoDB positron')
    HeatMapWithTime(df_day_list,
                    index=list(dfmap["Date"].sort_values().unique()),
                    auto_play=False,
                    radius=10,
                    gradient={
                        0.2: 'lightskyblue',
                        0.4: 'skyblue',
                        0.6: 'steelblue',
                        1.0: 'darkcyan'
                    },
                    min_opacity=0.5,
                    max_opacity=0.8,
                    use_local_extrema=True).add_to(ts_rain_map)
    return ts_rain_map


def get_predictions():
    data = load_data()
    saved_model = load_model('rf_model')
    st.write("## Predict Rainfall")
    st.write("Predictions Based on Trained Random Forest Classifer Model")
    input_cols = ['Location_cat', 'MinTemp', 'MaxTemp', 'Rainfall',
                  'Evaporation', 'Sunshine', 'WindGustDir_cat',
                  'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                  'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                  'Temp9am', 'Temp3pm', 'RainToday_cat']
    # input_data = [[3, 13, 22, 0.6, 5.4, 7.6, 13, 44, 70, 22,
    #                1007.2, 1007.1, 8.0, 21.8, 16.2, 21.7, 0]]
    col1, col2 = st.beta_columns(2)
    Location_cat = col1.text_input("Location_cat", 0)
    MinTemp = col1.slider("MinTemp",
                          data.MinTemp.min(), data.MinTemp.max(), 9.7, step=0.1)
    MaxTemp = col1.slider("MaxTemp",
                          data.MaxTemp.min(), data.MaxTemp.max(), 31.9, step=0.1)
    Rainfall = col1.slider("Rainfall",
                           data.Rainfall.min(), data.Rainfall.max(), 0.0, step=0.1)
    Evaporation = col1.slider("Evaporation",
                              data.Evaporation.min(), data.Evaporation.max(), 5.4, step=0.1)
    Sunshine = col1.slider("Sunshine",
                           data.Sunshine.min(), data.Sunshine.max(), 7.6, step=0.1)
    WindGustDir_cat = col1.text_input("WindGustDir_cat", 6)
    WindGustSpeed = col1.slider("WindGustSpeed",
                                data.WindGustSpeed.min(), data.WindGustSpeed.max(), 80.0, step=0.1)
    Humidity9am = col2.slider("Humidity9am",
                              data.Humidity9am.min(), data.Humidity9am.max(), 42.0, step=0.1)
    Humidity3pm = col2.slider("Humidity3pm",
                              data.Humidity3pm.min(), data.Humidity3pm.max(), 9.0, step=0.1)
    Pressure9am = col2.slider("Pressure9am",
                              data.Pressure9am.min(), data.Pressure9am.max(), 1008.9, step=0.1)
    Pressure3pm = col2.slider("Pressure3pm",
                              data.Pressure3pm.min(), data.Pressure3pm.max(), 1003.6, step=0.1)
    Cloud9am = col2.slider("Cloud9am",
                           data.Cloud9am.min(), data.Cloud9am.max(), 4.4474612602152455, step=0.1)
    Cloud3pm = col2.slider("Cloud3pm",
                           data.Cloud3pm.min(), data.Cloud3pm.max(), 4.509930082924903, step=0.1)
    Temp9am = col2.slider("Temp9am",
                          data.Temp9am.min(), data.Temp9am.max(), 18.3, step=0.1)
    Temp3pm = col2.slider("Temp3pm",
                          data.Temp3pm.min(), data.Temp3pm.max(), 30.2, step=0.1)
    RainToday_cat = col2.text_input("RainToday_cat", 0)
    input_data = [[Location_cat, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
                   WindGustDir_cat, WindGustSpeed, Humidity9am, Humidity3pm,
                   Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm,
                   RainToday_cat]]
    input_df = pd.DataFrame(input_data, columns=input_cols)
    st.write(input_df)
    predictions = predict_model(saved_model, data=input_df)
    rain_tomorrow = int(predictions["Label"])
    return rain_tomorrow


# Main
st.title("Australia Rain Prediction")
image = Image.open("data/aus_climate.jpg")
placeholder = st.image(image)

data = load_data()
map_data = load_map_data()

st.sidebar.header("Dataset:")
if st.sidebar.checkbox("Show Data"):
    placeholder.empty()
    st.write("## Dataset:")
    st.write(data.head())
if st.sidebar.checkbox("Show Feature Correlations"):
    placeholder.empty()
    st.write("## Features Correlation:")
    st.pyplot(get_corr_heatmap(data))
st.sidebar.header("Features:")
if st.sidebar.checkbox("Show Max Tempreature"):
    placeholder.empty()
    st.write("## Cities with High Tempreature:")
    st.plotly_chart(get_max_temp_bar_chart(data))
if st.sidebar.checkbox("Show Min Tempreature"):
    placeholder.empty()
    st.write("## Cities with Minimum Tempreature:")
    st.plotly_chart(get_min_temp_bar_chart(data))
if st.sidebar.checkbox("Show Rainfall"):
    placeholder.empty()
    st.write("## Cities with Rainfall:")
    st.plotly_chart(get_rain_bar_chart(data))
if st.sidebar.checkbox("Show Evaporation"):
    placeholder.empty()
    st.write("## Evaporation rate of the Cities:")
    st.plotly_chart(get_evaporation_scatter_chart(data))
st.sidebar.header("Maps:")
if st.sidebar.checkbox("Show Temp Map"):
    placeholder.empty()
    st.write("## Cities Map with Weather Markers:")
    aus_plot = get_weather_map(map_data)
    folium_static(aus_plot)
if st.sidebar.checkbox("Show Rainfall Timeseries Map"):
    placeholder.empty()
    st.write("## Rainfall Timeseries:")
    timeseries_plot = get_rainfall_timeseries_map(map_data)
    folium_static(timeseries_plot)
st.sidebar.header("Rainfall Prediction:")
if st.sidebar.checkbox("Predict"):
    placeholder.empty()
    rain_tomorrow = get_predictions()
    if int(rain_tomorrow) == 0:
        st.write("## It will not rain tomorrow!")
    else:
        st.write("## It will rain tomorrow!")
