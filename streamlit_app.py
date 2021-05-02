import streamlit as st
from streamlit_folium import folium_static

from pycaret.classification import *

from folium.plugins import HeatMapWithTime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
import altair as alt
from windrose import WindroseAxes

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


def get_wind_speed_plot(df):
    df = df.iloc[:500, :]
    plt.figure(figsize=[20, 20])
    plt.subplot(311)
    plt.plot(df['Date'], df['WindSpeed9am'],
             color='blue',
             linewidth=2,
             label='WindSpeed9am')
    plt.legend(loc='upper right')
    plt.title('Wind Gust Speed at 9AM')
    plt.subplot(312)
    plt.plot(df['Date'], df['WindSpeed3pm'],
             color='green',
             linewidth=2,
             label='WindSpeed3pm')
    plt.legend(loc='upper right')
    plt.title('Wind Gust Speed at 3PM')
    plt.subplot(313)
    plt.plot(df['Date'], df['WindGustSpeed'],
             color='violet',
             linewidth=2,
             label='WindGustSpeed')
    plt.legend(loc='upper right')
    plt.title('Wind Gust Speed')
    return plt


def get_wind_speed_altair_plot(df):
    wind_df = df[:1000]
    wind_df['Date'] = pd.to_datetime(wind_df['Date'])
    # cols = ["WindSpeed9am", "WindSpeed3pm"]
    # fig1 = alt.Chart(wind_df).mark_line(
    #     color='lightgreen'
    # ).encode(
    #     x='Date:T',
    #     y='WindSpeed9am:Q'
    # )
    # fig2 = alt.Chart(wind_df).mark_line(
    #     color='red'
    # ).encode(
    #     x='Date:T',
    #     y='WindSpeed3pm:Q'
    # )
    # alt_fig_all = fig1 + fig2
    # alt_fig_all.properties(
    #     title='Wind Speed',
    #     width=600,
    #     height=400
    # ).interactive()
    # return alt_fig_all
    chart = alt.Chart(wind_df).mark_point().encode(
        x='Date:T',
        y='WindGustSpeed:Q',
    ).properties(
        width=800,
        height=400
    ).interactive()
    return chart


def max_temp_evaporation_plot(df):
    plt.figure(figsize=(5, 5))
    sns.jointplot(data=df, x='MaxTemp', y='Evaporation', bins=100,
                  kind='hex', gridsize=30, marginal_kws={'color': '#e76f51'})
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 200})
    plt.xlabel('MaxTemp', fontsize=13, labelpad=15)
    plt.ylabel('Evaporation', fontsize=13, labelpad=15)
    return plt


def get_humidity_plot(df):
    df = df.iloc[:100]
    plt.figure(figsize=[20, 7])
    plt.plot(df['Date'], df['WindSpeed9am'], linewidth=2, label='WindSpeed9am')
    plt.plot(df['Date'], df['WindSpeed3pm'], linewidth=2, label='WindSpeed3pm')
    plt.legend(loc='upper left')
    plt.title('Humidity9am vs Humidity3pm by Date')
    return plt


def get_humidity_altair_plot(df):
    df = df.iloc[:100]
    df = df[["Date", "Humidity9am", "Humidity3pm"]]
    df = df.melt('Date', var_name='name', value_name='value')
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date:T'),
        y=alt.Y('value:Q'),
        color=alt.Color("name:N")
    ).properties(
        width=800,
        height=400
    ).interactive()
    return chart


def get_wind_dir_plot(df):
    ws = df['WindGustSpeed'].to_numpy()
    wd = df['WindGustDir_cat'].to_numpy() * 16
    ax = WindroseAxes.from_ax()
    ax.set_xticklabels(['N', 'NW',  'W', 'SW', 'S', 'SE', 'E', 'NE'])
    ax.set_theta_zero_location('N')
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
    ax.set_legend()
    return plt


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


def get_predictions(saved_model):
    st.write("## Predict Rainfall")
    st.write("Predictions Based on Trained Random Forest Classifer Model")
    input_cols = ['Location_cat', 'MinTemp', 'MaxTemp', 'Rainfall',
                  'Evaporation', 'Sunshine', 'WindGustDir_cat',
                  'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                  'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                  'Temp9am', 'Temp3pm', 'RainToday_cat']
    col1, col2 = st.beta_columns(2)
    col3_row1, col3_row2, col3_row3 = st.beta_columns(3)
    MinTemp = col1.slider("Mininum Tempreature",
                          data.MinTemp.min(), data.MinTemp.max(), 9.7, step=0.1)
    MaxTemp = col2.slider("Maximum Tempreature",
                          data.MaxTemp.min(), data.MaxTemp.max(), 31.9, step=0.1)
    Rainfall = col1.slider("Rainfall",
                           data.Rainfall.min(), data.Rainfall.max(), 0.0, step=0.1)
    Evaporation = col2.slider("Evaporation",
                              data.Evaporation.min(), data.Evaporation.max(), 5.4, step=0.1)
    Sunshine = col1.slider("Sunshine",
                           data.Sunshine.min(), data.Sunshine.max(), 7.6, step=0.1)
    WindGustSpeed = col2.slider("Wind Gust Speed",
                                data.WindGustSpeed.min(), data.WindGustSpeed.max(), 80.0, step=0.1)
    Humidity9am = col1.slider("Humidity at 9am",
                              data.Humidity9am.min(), data.Humidity9am.max(), 42.0, step=0.1)
    Humidity3pm = col2.slider("Humidity at 3pm",
                              data.Humidity3pm.min(), data.Humidity3pm.max(), 9.0, step=0.1)
    Pressure9am = col1.slider("Wind Pressure at 9am",
                              data.Pressure9am.min(), data.Pressure9am.max(), 1008.9, step=0.1)
    Pressure3pm = col2.slider("Wind Pressure at 3pm",
                              data.Pressure3pm.min(), data.Pressure3pm.max(), 1003.6, step=0.1)
    Cloud9am = col1.slider("Cloud at 9am",
                           data.Cloud9am.min(), data.Cloud9am.max(), 4.4474612602152455, step=0.1)
    Cloud3pm = col2.slider("Cloud at 3pm",
                           data.Cloud3pm.min(), data.Cloud3pm.max(), 4.509930082924903, step=0.1)
    Temp9am = col1.slider("Tempreature at 9am",
                          data.Temp9am.min(), data.Temp9am.max(), 18.3, step=0.1)
    Temp3pm = col2.slider("Tempreature at 3pm",
                          data.Temp3pm.min(), data.Temp3pm.max(), 30.2, step=0.1)
    Location_cat = col3_row1.selectbox("City Location", data.Location.unique())
    WindGustDir_cat = col3_row2.selectbox("Wind Gust Direction", data.WindGustDir.unique())
    RainToday_cat = col3_row3.selectbox("Did it Rain Today?", data.RainToday.unique())
    Location_index = np.where(data.Location.unique() == Location_cat)[0][0]
    WindGustDir_index = np.where(data.WindGustDir.unique() == WindGustDir_cat)[0][0]
    RainToday_index = np.where(data.RainToday.unique() == RainToday_cat)[0][0]
    input_data = [[Location_index, MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine,
                   WindGustDir_index, WindGustSpeed, Humidity9am, Humidity3pm,
                   Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm,
                   RainToday_index]]
    input_df = pd.DataFrame(input_data, columns=input_cols)
    predictions = predict_model(saved_model, data=input_df)
    rain_tomorrow = int(predictions["Label"])
    return rain_tomorrow


# Main
st.title("Australia Rain Prediction")
image = Image.open("data/aus_climate.jpg")
placeholder = st.image(image)

data = load_data()
map_data = load_map_data()
saved_model = load_model('rf_model')

st.sidebar.header("Dataset:")
if st.sidebar.checkbox("Show Data"):
    placeholder.empty()
    st.write("## Dataset")
    st.write(data.head())

if st.sidebar.checkbox("Show Feature Correlations"):
    placeholder.empty()
    st.write("## Features Correlation")
    st.pyplot(get_corr_heatmap(data))

st.sidebar.header("Features:")
if st.sidebar.checkbox("Show Max Tempreature"):
    placeholder.empty()
    st.write("## Cities with High Tempreature")
    st.plotly_chart(get_max_temp_bar_chart(data))
if st.sidebar.checkbox("Show Min Tempreature"):
    placeholder.empty()
    st.write("## Cities with Minimum Tempreature")
    st.plotly_chart(get_min_temp_bar_chart(data))
if st.sidebar.checkbox("Show Rainfall"):
    placeholder.empty()
    st.write("## Rainfall Rate")
    st.plotly_chart(get_rain_bar_chart(data))
if st.sidebar.checkbox("Show Evaporation"):
    placeholder.empty()
    st.write("## Evaporation Rate")
    st.plotly_chart(get_evaporation_scatter_chart(data))
if st.sidebar.checkbox("Show Wind Speed"):
    placeholder.empty()
    st.write("## Wind Speed")
    # st.pyplot(get_wind_speed_plot(data))
    st.altair_chart(get_wind_speed_altair_plot(data))
if st.sidebar.checkbox("Show Humidity"):
    placeholder.empty()
    st.write("## Humidity")
    # st.pyplot(get_humidity_plot(data))
    st.altair_chart(get_humidity_altair_plot(data))
if st.sidebar.checkbox("Show Wind Directions"):
    placeholder.empty()
    st.write("## Wind Directions")
    st.pyplot(get_wind_dir_plot(data))
if st.sidebar.checkbox("Show Comparisons"):
    placeholder.empty()
    st.write("## Max Tempreature vs Evaporation")
    st.pyplot(max_temp_evaporation_plot(data))

st.sidebar.header("Maps:")
if st.sidebar.checkbox("Show Temp Map"):
    placeholder.empty()
    st.write("## Cities Map with Weather Markers")
    aus_plot = get_weather_map(map_data)
    folium_static(aus_plot)
if st.sidebar.checkbox("Show Rainfall Timeseries Map"):
    placeholder.empty()
    st.write("## Rainfall Timeseries")
    timeseries_plot = get_rainfall_timeseries_map(map_data)
    folium_static(timeseries_plot)

st.sidebar.header("Rainfall Prediction:")
if st.sidebar.checkbox("Predict"):
    placeholder.empty()
    rain_tomorrow = get_predictions(saved_model)
    if int(rain_tomorrow) == 0:
        st.write("## It will not rain tomorrow!")
    else:
        st.write("## It will rain tomorrow!")
