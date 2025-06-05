import streamlit as st
import plotly.express as px
import pandas as pd
from pyspark.sql import SparkSession

# Pornire Spark
spark = SparkSession.builder.appName("Flight Delays Analysis").getOrCreate()

st.title("Flight Delay & Aircraft Info Dashboard")

# 1. Date întârzieri
file_path = "Airline_Delay_Cause.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Filtrare ani mai mici de 2020
df_spark = df.filter(~df.year.isin(2020, 2021, 2022))
st.subheader("Primele 5 linii din dataset întârzieri:")
st.write(df_spark.limit(5).toPandas())

# Histogramă arr_delay
df_spark_pd = df_spark.toPandas()
fig = px.histogram(df_spark_pd, x="arr_delay", nbins=50, title="Distribution of Arrival Delays")
st.plotly_chart(fig)

# Întârzieri medii pe an
avg_delays = df_spark.groupBy("year").agg(
    {"arr_delay": "avg", "carrier_delay": "avg", "weather_delay": "avg",
     "nas_delay": "avg", "security_delay": "avg", "late_aircraft_delay": "avg"}
).withColumnRenamed("avg(arr_delay)", "avg_arr_delay") \
 .withColumnRenamed("avg(carrier_delay)", "avg_carrier_delay") \
 .withColumnRenamed("avg(weather_delay)", "avg_weather_delay") \
 .withColumnRenamed("avg(nas_delay)", "avg_nas_delay") \
 .withColumnRenamed("avg(security_delay)", "avg_security_delay") \
 .withColumnRenamed("avg(late_aircraft_delay)", "avg_late_aircraft_delay")

avg_delays_pd = avg_delays.toPandas()
st.subheader("Average delays per year:")
st.write(avg_delays_pd)

fig_avg_delays = px.bar(
    avg_delays_pd,
    x="year",
    y=["avg_arr_delay", "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
       "avg_security_delay", "avg_late_aircraft_delay"],
    title="Average Delays Per Year"
)
st.plotly_chart(fig_avg_delays)

# 2. Date segment zboruri și lookup AircraftType
st.subheader("Flight segments with aircraft details")

# Încarcă CSV-uri cu pandas
df_main = pd.read_csv("T_T100D_SEGMENT_US_CARRIER_ONLY.csv")
df_lookup = pd.read_csv("T_AIRCRAFT_TYPES.csv")

# Asigură tipul string pentru join
df_main['AIRCRAFT_TYPE'] = df_main['AIRCRAFT_TYPE'].astype(str)
df_lookup['AC_TYPEID'] = df_lookup['AC_TYPEID'].astype(str)

# Join pe codul avionului
df_merged = pd.merge(df_main, df_lookup, how='left', left_on='AIRCRAFT_TYPE', right_on='AC_TYPEID')

# Afișează câteva coloane relevante
st.write(df_merged[['CARRIER', 'AIRCRAFT_TYPE', 'MANUFACTURER', 'SSD_NAME']].head(10))

# Încarcă CSV-ul cu întârzieri (din exemplul tău)
df_delays = pd.read_csv("Airline_Delay_Cause.csv")  # înlocuiește cu numele corect al fișierului

# Grupare întârzieri totale pe carrier
delays_grouped = df_delays.groupby('carrier')['arr_delay'].sum().reset_index()

# Număr zboruri per tip avion (SSD_NAME) și carrier
agg = df_merged.groupby(['CARRIER', 'SSD_NAME']).size().reset_index(name='num_flights')

# Join cu întârzierile totale pe carrier
agg = agg.merge(delays_grouped, how='left', left_on='CARRIER', right_on='carrier')
agg['arr_delay'] = agg['arr_delay'].fillna(0)

# Grafic bară cu întârzieri după tip avion (SSD_NAME) și carrier
fig = px.bar(agg, x='SSD_NAME', y='arr_delay', color='CARRIER',
             title="Total Arrival Delay by Aircraft Type (SSD_NAME) and Carrier",
             labels={'arr_delay': 'Total Arrival Delay', 'SSD_NAME': 'Aircraft Type (SSD_NAME)'})

st.plotly_chart(fig)

# Oprește Spark
spark.stop()
