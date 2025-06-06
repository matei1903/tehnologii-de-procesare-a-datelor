import streamlit as st
import plotly.express as px
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

# --- Spark init ---
spark = SparkSession.builder.appName("Flight Delays Analysis").getOrCreate()

st.title("Flight Delay & Aircraft Info Dashboard")

# === ÎNCĂRCARE DATE ÎNTÂRZIERI ===
df = spark.read.csv("Airline_Delay_Cause.csv", header=True, inferSchema=True)
df_spark = df.filter(~df.year.isin(2020, 2021, 2022))

df_spark_pd = df_spark.toPandas()
fig = px.histogram(
    df_spark_pd,
    x="arr_delay",
    nbins=50,
    title="Distribuția întârzierilor la sosire",
    labels={"arr_delay": "Timp întârziere (minute)"}
)
st.plotly_chart(fig)

# === ÎNTÂRZIERI MEDII PE ANI ===
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

st.markdown("**Legendă coloane întârzieri medii:**")
st.markdown("""
- `avg_arr_delay` → Total întârziere la sosire  
- `avg_carrier_delay` → Întârziere cauzată de compania aeriană  
- `avg_weather_delay` → Întârziere din cauza vremii  
- `avg_nas_delay` → Întârziere din cauza sistemului național de spațiu aerian (NAS)  
- `avg_security_delay` → Întârziere la controlul de securitate  
- `avg_late_aircraft_delay` → Întârziere din cauza sosirii târzii a altui avion
""")

avg_delays_pd_renamed = avg_delays_pd.rename(columns={
    "avg_arr_delay": "Total întârziere (min)",
    "avg_carrier_delay": "Întârziere companie",
    "avg_weather_delay": "Întârziere vreme",
    "avg_nas_delay": "Întârziere NAS",
    "avg_security_delay": "Întârziere securitate",
    "avg_late_aircraft_delay": "Întârziere avion anterior"
})
st.header("Întârzieri medii pe an:")
st.write(avg_delays_pd_renamed)

fig_avg = px.bar(
    avg_delays_pd,
    x="year",
    y=["avg_arr_delay", "avg_carrier_delay", "avg_weather_delay", "avg_nas_delay",
       "avg_security_delay", "avg_late_aircraft_delay"],
    title="Întârzieri medii pe an",
    labels={
        "avg_arr_delay": "Total întârziere la sosire",
        "avg_carrier_delay": "Întârziere companie aeriană",
        "avg_weather_delay": "Întârziere vreme",
        "avg_nas_delay": "Întârziere NAS (spațiu aerian)",
        "avg_security_delay": "Întârziere securitate",
        "avg_late_aircraft_delay": "Întârziere avion anterior",
        "year": "An"
    }
)

fig_avg.update_layout(
    shapes=[
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=2019.5,
            x1=2022.5,
            y0=0,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
            x=2021,
            y=1.05,  # puțin deasupra graficului
            xref="x",
            yref="paper",
            text="COVID",
            showarrow=False,
            font=dict(color="red", size=14, family="Arial Black"),
            bgcolor="LightSalmon",
            opacity=0.7,
        )
    ]
)

st.plotly_chart(fig_avg)

# === REGRESIE PE COMPANII (folosind df_spark) ===
st.subheader("Predicții întârzieri pentru 3 companii")

df_model = df_spark.select("carrier", "month", "arr_delay").dropna()

carrier_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_indexed").fit(df_model)
df_model = carrier_indexer.transform(df_model)

assembler = VectorAssembler(inputCols=["carrier_indexed", "month"], outputCol="features")
df_model = assembler.transform(df_model)

lr = LinearRegression(featuresCol="features", labelCol="arr_delay")
model = lr.fit(df_model)

sample_data = spark.createDataFrame([
    (2025, 6, "9E", "ABE"), (2025, 6, "AA", "JFK"), (2025, 6, "AS", "BOS"),
    (2025, 7, "9E", "ABE"), (2025, 7, "AA", "JFK"), (2025, 7, "AS", "BOS"),
    (2025, 8, "9E", "ABE"), (2025, 8, "AA", "JFK"), (2025, 8, "AS", "BOS"),
], ["year", "month", "carrier", "airport"])

sample_data = carrier_indexer.transform(sample_data)
sample_data = assembler.transform(sample_data)
predictions = model.transform(sample_data)

pandas_df = predictions.select("carrier", "month", "prediction").toPandas()
pandas_df["month_name"] = pandas_df["month"].map({6: "Iunie", 7: "Iulie", 8: "August"})

fig_line = px.line(pandas_df, x="month_name", y="prediction", color="carrier",
                   markers=True, title="Predicții întârzieri pe lună",
                   labels={"prediction": "Întârziere estimată (minute)", "month_name": "Lună"})
st.plotly_chart(fig_line)

fig_bar = px.bar(pandas_df, x="month_name", y="prediction", color="carrier",
                 barmode='group', title="Comparație întârzieri pe lună",
                 labels={"prediction": "Întârziere estimată (minute)", "month_name": "Lună"})
st.plotly_chart(fig_bar)

# === ZBORURI & AVIOANE ===
st.subheader("Zboruri și tipuri de avioane")

df_main = pd.read_csv("T_T100D_SEGMENT_US_CARRIER_ONLY.csv")
df_lookup = pd.read_csv("T_AIRCRAFT_TYPES.csv")
df_main['AIRCRAFT_TYPE'] = df_main['AIRCRAFT_TYPE'].astype(str)
df_lookup['AC_TYPEID'] = df_lookup['AC_TYPEID'].astype(str)

df_merged = pd.merge(df_main, df_lookup, how='left', left_on='AIRCRAFT_TYPE', right_on='AC_TYPEID')

agg = df_merged.groupby(['CARRIER', 'SSD_NAME']).size().reset_index(name='num_flights')
delays_grouped = df_spark.toPandas().groupby('carrier')['arr_delay'].sum().reset_index()
agg = agg.merge(delays_grouped, how='left', left_on='CARRIER', right_on='carrier')
agg['arr_delay'] = agg['arr_delay'].fillna(0)

fig_planes = px.bar(agg, x='SSD_NAME', y='arr_delay', color='CARRIER',
                   title="Întârziere totală pe tip de avion și companie",
                   labels={'arr_delay': 'Întârziere totală (minute)', 'SSD_NAME': 'Tip avion'})
st.plotly_chart(fig_planes)

# === TOP 10 AEROPORTURI ÎNTÂRZIERI ===
st.subheader("Top 10 aeroporturi cu cele mai mari întârzieri")

df_delays_pd = df_spark.toPandas()
delay_airport_year = df_delays_pd.groupby(['airport', 'year'])['arr_delay'].sum().reset_index(name='total_arr_delay')
top_10_airports = delay_airport_year.groupby('airport')['total_arr_delay'].sum().reset_index()\
    .sort_values(by='total_arr_delay', ascending=False).head(10)

filtered_df = delay_airport_year[delay_airport_year['airport'].isin(top_10_airports['airport'])]

fig_airports = px.bar(filtered_df, x='year', y='total_arr_delay', color='airport', barmode='group',
                      title='Top 10 aeroporturi cu cele mai mari întârzieri pe ani',
                      labels={'total_arr_delay': 'Întârziere totală (minute)', 'year': 'An'})

fig_airports.update_layout(
    shapes=[
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=2019.5,
            x1=2022.5,
            y0=0,
            y1=1,
            fillcolor="LightSalmon",
            opacity=0.3,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
        dict(
            x=2021,
            y=1.05,
            xref="x",
            yref="paper",
            text="COVID",
            showarrow=False,
            font=dict(color="red", size=14, family="Arial Black"),
            bgcolor="LightSalmon",
            opacity=0.7,
        )
    ]
)

st.plotly_chart(fig_airports)

# === PREDICȚII PE AEROPORTURI CU RANDOM FOREST (folosind df_spark) ===
st.subheader("Predicții întârzieri pe aeroporturi (2026)")

airport_indexer = StringIndexer(inputCol="airport", outputCol="airport_indexed").fit(df_spark)
df_indexed = airport_indexer.transform(df_spark)

df_model_airport = df_indexed.select("airport", "airport_indexed", "month", "arr_delay").dropna()
assembler_airport = VectorAssembler(inputCols=["airport_indexed", "month"], outputCol="features")
df_model_airport = assembler_airport.transform(df_model_airport)

rf = RandomForestRegressor(featuresCol="features", labelCol="arr_delay", numTrees=50,
                           maxDepth=7, maxBins=512, seed=42)
model_rf = rf.fit(df_model_airport)

airports = df_model_airport.select("airport").distinct().toPandas()['airport'].tolist()
months = list(range(1, 13))

predict_data = [(2026, m, a) for a in airports for m in months]
df_predict = spark.createDataFrame(predict_data, schema=["year", "month", "airport"])
df_predict = airport_indexer.transform(df_predict)
df_predict = assembler_airport.transform(df_predict)

predictions_airport = model_rf.transform(df_predict)
pred_pd = predictions_airport.select("airport", "month", "prediction").toPandas()
pred_pd['month_name'] = pred_pd['month'].map({1: "Ian", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mai", 6: "Iun",
                                              7: "Iul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"})
pred_pd['prediction'] = pred_pd['prediction'].apply(lambda x: max(x, 0))

annual_delay = pred_pd.groupby('airport')['prediction'].sum().reset_index().sort_values(by='prediction', ascending=False)

st.write("Estimare totală întârzieri (minute) pe aeroport în 2026:")
st.dataframe(annual_delay)

top10_2026 = annual_delay.head(10)
fig_top10 = px.bar(top10_2026, x='airport', y='prediction',
                   title="Top 10 aeroporturi cu cele mai mari întârzieri estimate în 2026",
                   labels={'prediction': 'Întârziere estimată (min)', 'airport': 'Aeroport'})
st.plotly_chart(fig_top10)

fig_months = px.line(pred_pd[pred_pd['airport'].isin(top10_2026['airport'])],
                     x='month_name', y='prediction', color='airport',
                     markers=True, title="Predicții lunare întârzieri 2026 (Top 10 aeroporturi)",
                     labels={"prediction": "Întârziere estimată (min)", "month_name": "Lună"})
st.plotly_chart(fig_months)

# === Închide Spark ===
spark.stop()
