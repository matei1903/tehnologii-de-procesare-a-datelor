from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt



spark = SparkSession.builder \
    .appName("Flight Delays Analysis") \
    .getOrCreate()


df = spark.read.csv("Airline_Delay_Cause.csv", header=True, inferSchema=True)


#df.show(10)


df.printSchema()

# Număr total de zboruri
print("Total rows:", df.count())

# Filtrare: elimină anii 2020, 2021 și 2022
filtered_df = df.filter(~df.year.isin(2020, 2021, 2022))

#Tendinta lunara a intarzierilor

# Calcularea întârzierilor totale (arr_del15) pentru fiecare lună
monthly_delays = filtered_df.groupBy("month").agg(
    F.sum("arr_del15").alias("total_delays")  
)

# Ordonarea rezultatele după lună
monthly_delays.orderBy("month").show()

# Verificarea numărului de zboruri pe an
flight_counts = filtered_df.groupBy("year").count()
flight_counts.show()



# Distribuția întârzierilor pentru fiecare an


delays = filtered_df.select("year", "arr_delay", "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay").toPandas()

# Crearea unei histograme pentru întârzierile totale
plt.figure(figsize=(10,6))
plt.hist(delays["arr_delay"].dropna(), bins=50, alpha=0.7, label="Total Delay")
plt.hist(delays["carrier_delay"].dropna(), bins=50, alpha=0.7, label="Carrier Delay")
plt.hist(delays["weather_delay"].dropna(), bins=50, alpha=0.7, label="Weather Delay")
plt.hist(delays["nas_delay"].dropna(), bins=50, alpha=0.7, label="NAS Delay")
plt.hist(delays["security_delay"].dropna(), bins=50, alpha=0.7, label="Security Delay")
plt.hist(delays["late_aircraft_delay"].dropna(), bins=50, alpha=0.7, label="Late Aircraft Delay")

plt.title('Distribution of Delays by Type')
plt.xlabel('Minutes of Delay')
plt.ylabel('Frequency')
plt.legend()
plt.show()


delays = filtered_df.select("year", "weather_delay").toPandas()

# Crearea unei histograme pentru întârzierile din cauza vremii
plt.figure(figsize=(10,6))
plt.hist(delays["weather_delay"].dropna(), bins=50, alpha=0.7, label="Weather Delay", color='skyblue')

plt.title('Distribution of Weather Delays')
plt.xlabel('Minutes of Weather Delay')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#filtered_df.show()

spark.stop()
