import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("socioeconomic_crime.csv", sep=",")

gdp_capita = df.groupby("State Abbr")["GDP per capita"].mean()

plt.figure(figsize=(20, 6))
plt.bar(gdp_capita.index, gdp_capita.values)
plt.xlabel("State")
plt.ylabel("GDP Per Capita")
plt.title("States GDP Per Capita (2005-2015)")
plt.show()
print(gdp_capita)

df["Violent Crime Rate"] = (df["Violent Crime"] / df["Population"]) * 100000

crime_rate = df.groupby("State Abbr")["Violent Crime Rate"].mean()

plt.figure(figsize=(20, 6))
plt.bar(crime_rate.index, crime_rate.values)
plt.xlabel("State")
plt.ylabel("Violent Crime")
plt.title("States Violent Crime per 100,000 People (2005-2015)")
plt.show()

plt.scatter(crime_rate.values, gdp_capita)
plt.xlabel("Violent Crime Rate per 100,000 people")
plt.ylabel("GDP per Capita")
plt.title("Violent Crime Rate vs GDP per Capita")
plt.show()

region_gdp = df.groupby("Location")["GDP per capita"].mean()
plt.bar(region_gdp.index, region_gdp.values)
plt.xlabel("Regions")
plt.ylabel("GDP per Capita")
plt.title("GDP per Capita of Major US Regions")
plt.show()

region_crime = df.groupby("Location")["Violent Crime Rate"].mean()
plt.bar(region_crime.index, region_crime.values)
plt.xlabel("Regions")
plt.ylabel("Violent Crime Rate per 100,000 people")
plt.title("Violent Crime Rates of US Regions")
plt.show()

region_education = df.groupby("Location")["People with less than 9 years of education/people with college degree or above"].mean()
plt.bar(region_education.index, region_education.values)
plt.xlabel("Regions")
plt.ylabel("People with less than 9 years of education/people with college degree or above", fontsize = 8)
plt.title("Educational Attainment")
plt.show()
