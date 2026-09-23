### Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### Import Data
df = pd.read_csv("juco400m.csv", sep=",")

### Extract the numeric column
times = df["TIME"]

### Create the histogram
plt.hist(df["TIME"], bins=10, color="red", edgecolor="black")
plt.title("State 400m Times")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

### Average Time Function
def average_time(times):
    return np.mean(times)
avg = average_time(df["TIME"])
print("\nAverage 400m Time:", f"{avg:.3f}")

### Average State Winning Time Function
def average_winning_time(df):
    winners = df[df["PLACE"] == 1]
    return np.mean(winners["TIME"])
avg_time = average_winning_time(df)
print("\nAverage Winning Time:", f"{avg_time:.3f}")

### Average Time to Place Top 3
def podium_time(df):
    podium = df[df["PLACE"] < 4]
    return np.mean(podium["TIME"])
avg_podium_time = podium_time(df)
print("\nAverage Time to Place Top 3 is:", f"{avg_podium_time:.3f}")

### Time Needed to Score at State
def scoring_time(df):
    scorer = df[df["SCORE"] > 0]
    return np.mean(scorer["TIME"])
avg_time_to_score = scoring_time(df)
print("\nAverage Time to Score Points is:", f"{avg_time_to_score:.3f}")

winners = df[df["PLACE"] == 1]
plt.hist(winners["TIME"], bins=3, color="blue", edgecolor="black")

plt.title("State Champ Times")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

plt.scatter(df.SCORE, df.TIME, color="black")
plt.title("State 400m Points and Times")
plt.xlabel("Points")
plt.ylabel("Times")
plt.show()