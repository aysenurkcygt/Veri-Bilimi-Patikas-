import pandas as pd
import numpy as np

data=pd.read_csv("covid_19_clean_complete.csv")
copydata=data.copy()

data.head(10)
data.info()

data.describe().T
data.isnull().sum()

data["Province/State"].unique()
data["Province/State"].value_counts()

data_2=data.fillna("0")
data_2.head()

data["WHO Region"].drop_duplicates(keep="first").dropna()

#VISUALIZATION

#1
rate = data_2.TotalCases / data_2.TotalTests
c = np.array(["Mexico", "Bolivia", "South Korea"])

plt.bar(c, rate, color = "Brown")
plt.title('Country-Based the Rate of Cases per Test')
plt.ylabel('Rate', fontsize=8)
plt.xlabel('Countries', fontsize=8)
plt.show()

#2

x = data.loc[cov19["WHO Region"] == "Americas"].TotalDeaths.sum()
y = data.loc[cov19["WHO Region"] == "Europe"].TotalDeaths.sum()
z = data.loc[cov19["WHO Region"] == "Africa"].TotalDeaths.sum()
t = data.loc[cov19["WHO Region"] == "EasternMediterranean"].TotalDeaths.sum()
y = data.loc[cov19["WHO Region"] == "South-EastAsia"].TotalDeaths.sum()
w = data.loc[cov19["WHO Region"] == "WesternPacific"].TotalDeaths.sum()

sum_death = np.array([x, y, z, t, y, w]) 
regions = ["America", "Europe", "Africa", "Eastern Mediterranean", "South-East Asia", "Western Pacific"]
color = ["#82d308", "#0707bc", "#d82708", "#bc0798", "#eae609", "#7f007f"]

plt.pie(sum_death, labels = regions, colors = color)
plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.2, 1, 1))
plt.show()
