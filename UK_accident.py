import pandas as pd
import numpy as np

data=pd.read_csv("accidents_2012_to_2014.csv")

data.head()
data.tail()
data.shape
data.info()
data.describe().T

data.isnull().sum()
data["Accident_Index"].unique()
data["Accident_Index"].value_counts()
data_2= data.loc[:,["Accident_Severity","Number_of_Casualties","Day_of_Week","Date","Time","Local_Authority_(District)"
                         ,"Local_Authority_(Highway)","Road_Type","Speed_limit","Light_Conditions","Weather_Conditions",
                        "Road_Surface_Conditions","Year"]]

data_2.head()

data_2.isnull().sum()

data_2=data_2.dropna()

data_2.isnull().sum()

data_2["Year"].unique()
data_2.Year = data_2.Year.astype(str)

accidentNumber = data_2.groupby("Year")

sum_accidentNumber=pd.DataFrame(accidentNumber.size().sort_values(ascending=False),columns=["Sayı"])
sum_accidentNumber.head()

import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(y = sum_accidentNumber["Sayı"] , x = sum_accidentNumber.index)
plt.title("Yıllara göre kaza sayısı")

data.Speed_limit = data.Speed_limit.astype(str)
speed_sayisi = data.groupby("Speed_limit")

toplam_speed=pd.DataFrame(speed_sayisi.size().sort_values(ascending=False),columns=["Speed_limit_toplam"])

sns.barplot(y = toplam_speed["Speed_limit_toplam"] , x = toplam_speed.index)
plt.title("Hıza göre kaza sayısı")
