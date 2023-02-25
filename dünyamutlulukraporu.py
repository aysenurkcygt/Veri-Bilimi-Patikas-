import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('2019.csv')
data.head()

data.info()

data.isnull().sum()

data.columns = data.columns.str.replace(' ', '_')
data.head()

data["Country or region"].unique()

#EN MUTLU 5 ÜLKE

sns.barplot(y = data["Country or region"][:5], x = data["Score"][:10])

#EN MUTSUZ 5 ÜLKE

sns.barplot(y = data["Country or region"][-5:], x = data["Score"][-10:])

