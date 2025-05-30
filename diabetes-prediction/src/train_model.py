import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statistics import mean, mode

# Importing data
data = pd.read_csv(r"C:\Users\User\Downloads\diabetes.csv")

# Exploring data
print(data.head())
print(data.columns)
print(data.isna().sum())
print(data.nunique())

# Filling missing values
data.loc[data["Glucose"] == 0, "Glucose"] = mode(data["Glucose"])
data.loc[data["BloodPressure"] == 0, "BloodPressure"] = mode(data["BloodPressure"])
data.loc[data["BMI"] == 0, "BMI"] = mode(data["BMI"])

# Some figures
fig, axes = plt.subplots(3, 2)

sns.histplot(data["Glucose"], ax=axes[0, 0])
axes[0, 0].set_title("Glucose level distribution")
sns.histplot(data["BloodPressure"], ax=axes[0, 1])
axes[0, 1].set_title("Blood pressure distribution")
sns.histplot(data["BMI"], ax=axes[1, 0])
axes[1, 0].set_title("BMI distribution")
sns.histplot(data["Outcome"], ax=axes[1, 1])
axes[1, 1].set_title("Outcome")
sns.histplot(data["DiabetesPedigreeFunction"], ax=axes[2, 0])
axes[2, 0].set_title("DPF distribution")
sns.histplot(data["Insulin"], ax=axes[2, 1])
axes[2, 1].set_title("Insulin distribution")
plt.tight_layout()
plt.show()

# Feature engineering
data["Age>45"] = (data["Age"] > 45).astype(int)
data["HasHighGlucose"] = (data["Glucose"] > 140).astype(int)
data["HighBloodPressure"] = (data["BloodPressure"] > 130).astype(int)
data["Obesity"] = (data["BMI"] > 30).astype(int)
data["InsulinResistance"] = (data["Insulin"] > 60).astype(int)
data["HighDPF"] = (data["DiabetesPedigreeFunction"] > 0.8).astype(int)
data["HighScores"] = (data["HasHighGlucose"] + data["Age>45"] + data["HighDPF"] +
    data["HighBloodPressure"] + data["Obesity"] + data["InsulinResistance"]
)

# Dividing data for labels and features
y = data["Outcome"]
X = data.drop("Outcome", axis=1)

# All columns are numerical columns, so there is no need for encoding categorical ones
print(X.dtypes)

# RandomForestRegressor
my_model_1 = RandomForestRegressor(n_estimators=100)
scores_1 = -cross_val_score(my_model_1, X, y, cv=5, scoring="neg_mean_absolute_error")
print("RandomForestRegressor: ", mean(scores_1))

# XGBRegressor
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.01)
scores_2 = -cross_val_score(my_model_2, X, y, cv=5, scoring="neg_mean_absolute_error")
print("XGBRegressor: ", mean(scores_2))

# LogisticRegression
# Scaling the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Model
my_model_3 = LogisticRegression(solver="liblinear")
scores_3 = -cross_val_score(my_model_3, X, y, cv=5, scoring="neg_mean_absolute_error")
print("LogisticRegression: ", mean(scores_3))
