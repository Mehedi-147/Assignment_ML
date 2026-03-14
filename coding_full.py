import pandas as pd
import numpy as np 

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

## Data
df = pd.read_csv("/Users/mehedihasan/Downloads/airline passenger satisfaction/airline_passenger_satisfaction.csv")
print(df.head())

# Data exploration
print(df.isnull().sum())
missing_values = df[df['Arrival Delay'].isnull()]
print(missing_values)
subset = df[df["Arrival Delay"].isnull()][["Departure Delay", "Flight Distance"]]
print(subset)

#data prep
df_known = df[df['Arrival Delay'].notnull()]   # Known (can be used to train)
df_missing = df[df['Arrival Delay'].isnull()]  # Missing (need to be predicted)
X_train =df_known[['Departure Delay', 'Flight Distance']]
Y_train = df_known["Arrival Delay"]
x_predict = df_missing[["Departure Delay", 'Flight Distance']]
model = LinearRegression()
model.fit(X_train, Y_train)
predicted_values = model.predict(x_predict)
df.loc[df_missing.index, "Arrival Delay"] = predicted_values
missing_values = df[df['Arrival Delay'].isnull()]
print(missing_values)

### Evaluation of model(Scatter plot)
points = plt.scatter(Y_train, model.predict(X_train), color='red', alpha=0.5)
line_x = Y_train.min(), Y_train.max()
line_y = Y_train.min(), Y_train.max()
diagonal = plt.plot(line_x, line_y)
plt.xlabel("actual arrival delays")
plt.ylabel("predicted arrival delays")
plt.title("Model predictions VS actual")
plt.close()

## Comparisons of evaluations (errors)
y_predicted = model.predict(X_train)
y_true = Y_train
mae = mean_absolute_error(y_predicted, y_true)
mse = mean_squared_error(y_predicted, y_true)
r2 = r2_score(y_true, y_predicted)
print(mae,mse,r2)
df.to_csv("airline_cleaned.csv", index=False)

## Principle component analysis and scaling 
service_columns = [
    "Departure and Arrival Time Convenience",
    "Ease of Online Booking",
    "Check-in Service",
    "Online Boarding",
    "Gate Location",
    "On-board Service",
    "Seat Comfort",
    "Leg Room Service",
    "Cleanliness",
    "Food and Drink",
    "In-flight Service",
    "In-flight Wifi Service",
    "In-flight Entertainment",
    "Baggage Handling"
]
df_service = df[service_columns]

## Scaling/Normalization

scaler = StandardScaler()
scaler.fit(df_service)
scaled = scaler.transform(df_service)

pca = PCA()
pca.fit(scaled)
pc_values= pca.transform(scaled)

print(pca.explained_variance_ratio_)
print(pca.components_)

# PCA plot for interpretation
PC_1 = pc_values[:, 0]
PC_2 = pc_values[:, 1]
temp_dict = { "Satisfied": "green"
                ,"Neutral or Dissatisfied": "red"
             }
df["satisfaction_color"] = df["Satisfaction"].map(temp_dict)

scatt_points = plt.scatter(PC_1, PC_2, color=df["satisfaction_color"], alpha=0.4)
plt.xlabel("Services onboard")
plt.ylabel("Pre-boarding Facilities")
plt.title("Hidden Service Factors Influencing Passenger Satisfaction (PCA Analysis)")
plt.close()

# Explanatory data analysis with different models

le = LabelEncoder()
df['target'] = le.fit_transform(df["Satisfaction"])
print(le.classes_)

# Splitting(train-test) for Delay-only Model
x = df[["Arrival Delay", "Departure Delay"]] # Features (2 columns)
y = df['target'] ## Target column
x_train, x_test, y_train, y_test = train_test_split( 
    x,
    y,
    test_size= 0.2,
    random_state=42,
    stratify= y
)
## testting the split
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))
## Scaling on features
scale = StandardScaler()
train_scale =scale.fit_transform(x_train)
test_scale = scale.transform(x_test)
log_reg_delay = LogisticRegression()
log_reg_delay.fit(train_scale,y_train)
prediction = log_reg_delay.predict(test_scale)
coefficents = log_reg_delay.coef_.flatten()
## Coefficinet bar chart(Satisfaction evaluation)
x_values = ["Arrival Delay", "Departure Delay"]
y_values = coefficents
bar = plt.bar(x_values, y_values, 0.8)
base_line =plt.axhline(y=0, color="Black", linestyle='--', linewidth=1.5)
plt.xlabel("Delay type")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Coefficient Plot")
plt.close()
## Evaluation of Logistic Regeression model
acc_score = accuracy_score(y_test, prediction)
cm = confusion_matrix(y_test, prediction)
print("accuracy_score:", acc_score)
print(cm)
## Final model logistic regression
df["PC_1"] = PC_1 # Adding numpy arrays to df
df["PC_2"] = PC_2
df["Customers"] = le.fit_transform(df["Customer Type"])
df = pd.get_dummies(df, columns=["Class"], drop_first=True)
print(df.columns)
X_reg = df[["Arrival Delay", 'Departure Delay', "PC_1", "PC_2", "Customers", 'Class_Economy', 'Class_Economy Plus' ]]
Y_reg = df["target"]
print(df[["Arrival Delay","Departure Delay","PC_1","PC_2"]].describe())

X_reg_train, X_reg_test, Y_reg_train, Y_reg_test = train_test_split(
    X_reg,
    Y_reg,
    test_size=0.2,
    random_state=42,
    stratify= Y_reg
)
## Modelling logistic regression
log_reg_full = LogisticRegression()
log_reg_full.fit(X_reg_train, Y_reg_train)
full_prediction =log_reg_full.predict(X_reg_test)
columns = log_reg_full.feature_names_in_
full_coeff = abs(log_reg_full.coef_.flatten())
## Evaluation Final logistic regression model
final_acc_score = accuracy_score(Y_reg_test,full_prediction)
final_cm = confusion_matrix(Y_reg_test,full_prediction)
print("Model performance:",final_acc_score)
print("Confusion matrics:", final_cm)
## Coefficients Bar Chart (Plot evaluation)
Final_X_values = ["Arrival Delay", 'Departure Delay', "PC_1", "PC_2", "Customers", 'Class_Economy', 'Class_Economy Plus']
Final_y_values = full_coeff
bars = plt.bar(Final_X_values, Final_y_values, 0.8)
base_line_final =plt.axhline(y=0, color="Black", linestyle='--', linewidth=1.5)
plt.xlabel("Features Type")
plt.ylabel("Feature's Coefficients")
plt.title("Final Logistic regression Coefficients Plot")
plt.close()

## Feature importance Ranking chart(Sorted coefficients)
pairs = list(zip(columns, full_coeff))
sorting_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

for f, coeff in sorting_pairs:
    print((f'{f} {round(coeff,3)}'))

## AUC-ROC curve for Classification model
y_pred_prob = log_reg_full.predict_proba(X_reg_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_reg_test, y_pred_prob)
roc_auc= auc(fpr, tpr)
## Plotting Roc curve
plt.figure()
plt.plot(fpr, tpr, lw =2, label=f'ROC curve (area)=, {roc_auc:.2f}')
plt.plot([0,1],[0,1], color="Navy", linestyle='--', label="random guess")
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
    














































    






















