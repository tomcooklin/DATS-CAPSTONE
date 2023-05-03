import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency

# Load the data and preprocess it
df = pd.read_csv("/Users/tomcooklin/Desktop/subset.csv") 
df = df.rename(columns={"REGION.x": "REGION", "ST.x": "ST", "ADJINC.x": "ADJINC", "PUMA.x": "PUMA", "DIVISION.x": "DIVISION", "REGION.x.1": "REGION"})

df_white = df[df['RAC1P'] == 'White']
df_white = df_white.sample(n=137689, random_state=1)

df_black = df[df['RAC1P'] == 'Black']
df_black = df_black.sample(n=137689, random_state=1)

df_other = df[df['RAC1P'] == 'Other']
df_other = df_other.sample(n=137689, random_state=1)

df_asian = df[df['RAC1P'] == 'Asian']

df = pd.concat([df_other, df_asian, df_black, df_white], ignore_index=True)

df = df[df.AGEP > 17] #older than 17
df = df[df.INCOME > 15080] #minimum full time wage
df = df[df.COW != 8]
df = df[df.COW != 9] # remove unemployed and non paid workers

df = df[df['INCOME'] <= df['INCOME'].quantile(0.95)] #remove outliers

df['SOCP'] = df['SOCP'].str[:2]
df['SOCP'] = pd.to_numeric(df['SOCP'])

df = df[['AGEP', 'SEX', 'RAC1P', 'COW', 'SCHL', 'ST', 'INCOME', 'SOCP']]
df = df.dropna()

X = df[['AGEP', 'SEX', 'INCOME', 'COW', 'SCHL', 'ST', 'SOCP']]
y = df['RAC1P']

one_hot_encoded = pd.get_dummies(df['SOCP'], prefix='SOCP')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['COW'], prefix = 'COW')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['SCHL'], prefix = 'SCHL')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['ST'], prefix = 'ST')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['SEX'], prefix = 'SEX')
X = pd.concat([X, one_hot_encoded], axis=1)

X = X.drop(columns=['SEX', 'INCOME', 'COW', 'SCHL', 'ST', 'SOCP'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(n_estimators=50, random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print confusion matrix
plot_confusion_matrix(conf_mat=cm, figsize=(4, 4), cmap='Blues')

accuracy = accuracy_score(y_test, y_pred)

# print accuracy
print("Accuracy: " + str(accuracy))

print("-------------------")

print("Chi-Squared Test Results")

chi2, p, dof, expected = chi2_contingency(cm)

# Print the test results
print(f"Chi-squared statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("Expected values:")
print(expected)
