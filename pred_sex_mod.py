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

df = df[df.AGEP > 17] #older than 17
df = df[df.INCOME > 15080] #minimum full time wage
df = df[df.COW != 8]
df = df[df.COW != 9] # remove unemployed and non paid workers

df = df[['AGEP', 'SEX', 'RAC1P', 'COW', 'SCHL', 'ST', 'INCOME', 'SOCP']]
df = df.dropna()

#df = df[df['INCOME'] <= df['INCOME'].quantile(0.95)] #remove outliers

df['SOCP'] = df['SOCP'].str[:2]
df['SOCP'] = pd.to_numeric(df['SOCP'])

#X = df[['AGEP', 'RAC1P', 'INCOME', 'COW', 'SCHL', 'ST', 'SOCP']]
#y = df['SEX']

one_hot_encoded = pd.get_dummies(df['SOCP'], prefix='SOCP')
df= pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['COW'], prefix = 'COW')
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['SCHL'], prefix = 'SCHL')
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['ST'], prefix = 'ST')
df = pd.concat([df, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['RAC1P'], prefix = 'RAC1P')
df = pd.concat([df, one_hot_encoded], axis=1)

df = df.drop(columns=['RAC1P', 'INCOME', 'COW', 'SCHL', 'ST', 'SOCP'])

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

n=100000 #size of test sample

df = df.iloc[:-n]
test_df = df.iloc[-n:]

test_df_male = test_df[test_df['SEX'] == 1]
test_df_male = test_df_male.sample(n=46236, random_state=1)

test_df_female = test_df[test_df['SEX'] == 2]

test_df = pd.concat([test_df_female, test_df_male], ignore_index=True)

X_train = df.drop('SEX', axis=1)
X_test = test_df.drop('SEX', axis=1)
y_train = df[['SEX']]
y_test = test_df['SEX']

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