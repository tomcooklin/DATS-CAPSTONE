from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("/Users/tomcooklin/Desktop/sample.csv")

df = df.sample(n=100000, random_state=1)

df = df.rename(columns={"REGION.x": "REGION", "ST.x": "ST", "ADJINC.x": "ADJINC", "PUMA.x": "PUMA", "DIVISION.x": "DIVISION", "REGION.x.1": "REGION"})
df = df[df.AGEP > 17] #older than 17
df = df[df.INCOME > 15080] #minimum full time wage
df = df[df.COW != 8]
df = df[df.COW != 9] # remove unemployed and non paid workers

df = df[df['INCOME'] <= df['INCOME'].quantile(0.95)] #remove outliers

# Select the relevant features (age, gender, race, education, class of worker) and the target variable (income)
df = df[['AGEP', 'SEX', 'RAC1P', 'COW', 'SCHL', 'ST', 'INCOME', 'SOCP']]

df = df.dropna()

df['COW'] = df['COW'].astype(int)

df['SOCP'] = df['SOCP'].str[:2]
df['SOCP'] = pd.to_numeric(df['SOCP'])

le = LabelEncoder()
df['RAC1P'] = le.fit_transform(df['RAC1P'])

# Create a heatmap of the correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = df[['AGEP', 'SEX', 'RAC1P', 'COW', 'SCHL', 'ST', 'SOCP']]

one_hot_encoded = pd.get_dummies(df['SOCP'], prefix='SOCP')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['RAC1P'], prefix = 'RAC1P')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['COW'], prefix = 'COW')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['SCHL'], prefix = 'SCHL')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['ST'], prefix = 'ST')
X = pd.concat([X, one_hot_encoded], axis=1)

one_hot_encoded = pd.get_dummies(df['SEX'], prefix = 'SEX')
X = pd.concat([X, one_hot_encoded], axis=1)

y = df['INCOME']

# Normalize the input features
scaler = StandardScaler()

agep_col = X[['AGEP']]  # Extract column as a 2D array
agep_col_scaled = scaler.fit_transform(agep_col)  # Fit and transform the column
X['AGEP'] = agep_col_scaled 

X = X.drop(columns=['SEX', 'RAC1P', 'COW', 'SCHL', 'ST', 'SOCP'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#random forest
model = RandomForestRegressor(n_estimators=100, random_state=1)
model.fit(X_train, y_train)
y_pred_rf = model.predict(X_test)

scores_mean = []
scores_median = []
for i in range(1,20):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_pred = np.mean([y_pred_knn, y_pred_rf], axis=0)
    scores_mean.append(mean_absolute_error(y_pred.reshape(-1, 1), np.array(y_test).reshape(-1, 1)))
    scores_median.append(median_absolute_error(y_pred.reshape(-1, 1), np.array(y_test).reshape(-1, 1)))

sns.lineplot(range(1,20), scores_mean, label = 'Mean Abs Error')
sns.lineplot(range(1,20), scores_median, label = 'Median Abs Error')
plt.xlabel('Value of K for Model')
plt.ylabel('Error')
plt.title("Error for each number of K")
plt.ticklabel_format(style='plain')
plt.xticks(range(0, 21, 2), range(0, 21, 2))
plt.legend()
plt.show()  

######## USER RESPONSE #########

user1 = X.head(1)
user1.iloc[0] = 0

ans = "Y"

while ans == "Y":
    age = input("What is your age?\n Answer: ")
    sex = input("What is your sex?\n" + '\n'.join([
    "1: Male",
    "2: Female",
    "Answer: "
]))

    race = input("What is your race?\n" + '\n'.join([
    "0: Asian",
    "1: Black",
    "2: Other",
    "3: White",
    "Answer: "
]))
    schl = input("What is your highest level of education?\n" + '\n'.join([
    "1: No schooling completed",
    "2: Nursery school, preschool",
    "3: Kindergarten",
    "4: Grade 1",
    "5: Grade 2",
    "6: Grade 3",
    "7: Grade 4",
    "8: Grade 5",
    "9: Grade 6",
    "10: Grade 7",
    "11: Grade 8",
    "12: Grade 9",
    "13: Grade 10",
    "14: Grade 11",
    "15: 12th grade - no diploma",
    "16: Regular high school diploma",
    "17: GED or alternative credential",
    "18: Some college, but less than 1 year",
    "19: 1 or more years of college credit, no degree",
    "20: Associate's degree",
    "21: Bachelor's degree",
    "22: Master's degree",
    "23: Professional degree beyond a bachelor's degree",
    "24: Doctorate degree",
    "Answer: "
]))
    state = input("What state are you from?\n" + '\n'.join([
    "01: Alabama",
    "02: Alaska",
    "04: Arizona",
    "05: Arkansas",
    "06: California",
    "08: Colorado",
    "09: Connecticut",
    "10: Delaware",
    "11: District of Columbia",
    "12: Florida",
    "13: Georgia",
    "15: Hawaii",
    "16: Idaho",
    "17: Illinois",
    "18: Indiana",
    "19: Iowa",
    "20: Kansas",
    "21: Kentucky",
    "22: Louisiana",
    "23: Maine",
    "24: Maryland",
    "25: Massachusetts",
    "26: Michigan",
    "27: Minnesota",
    "28: Mississippi",
    "29: Missouri",
    "30: Montana",
    "31: Nebraska",
    "32: Nevada",
    "33: New Hampshire",
    "34: New Jersey",
    "35: New Mexico",
    "36: New York",
    "37: North Carolina",
    "38: North Dakota",
    "39: Ohio",
    "40: Oklahoma",
    "41: Oregon",
    "42: Pennsylvania",
    "44: Rhode Island",
    "45: South Carolina",
    "46: South Dakota",
    "47: Tennessee",
    "48: Texas",
    "49: Utah",
    "50: Vermont",
    "51: Virginia",
    "53: Washington",
    "54: West Virginia",
    "55: Wisconsin",
    "56: Wyoming",
    "72: Puerto Rico",
    "Answer: "
]))
    cow = input("What is your employment type?\n" + '\n'.join([
    "1: Employee of a private for-profit company or business, or of an individual, for wages, salary, or commissions",
    "2: Employee of a private not-for-profit, tax-exempt, or charitable organization",
    "3: Local government employee (city, county, etc.)",
    "4: State government employee",
    "5: Federal government employee",
    "6: Self-employed in own not incorporated business, professional practice, or farm",
    "7: Self-employed in own incorporated business, professional practice or farm",
    "Answer: "
]))
    socp = input("What is your considered standard occupational classification?\n" + '\n'.join([
    "11: Chief Executives, Legislators and other Managers",
    "13: Business or Finance Employee",
    "15: Computer Science Department",
    "17: Engineer",
    "19: Scientist",
    "21: Psychologist, Social Worker, Counselor",
    "23: Lawyer, Paralegal",
    "25: Education",
    "27: Entertainment",
    "29: Medicine",
    "31: Health Services",
    "33: Police, Firefighter, Detective, Enforcement",
    "35: Chef, Dishwasher, Waiter",
    "37: Cleaners, Landscaper",
    "39: Barber, Beauty, Animal Worker",
    "41: Sales",
    "43: Office, Clerk",
    "45: Agriculture",
    "47: Construction",
    "49: Repairs",
    "51: Production",
    "53: Transportation",
    "55: Military",
    "Answer: "
])) 

    
    #age
    user = user1
    age = scaler.transform(np.array(age).reshape(1, -1))
    user.loc[:, 'AGEP'] = float(age[0])
    
    #sex
    sex_hot = str('SEX_' + str(sex))
    user.loc[:, sex_hot] = 1
    
    #race
    race_hot = str('RAC1P_' + str(race))
    user.loc[:, race_hot] = 1
    
    #education
    schl_hot = str('SCHL_' + str(schl))
    user.loc[:, schl_hot] = 1
    
    #state 
    st_hot = str('ST_' + str(state))
    user.loc[:, st_hot] = 1
    
    #cow
    cow_hot = str('COW_' + str(cow))
    user.loc[:, cow_hot] = 1
    
    #socp
    socp_hot = str('SOCP_' + str(socp))
    user.loc[:, socp_hot] = 1
    
    # Make a prediction using the model
    pred_knn = knn.predict(user)
    pred_rf = model.predict(user)
    
    prediction = np.mean(np.concatenate((pred_knn, pred_rf)))
    
    print(prediction)
    
    ans = input("Would like to re-input your data? Y/N \nAnswer:")



