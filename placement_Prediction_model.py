import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions

df = pd.read_csv(r'd:\PROJECTS\ml-project-1\placement.csv')

# Drop the first column (unnecessary index column) --> clean the data
df = df.iloc[:,1:]
#print(df.head())


 #plotting the data to visualize relationships
 #cgpa on x-axis and iq on y-axis
 #Exploratory Data Analysis (EDA)
graph = plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Scatter plot of CGPA vs IQ')
plt.show()

# separating independent and dependent variables
x = df[['cgpa', 'iq']].values
y = df['placement'].values

# splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# training the logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# evaluating the model
m = model.predict(x_test)
print("Predicted values:", m)  
z = y_test
print("Actual values:", z)
accuracy = model.score(x_test, y_test)
print(f'Model Accuracy: {accuracy*100:.2f}%')

# Plotting decision regions
plot_decision_regions(x_train, y_train, clf=model, legend=2)
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Decision Regions for Logistic Regression')
plt.show()