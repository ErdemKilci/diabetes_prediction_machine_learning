#importing the dependencies
import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#data colection and analysis
#PIMA diabetes dataset

#loading the diabetes dataset to pandas dataframe
diabetes_dataset = pd.read_csv('diabetes.csv')

#Printing the first 5 rows of the dataframe
print(diabetes_dataset.head())

#number of rows and columns in this dataset
print(diabetes_dataset.shape)

#getting the statistical measures of the data
print(diabetes_dataset.describe())

#label 1 for diabetic and 0 for non-diabetic
print(diabetes_dataset['Outcome'].value_counts())

#groups the outcome by the mean of the other column's values
#it shows the middle value of the diabetic and non-diabetic people's statictics
diabetes_dataset.groupby('Outcome').mean()

#separating the data and the labels
x= diabetes_dataset.drop(columns='Outcome', axis=1)
y= diabetes_dataset['Outcome']

#data standardization
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
print(standardized_data)

x = standardized_data
y=diabetes_dataset['Outcome']
print(x)
print(y)

#train test split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, train_x.shape, test_x.shape)

classifier = svm.SVC(kernel='linear')

#training the support vector machine classifier
classifier.fit(train_x, train_y)

#model evaluation
#accuracy score on the training data
train_x_prediction = classifier.predict(train_x)
training_data_accuracy = accuracy_score(train_x_prediction, train_y)

print('Accuracy score of the training data : ', training_data_accuracy)

#accuracy score on the test data
test_x_prediction = classifier.predict(test_x)
test_data_accuracy = accuracy_score(test_x_prediction, test_y)
print('Accuracy score of the test data : ', test_data_accuracy)

#making a predictive system
input_data = (2,197,70,45,543,30.5,0.158,53)

#changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# creating a DataFrame with feature names
input_data_df = pd.DataFrame(input_data_reshaped, columns=diabetes_dataset.columns[:-1])

# standardizing the input data using the same scaler
std_data = scaler.transform(input_data_df)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
    print('The person is diabetic')