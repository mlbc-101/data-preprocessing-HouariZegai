""" Do not delete this section, Please Commit your changes after implementing the necessary code.

- The data file called Social_Network_Ads.csv.
- Your Job is to preprocess this data because we gonna use it later one in the course.

The Features of this dataset are:
	- UserID: Which represent id of user in the database.
	- Gender: Can be male or female.
	- EstimatedSalary: The salary of the user.
	- Purchased: An integer number {1 if the user purshased something, 0 otherwise}
	
	The target variable for this data is the purshased status.

Happy coding."""

# Step 0: import the necessary libraries: pandas, matplotlib.pyplot, and numpy.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: load your dataset using pandas
data = pd.read_csv('Social_Network_Ads.csv')
data.head()

X = data.drop(['User ID', 'Purchased'], axis = 1)
y = data['Purchased']

# Step 2: Handle Missing data if they exist.
# summarize missing data (check if there is a missing data)
X.isnull().sum()

# Step 3: Encode the categorical variables.
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X['Gender'] = encoder.fit_transform(X['Gender'])

X.head()

# Step 4: Do Feature Scaling if necessary.
from sklearn.preprocessing import StandardScaler
# standardise the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Final Step: Train/Test Splitting.
from sklearn.model_selection import train_test_split
train_x, train_y, test_x, test_y = train_test_split(X, y, test_size = 0.3, random_state = 19)
