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
data = pd.read_csv('D:\Learn\SelfLearn\ML\Practise\data-preprocessing-HouariZegai\Social_Network_Ads.csv')
X = data.iloc[:, :4]
y = data.iloc[:,4]

# Step 2: Handle Missing data if they exist.
from sklearn.preprocessing import Imputer
im = Imputer()
X[:, 1:4] = im.fit_transform(X[:, 1:4])

# Step 3: Encode the categorical variables.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl_x = LabelEncoder()
y = lbl_x.fit_transform(y)


# Step 4: Do Feature Scaling if necessary.

# Final Step: Train/Test Splitting.
