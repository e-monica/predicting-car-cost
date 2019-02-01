#Disclaimer: the information and steps are laid out in the coursework provided by Dataquest.io.
#Corresponding solutions accompany them. 

# In this course, we explored the fundamentals of machine learning using the k-nearest neighbors algorithm. 
# In this guided project, you'll practice the machine learning workflow you've learned so far to predict 
# a car's market price using its attributes. The data set we will be working with contains information 
# on various cars. For each car we have information about the technical aspects of the vehicle such as 
# the motor's displacement, the weight of the car, the miles per gallon, how fast the car accelerates, 
# and more. You can read more about the data set here and can download it directly from here.

# Read imports-85.data into a dataframe named cars. If you read in the file using pandas.read_csv() 
# without specifying any additional parameter values, you'll notice that the column names don't match 
# the ones in the dataset's documentation. Why do you think this is and how can you fix this?
# Determine which columns are numeric and can be used as features and which column is the target column.
# Display the first few rows of the dataframe and make sure it looks like the data set preview.

import pandas as pd
import numpy as np

pd.options.display.max_columns = 99


cols = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
        'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
        'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv('imports-85.data', names=cols)


cars


# Select only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 'compression-rate', 
'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
numeric_cars = cars[continuous_values_cols]

numeric_cars.head(5)

# As we learned in this course, we usually can't have any missing values if we want to use them for 
# predictive modeling. Based on the data set preview from the last step, we can tell that the 
# normalized-losses column contains missing values represented using "?". Let's replace these values and 
# look for the presence of missing values in other numeric columns. Let's also rescale the values in the 
# numeric columns so they all range from 0 to 1.

# Use the DataFrame.replace() method to replace all of the ? values with the numpy.nan missing value.
# Because ? is a string value, columns containing this value were cast to the pandas object data type 
# (instead of a numeric type like int or float). After replacing the ? values, determine which columns 
# need to be converted to numeric types. You can use either the DataFrame.astype() or the Series.astype() 
# methods to convert column types.
# Return the number of rows that have a missing value for the normalized-losses column. Determine how 
# you should handle this column. You could:
# Replace the missing values using the average values from that column.
# Drop the rows entirely (especially if other columns in those rows have missing values).
# Drop the column entirely.
# Explore the missing value counts for the other numeric columns and handle any missing values.
# Of the columns you decided to keep, normalize the numeric ones so all values range from 0 to 1.

#Data Cleaning
numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars

numeric_cars = numeric_cars.astype('float')
numeric_cars.isnull().sum()

# Because `price` is the column we want to predict, let's remove any rows with missing `price` values.
numeric_cars = numeric_cars.dropna(subset=['price'])
numeric_cars.isnull().sum()

# Replace missing values in other columns using column means.
numeric_cars = numeric_cars.fillna(numeric_cars.mean())

# Confirm that there's no more missing values!
numeric_cars.isnull().sum()

# Normalize all columnns to range from 0 to 1 except the target column.
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col

# Let's start with some univariate k-nearest neighbors models. Starting with simple models before 
# moving to more complex models helps us structure your code workflow and understand the features better.
# Create a function, named knn_train_test() that encapsulates the training and simple validation process.
# This function should have 3 parameters -- training column name, target column name, and the dataframe 
# object.
# This function should split the data set into a training and test set.
# Then, it should instantiate the KNeighborsRegressor class, fit the model on the training set, and 
# make predictions on the test set.
# Finally, it should calculate the RMSE and return that value.
# Use this function to train and test univariate models using the different numeric columns in the 
# data set. Which column performed the best using the default k value?
# Modify the knn_train_test() function you wrote to accept a parameter for the k value.
# Update the function logic to use this parameter.
# For each numeric column, create, train, and test a univariate model using the following k values 
# (1, 3, 5, 7, and 9). Visualize the results using a scatter plot or a line plot.

# Univariate Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, target_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    # Fit a KNN model using default k value.
    knn.fit(train_df[[train_col]], train_df[target_col])
    
    # Make predictions using model.
    predicted_labels = knn.predict(test_df[[train_col]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse

rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so 
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


#################################################################################
# Let's modify the knn_train_test() function we wrote in the last step to work with multiple columns.
# Instructions

# Modify the knn_train_test() function to accept a list of column names (instead of just a string). 
# Modify the rest of the function logic to use this parameter:
# Instead of using just a single column for train and test, use all of the columns passed in.
# Use a the default k value from scikit-learn for now (we'll tune the k value in the next step).
# Use the best 2 features from the previous step to train and test a multivariate k-nearest neighbors 
# model using the default k value.
# Use the best 3 features from the previous step to train and test a multivariate k-nearest neighbors 
# model using the default k value.
# Use the best 4 features from the previous step to train and test a multivariate k-nearest neighbors 
# model using the default k value.
# Use the best 5 features from the previous step to train and test a multivariate k-nearest neighbors 
# model using the default k value.
# Display all of the RMSE values.

def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

k_rmse_results


############################################################################################
import matplotlib.pyplot as plt
%matplotlib inline

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


# Multivariate Model
# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
series_avg_rmse.sort_values()

################################################################################################
def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

two_best_features = ['horsepower', 'width']
rmse_val = knn_train_test(two_best_features, 'price', numeric_cars)
k_rmse_results["two best features"] = rmse_val

three_best_features = ['horsepower', 'width', 'curb-weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

six_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg', 'length']
rmse_val = knn_train_test(six_best_features, 'price', numeric_cars)
k_rmse_results["six best features"] = rmse_val

k_rmse_results


########################################################################################################
# Let's now optimize the model that performed the best in the previous step.

# Instructions

# For the top 3 models in the last step, vary the hyperparameter value from 1 to 25 and plot the 
# resulting RMSE values.
# Which k value is optimal for each model? How different are the k values and what do you think 
# accounts for the differences?

# Multivariate Model
def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

three_best_features = ['horsepower', 'width', 'curb-weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb-weight', 'city-mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb-weight' , 'city-mpg' , 'highway-mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

k_rmse_results



###########################################################################################
for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

# That's it for the guided steps. Here are some ideas for next steps:

# Modify the knn_train_test() function to use k-fold cross validation instead of test/train validation.
# Modify the knn_train_test() function to perform the data cleaning as well.