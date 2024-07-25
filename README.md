Introduction: We work on the California Housing Dataset in this project. The data contains information from the 1990 California Census. The aim is to predict the market value of a house based on features like latitude, longitude, age of house and number of bedrooms, among few others.

Importing the libraries: We import sklearn, Numpy, Pandas, Matplotlib and Pyplot libraries. The random seed is set to 42 so that the code can produce the same results every time it is executed

Loading the dataset: The dataset is located in the path- ‘/cxldata/datasets/project/housing/housing.csv’. It is read using the read_csv function of Pandas

Exploring the dataset: ‘info’ and ‘describe’ methods are used to obtain more information about the dataset. ‘hist’ method is used to plot histograms of all features of the dataset

Splitting the dataset: We split the dataset into train and test sets. We use the StratifiedShuffleSplit method from the sklearn library which provides train/test indices to split data in train/test sets. ‘test_size’ is set as 0.2 to create a 80:20 split in the dataset. 

Visualizing the geographic distribution of the data: We visualize how the income categories are distributed geographically to get a better understanding of how the housing prices are related to the location (e.g., close to the ocean) and to the population density. We do this by creating a scatter plot using Matplotlib

Creating a correlation matrix: We add three new relevant features to the dataset through feature engineering. We then create a correlation matrix using the ‘corr’ method to see the correlation coefficients between different variables. The correlation coefficient is a statistical measure of the strength of the relationship between the relative movements of two variables. To visualize it, we plot a scatter plot using the scatter_matrix method from Pandas

Filling in the missing data: We impute the missing values using the SimpleImputer class by considering the median value for that feature. We do not consider mean as it can get affected by outliers

Handling Categorical Attributes: The model does not understand categorical values, so we turn this into a numerical value using onehot encoding. Onehot encoding creates one binary attribute per category: one attribute equal to 1 when the category is <1H OCEAN (and 0 otherwise), another attribute equal to 1 when the category is INLAND (and 0 otherwise), and so on. The output is a SciPy sparse matrix, instead of a NumPy array. After onehot encoding, we get a matrix with thousands of columns, and the matrix is full of 0s except for a single 1 per row. Using up tons of memory mostly to store zeros would be very wasteful, so instead a sparse matrix only stores the location of the nonzero elements.

Creating custom transfer and transformation pipelines: We create a custom transformer to combine the attributes that we created earlier. Then we use a pipeline to process the data by first imputing it using SimpleImputer, then using the custom transformer created earlier to merge the columns, and finally, using the StandardScaler class to scale the entire training data

Training a Decision Tree model: We train a Decision Tree model on the data we prepared to see how it performs using the DecisionTreeRegressor class from Scikit-learn. To evaluate the performance of our model, we import the mean_squared_error class from Scikit-learn. We predict our model using the ‘predict’ method. On evaluation, the mean squared error comes out to be 0, hinting at a possible case of overfitting

Training a Random Forest Model: We then train a Random Forest model the same way using the RandomForestRegressor class from Scikit-learn. We get a rmse of 19561.60, which is also lower than expected

Fine tuning the model with Cross Validation: Cross validation is a resampling technique that is used to evaluate machine learning models on a limited data sample. The training set is split into k smaller sets. A model is trained using k-1 of the folds as training data. The resulting model is validated on the remaining part of the data. The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. Using cross_val_score from Scikit-learn, we calculate the cross validation score of our decision tree model and random forest model. The random forest model with a mean rmse score of 50696.80 performs better than the decision tree model with a mean rmse score of 71407.70. Notably, both perform worse than when trained individually, which strongly hints that the individual models were overfitting the training data.

Fine tuning the model with Grid Search: We further fine tune our model using hyper parameter tuning through GridSeachCV. It loops through predefined hyperparameters and fits our estimator (model) on the training set. We then select the best set of parameters from the listed hyperparameters to use with our Random Forest model.

Analyzing and evaluating the best model: Finally, we make predictions using our model. We also evaluate those predictions to determine how good our model is in predicting attributes it has not seen. The model predicts the median house values of the test set with a root mean squared error of 47730.23.
