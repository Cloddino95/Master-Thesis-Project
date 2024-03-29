from Dataset import psychometric_1A, risk_ratings_1A
import gensim
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error

psychometric_1A = psychometric_1A.copy()  # is it useful? yes because any changes you make after this line will not affect the original data. But  points to this new copied DataFrame.
# 125(risk sources)*9(dimensions) = 1125 columns
# 150 rows (75 x 2) but her we have 75  real rows because the rest is NA

# how to create a list made of the values in the first row of risk_ratings_1A
risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()

# replace the space between each 2-words risk source with an underscore, e.g. clothes drier -> clothes_drier
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]

"""all the risk sources names in psychometric_1A matches with the risk sources names in risk_ratings_1A"""

# create a new vector of names which repeat each name 9 times in order to match the number of columns in psychometric_1A
risk_source_name_9dim = [word for word in risk_source_name_A for i in range(9)]

# replace the first row of psychometric_1A with the new vector of names
psychometric_1A.iloc[0] = risk_source_name_9dim

# Load the model from disk
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# Check which risk sources are in the model's vocabulary
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]

# Create a list of omitted words
omitted_word_A = [word for word in risk_source_name_A if word not in valid_risk_source_name_A]

# create a new vector of valid names which repeat each name 9 times in order to match the number of columns in psychometric_1A
# valid_risk_source_name_9dim = [word for word in valid_risk_source_name_A for i in range(9)]

psychometric_1A_small = psychometric_1A.copy()

# name the columns with the risk source names and remove the first row (that are the r.source names):
psychometric_1A_small.columns = risk_source_name_9dim
psychometric_1A_small = psychometric_1A_small.drop(psychometric_1A_small.index[0])

# remove risk sources not included in word2vec model's vocabulary:
psychometric_1A_small = psychometric_1A_small.drop(omitted_word_A, axis=1)

# remove NA
psychometric_1A_small = psychometric_1A_small.dropna(axis=0, how='all')

psychometric_1A_small = psychometric_1A_small.astype(float)

# do the mean of each column of psychometric_1A_small
mean_psy = psychometric_1A_small.mean(axis=0)

# create a new dataframe with the mean of each row of psychometric_1A_small
new_df = pd.DataFrame(mean_psy)
new_df = new_df.T

# reshape the df in a way that each row is a risk source and each column is a dimension
new_shape = (new_df.shape[1] // 9, 9)

psy_df = pd.DataFrame(new_df.values.reshape(new_shape), columns=["1 voluntariness", "2 immediacy of death",
                                                                 "3 known or unknown risk", "4 knowledge to science",
                                                                 "5 controllability", "6 novelty",
                                                                 "7 catastrophic potential of the risk",
                                                                 "8 the amount of dread",
                                                                 "9 potential for fatal consequences"])
psy_df.index = valid_risk_source_name_A

# add a column with the mean of each row
#psy_df.insert(0, "mean_ratings", psy_df.mean(axis=1))

# ----------------------------------------------------------------------------------------------------------------------
risk_ratings_1A_small = risk_ratings_1A.copy()
# name the columns with the risk source names and remove the first row (that are the r.source names):
risk_ratings_1A_small.columns = risk_source_name_A
risk_ratings_1A_small = risk_ratings_1A_small.drop(risk_ratings_1A_small.index[0])
# remove risk sources not included in word2vec model's vocabulary:
risk_ratings_1A_small = risk_ratings_1A_small.drop(omitted_word_A, axis=1)
# transpose the risk_ratings_1A_small dataframe:
risk_ratings_1A_small = risk_ratings_1A_small.T
# remove all the columns of risk_ratings_1A_small that are composed of only NA values:
risk_ratings_1A_small = risk_ratings_1A_small.dropna(axis=1, how='all')
# the values in risk_ratings_1A_small are not float but objects, so I need to convert them to float:
risk_ratings_1A_small = risk_ratings_1A_small.astype(float)
# calculate the mean of each row (risk source) and add it to the data_300dim dataframe:
mean_rows = risk_ratings_1A_small.mean(axis=1)
# ----------------------------------------------------------------------------------------------------------------------
psy_df.insert(0, "mean_ratings", mean_rows)

"""
# rescale the mean value from [-100, 100] to [1, 7]
min_val = -100
max_val = 100
new_min = 1
new_max = 7
psy_df['mean_ratings'] = ((new_max - new_min) / (max_val - min_val)) * (psy_df['mean_ratings'] - min_val) + new_min """

# run a simple linear regression to predict the perceived riskiness of the risk sources, yi, from the corresponding xi.
X = psy_df.drop('mean_ratings', axis=1)
y = psy_df['mean_ratings']

psyco_1A_X = X.describe()
mean_psyco_1A_X = psyco_1A_X.mean(axis=1)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing and training set
score = model.score(X_test, y_test)
score2 = model.score(X_train, y_train)
print(f'R^2 score on the testing set: {score:.2f}')
print(f'R^2 score on the training set: {score2:.2f}')

# Calculate cross-validated R2 score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-validated R^2 score: {cv_scores.mean():.2f} +/- {cv_scores.std():.2f}')

# create a pd dataframe with the metrics (score and cv_score) results of the linear regression
results_list = {'R2 test': score, 'R2 training': score2, 'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std()}
metricA = pd.DataFrame(results_list, columns=['R2 test', 'R2 training', 'cv_score_mean', 'cv_score_std'],
                       index=range(1))
metricA.index = ['results']

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.2f}')

# Calculate cross-validated RMSE
mse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-mse_scores)
print(f'Cross-validated RMSE: {rmse_scores.mean():.2f} +/- {rmse_scores.std():.2f}')

# add the results to the dataframe
results_list.update({'RMSE test': rmse, 'cv_RMSE_mean': rmse_scores.mean(), 'cv_RMSE_std': rmse_scores.std()})
metricA = pd.DataFrame(results_list, index=[0])
metricA.index = ['results']

final = metricA.drop(metricA.columns[[1, 2, 5]], axis=1)
final.columns = ['R2 (mean)', 'R2 (SD)', 'RMSE (mean)', 'RMSE (SD)']
