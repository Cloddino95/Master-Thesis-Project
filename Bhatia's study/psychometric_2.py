from Dataset import psychometric_2, risk_ratings_2
import gensim
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error

psychometric_2 = psychometric_2.copy()

# how to create a list made of the values in the first row of risk_ratings_1A
risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()

# replace the space between each 2-words risk source with an underscore, e.g. clothes drier -> clothes_drier
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# create a new vector of names which repeat each name 9 times in order to match the number of columns in psychometric_2
risk_source_name_9dim2 = [word for word in risk_source_name_2 for i in range(9)]

# replace the first row of psychometric_2 with the new vector of names
psychometric_2.iloc[0] = risk_source_name_9dim2

# Load the model from disk
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# Check which risk sources are in the model's vocabulary
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

# Create a list of omitted words
omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

# create a new vector of valid names which repeat each name 9 times in order to match the number of columns in psychometric_2
# valid_risk_source_name_9dim = [word for word in valid_risk_source_name_A for i in range(9)]

# psychometric_2_small = psychometric_2.copy()

# name the columns with the risk source names and remove the first row (that are the r.source names):
psychometric_2.columns = risk_source_name_9dim2
psychometric_2 = psychometric_2.drop(psychometric_2.index[0])

# remove risk sources not included in word2vec model's vocabulary:
# psychometric_2 = psychometric_2.drop(omitted_word_2, axis=1)

psychometric_2 = psychometric_2.astype(float)

# remove NA
psychometric_2 = psychometric_2.dropna(axis=0, how='all')

# do the mean of each column of psychometric_1A_small
mean_psy = psychometric_2.mean(axis=0)

# create a new dataframe with the mean of each row of psychometric_2
new_df = pd.DataFrame(mean_psy)
new_df = new_df.T

# reshape the df in a way that each row is a risk source and each column is a dimension
new_shape = (new_df.shape[1] // 9, 9)

psy_df_2 = pd.DataFrame(new_df.values.reshape(new_shape), columns=["1 voluntariness", "2 immediacy of death",
                                                                   "3 known or unknown risk", "4 knowledge to science",
                                                                   "5 controllability", "6 novelty",
                                                                   "7 catastrophic potential of the risk",
                                                                   "8 the amount of dread",
                                                                   "9 potential for fatal consequences"])
psy_df_2.index = valid_risk_source_name_2

# add a column with the mean of each row
# psy_df.insert(0, "mean_ratings", psy_df.mean(axis=1))

# ----------------------------------------------------------------------------------------------------------------------
risk_ratings_2_small = risk_ratings_2.copy()
risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.index[0])
risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)
risk_ratings_2_small = risk_ratings_2_small.T
risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')
risk_ratings_2_small = risk_ratings_2_small.astype(float)
mean_rows = risk_ratings_2_small.mean(axis=1)
# ----------------------------------------------------------------------------------------------------------------------
psy_df_2.insert(0, "mean_ratings", mean_rows)

# run a simple linear regression to predict the perceived riskiness of the risk sources, yi, from the corresponding xi.
X = psy_df_2.drop('mean_ratings', axis=1)
y = psy_df_2['mean_ratings']

psyco_2_X = X.describe()
mean_psyco_2_X = psyco_2_X.mean(axis=1)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the testing and traing set (R^2 based on the pred() function)
y_pred = model.predict(X_test)
y_pred2 = model.predict(X_train)
r2 = r2_score(y_test, y_pred)
print(f'R^2 score on the testing set PREDICTION: {r2:.2f}')

# Evaluate the model on the testing and training set (R^2 based on the score() function)
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
metric2 = pd.DataFrame(results_list, columns=['R2 test', 'R2 training', 'cv_score_mean', 'cv_score_std'],
                      index=range(1))
metric2.index = ['results']

# columns=['R2 test', 'R2 training', 'cv_score', 'cv_score_std'],
# metrics = pd.DataFrame({'R2 test': [score], 'R2 training': [score2], 'cv_score': [cv_scores.mean(), cv_scores.std()]},
# index=range(len(results_list_A)))

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
metric2 = pd.DataFrame(results_list, index=[0])
metric2.index = ['results']

final2 = metric2.drop(metric2.columns[[1, 2, 5]], axis=1)
final2.columns = ['R2 (mean)', 'R2 (SD)', 'RMSE (mean)', 'RMSE (SD)']
