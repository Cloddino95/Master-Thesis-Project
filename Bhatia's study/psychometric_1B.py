from Dataset import psychometric_1B, risk_ratings_1B
import gensim
from gensim.models import KeyedVectors
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error

psychometric_1B = psychometric_1B.copy()  # is it useful?

# how to create a list made of the values in the first row of risk_ratings_1A
risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()

# replace the space between each 2-words risk source with an underscore, e.g. clothes drier -> clothes_drier
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]

# create a new vector of names which repeat each name 9 times in order to match the number of columns in psychometric_1A
risk_source_name_9dimB = [word for word in risk_source_name_B for i in range(9)]

# replace the first row of psychometric_1A with the new vector of names
psychometric_1B.iloc[0] = risk_source_name_9dimB

# Load the model from disk
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# Check which risk sources are in the model's vocabulary
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]

# Create a list of omitted words
omitted_word_B = [word for word in risk_source_name_B if word not in valid_risk_source_name_B]

# create a new vector of valid names which repeat each name 9 times in order to match the number of columns in psychometric_1A
# valid_risk_source_name_9dim = [word for word in valid_risk_source_name_A for i in range(9)]

psychometric_1B_small = psychometric_1B.copy()

# name the columns with the risk source names and remove the first row (that are the r.source names):
psychometric_1B_small.columns = risk_source_name_9dimB
psychometric_1B_small = psychometric_1B_small.drop(psychometric_1B_small.index[0])

# remove risk sources not included in word2vec model's vocabulary:
psychometric_1B_small = psychometric_1B_small.drop(omitted_word_B, axis=1)

psychometric_1B_small = psychometric_1B_small.astype(float)

# remove NA
psychometric_1B_small = psychometric_1B_small.dropna(axis=0, how='all')

# do the mean of each column of psychometric_1A_small
mean_psy = psychometric_1B_small.mean(axis=0)

# create a new dataframe with the mean of each row of psychometric_1A_small
new_df = pd.DataFrame(mean_psy)
new_df = new_df.T

# reshape the df in a way that each row is a risk source and each column is a dimension
new_shape = (new_df.shape[1] // 9, 9)

psy_df_B = pd.DataFrame(new_df.values.reshape(new_shape), columns=["1 voluntariness", "2 immediacy of death",
                                                                   "3 known or unknown risk", "4 knowledge to science",
                                                                   "5 controllability", "6 novelty",
                                                                   "7 catastrophic potential of the risk",
                                                                   "8 the amount of dread",
                                                                   "9 potential for fatal consequences"])
psy_df_B.index = valid_risk_source_name_B

# add a column with the mean of each row
# psy_df.insert(0, "mean_ratings", psy_df.mean(axis=1))

# ----------------------------------------------------------------------------------------------------------------------
risk_ratings_1B_small = risk_ratings_1B.copy()
risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])
risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)
risk_ratings_1B_small = risk_ratings_1B_small.T
risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')
risk_ratings_1B_small = risk_ratings_1B_small.astype(float)
mean_rows = risk_ratings_1B_small.mean(axis=1)
# ----------------------------------------------------------------------------------------------------------------------
psy_df_B.insert(0, "mean_ratings", mean_rows)

# run a simple linear regression to predict the perceived riskiness of the risk sources, yi, from the corresponding xi.
X = psy_df_B.drop('mean_ratings', axis=1)
y = psy_df_B['mean_ratings']

psyco_1B_X = X.describe()
mean_psyco_1B_X = psyco_1B_X.mean(axis=1)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
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
metricB = pd.DataFrame(results_list, columns=['R2 test', 'R2 training', 'cv_score_mean', 'cv_score_std'],
                       index=range(1))
metricB.index = ['results']

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
metricB = pd.DataFrame(results_list, index=[0])
metricB.index = ['results']

finalB = metricB.drop(metricB.columns[[1, 2, 5]], axis=1)
finalB.columns = ['R2 (mean)', 'R2 (SD)', 'RMSE (mean)', 'RMSE (SD)']
