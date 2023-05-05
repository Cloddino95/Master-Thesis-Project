# here I am using the ratings from the 1B dataset to be forecasted by the 300-vector representation of RANDOM words

from gensim.models import KeyedVectors
from Dataset import risk_ratings_1B
import numpy as np
import gensim
# import gensim.downloader as api
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from tqdm import tqdm

model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Thesis_Clo/word2vec-google-news-300.bin')

risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]

omitted_word_B = [word for word in risk_source_name_B if word not in valid_risk_source_name_B]

# noinspection PyUnresolvedReferences
risk_source_vector_B = [model[word] for word in valid_risk_source_name_B]

dict_risk_source_vectors_B = dict(zip(valid_risk_source_name_B, risk_source_vector_B))


def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_B[a]
    return dict_risk_source_vectors_B[source_name]


xi_vectors_B = []
for i in range(len(valid_risk_source_name_B)):
    xi = get_vector_for_source_i(i)
    xi_vectors_B.append(xi)

mat_xi_300dimB = np.array(xi_vectors_B)

data_300dimB = mat_xi_300dimB.copy()

data_300dimB = pd.DataFrame(data_300dimB, index=valid_risk_source_name_B)
# ---------------------------------------------------------------RATINGS------------------------------------------------
risk_ratings_1B_small = risk_ratings_1B.copy()

risk_ratings_1B_small.columns = risk_source_name_B
risk_ratings_1B_small = risk_ratings_1B_small.drop(risk_ratings_1B_small.index[0])

risk_ratings_1B_small = risk_ratings_1B_small.drop(omitted_word_B, axis=1)

risk_ratings_1B_small = risk_ratings_1B_small.T

risk_ratings_1B_small = risk_ratings_1B_small.dropna(axis=1, how='all')

risk_ratings_1B_small = risk_ratings_1B_small.astype(float)

mean_rows = risk_ratings_1B_small.mean(axis=1)
data_300dimB.insert(0, "mean_ratings", mean_rows)
# ---------------------------------------------------------------END----------------------------------------------------
# ----------------------------------------------------------RANDOM WORDS------------------------------------------------
# Get the list of words in the model
word_list = list(model.index_to_key)
import random
random.seed(42)
# Create a vector of 114 random words
random_words = np.random.choice(word_list, size=114, replace=False)
# Get the vector representation of the random words
random_words_vectors = [model[word] for word in random_words]
# Create a dictionary with the random words and their vector representation
dict_random_words_vectors = dict(zip(random_words, random_words_vectors))


def get_vector_for_source_i(a):
    source_name = random_words[a]
    return dict_random_words_vectors[source_name]


rand_vectors = []
for i in range(len(random_words)):
    xi = get_vector_for_source_i(i)
    rand_vectors.append(xi)

mat_rand = np.array(rand_vectors)

df_mat_rand = mat_rand.copy()

df_mat_rand = pd.DataFrame(df_mat_rand, index=random_words)
mean_rows = mean_rows.values
df_mat_rand.insert(0, "mean_ratings", mean_rows)
# ---------------------------------------------------------------END----------------------------------------------------

X = df_mat_rand.drop('mean_ratings', axis=1)
y = df_mat_rand['mean_ratings']

# Define the values of the parameter C to evaluate (SVM, Lasso & Ridge) and K for KNN
parameter_values = {
    'SVR-RBF': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-polynomial': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'SVR-sigmoid': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Lasso Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    'Ridge Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KNN Regression': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# Define the models to evaluate
models = {'SVR-RBF': SVR(kernel='rbf'),
          'SVR-polynomial': SVR(kernel='poly'),
          'SVR-sigmoid': SVR(kernel='sigmoid'),
          'Lasso Regression': Lasso(),
          'Ridge Regression': Ridge(),
          'KNN Regression': KNeighborsRegressor()}

# Define the k-fold cross validation parameters
n_splits = 10
n_repeats = 10
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list_B = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list_B.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})

# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_random = pd.DataFrame(results_list_B, index=range(len(results_list_B)))

# Save the results dataframe to a csv file
# results_df_B.to_csv('results_df_1000_customized_1B_AGG.csv', index=False)
