import gensim
from gensim.models import KeyedVectors
# import gensim.downloader as api
from Dataset import risk_ratings_1A
import numpy as np
import pandas as pd
# pd.options.mode.chained_assignment = None  # suppress the warning line 171
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RepeatedKFold
from tqdm import tqdm  # to show the progress bar
# import matplotlib.pyplot as plt
# import seaborn as sns
from psychometric_1A import psy_df

# Load the model from disk
model = gensim.models.KeyedVectors.load('word2vec-google-news-300.bin')

# how to create a list made of the values in the first row of risk_ratings_1A
risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()

# replace the space between each 2-words risk source with an underscore, e.g. clothes drier -> clothes_drier
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]

# Check which risk sources are in the model's vocabulary
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]

# Create a list of omitted words
omitted_word_A = [word for word in risk_source_name_A if word not in valid_risk_source_name_A]

# Create a list of Word2Vec vectors for each source
# noinspection PyUnresolvedReferences
risk_source_vector_A = [model[word] for word in valid_risk_source_name_A]

# 3) Create a dictionary source_vectors where the keys are the risk_source_name and the values are the corresponding
# Word2Vec vectors using the dict() function:
dict_risk_source_vectors_A = dict(zip(valid_risk_source_name_A, risk_source_vector_A))

def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_A[a]
    return dict_risk_source_vectors_A[source_name]


# Get the Word2Vec vector for each risk source i
xi_vectors_A = []
for i in range(len(valid_risk_source_name_A)):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors_A.append(xi)

# Convert the list of vectors to a numpy array
mat_xi_300dim_A = np.array(xi_vectors_A)

# create a copy of the above array to use it for the analysis:
data_300dim_A = mat_xi_300dim_A.copy()

# how to change rows name of data_300dim with the words in valid_risk_source_name_A:
data_300dim_A = pd.DataFrame(data_300dim_A, index=valid_risk_source_name_A)

# create a copy of risk_ratings_1A dataframe for the analysis:
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
data_300dim_A.insert(0, "mean_ratings", mean_rows)

# import the psychometric dataframe and drop the column 'mean_ratings'
psy_df1 = psy_df.copy()
psy_df1 = psy_df1.drop('mean_ratings', axis=1)

# concatenate the data_300dim dataframe with the psychometric dataframe in order to perform the analysis on 309 dimensions:
data_300dim_A = pd.concat([data_300dim_A, psy_df1], axis=1)

X = data_300dim_A.drop('mean_ratings', axis=1)
y = data_300dim_A['mean_ratings']

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
n_repeats = 1000
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Convert non-string feature names to strings to avoid FutureWarning
if not all(isinstance(col, str) for col in X.columns):
    X.columns = [str(col) for col in X.columns]
if not isinstance(y.name, str):
    y.name = str(y.name)

# create a list of dictionaries to stores the results. (useful to convert/store the results to a different format)
results_list_A = []

# Loop through each model and each value of C or k
for model_name, model in tqdm(models.items()):
    if model_name == 'KNN Regression':
        for k in parameter_values[model_name]:
            # Set the value of k for the model
            model.set_params(n_neighbors=k)
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})

    else:
        for C in tqdm(parameter_values[model_name]):
            # Set the value of C for the model
            model.set_params(alpha=C) if model_name in ['Lasso Regression', 'Ridge Regression'] else model.set_params(
                C=C)
            # Perform cross-validation
            scores = cross_val_score(model, X=X, y=y, cv=rkf, scoring='r2')
            # Store the results
            results_list_A.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(scores), 'std_score_R2': np.std(scores)})


# Convert the list of results to a pandas DataFrame with the appropriate index
results_df_A = pd.DataFrame(results_list_A, index=range(len(results_list_A)))


# Sort the results by the mean R^2 score
# results_df = results_df.sort_values(by=['mean_score_R2'], ascending=False)

# Save the results dataframe to a csv file
results_df_A.to_csv('results_df_1000_customized_1A_COMBINED.csv', index=False)


