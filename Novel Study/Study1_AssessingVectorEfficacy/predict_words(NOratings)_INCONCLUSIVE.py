from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score  # , RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve


model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

# noinspection PyUnresolvedReferences
risk_source_vector_2 = [model[word] for word in valid_risk_source_name_2]

dict_risk_source_vectors_2 = dict(zip(valid_risk_source_name_2, risk_source_vector_2))


def get_vector_for_source_i(a):
    source_name = valid_risk_source_name_2[a]
    return dict_risk_source_vectors_2[source_name]


xi_vectors_2 = []
for i in range(len(valid_risk_source_name_2)):
    xi = get_vector_for_source_i(i)
    xi_vectors_2.append(xi)

mat_xi_300dim_2 = np.array(xi_vectors_2)

data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

risk_ratings_2_small = risk_ratings_2.copy()
risk_ratings_2_small.columns = risk_source_name_2
risk_ratings_2_small = risk_ratings_2_small.drop(risk_ratings_2_small.index[0])
risk_ratings_2_small = risk_ratings_2_small.drop(omitted_word_2, axis=1)
risk_ratings_2_small = risk_ratings_2_small.T
risk_ratings_2_small = risk_ratings_2_small.dropna(axis=1, how='all')
risk_ratings_2_small = risk_ratings_2_small.astype(float)

mean_rows = risk_ratings_2_small.mean(axis=1)
data_300dim_2.insert(0, "mean_ratings", mean_rows)
names = pd.DataFrame(valid_risk_source_name_2, columns=["names"])

X = data_300dim_2.drop('mean_ratings', axis=1)
y = names     # data_300dim_2['mean_ratings']

"""
The task of predicting a category (or a word, in your case) is a classification task rather than a regression task. 
Models like Logistic Regression, Random Forests, and Support Vector Machines (SVM) are used for this purpose.

To use **Support Vector Machines (SVM)** for multi-class classification, you would need to represent your words as integers. 
This can be done with the help of sklearn's LabelEncoder which converts each unique string value into a unique integer.

Remember that using this approach means that the model doesn't actually understand the meaning of the words, it just uses the 
integer representation to classify the input vectors. Also, this approach may not yield good results if the relationship between
the words and the vectors isn't clear or is nonlinear in nature. Using more advanced techniques like RNN or LSTM could yield 
better results if this is the case.
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

encoder = LabelEncoder()
y_integers = encoder.fit_transform(y)

# Split the data into train and test datasets
train_indices, test_indices = train_test_split(range(len(X)), test_size=0.2, random_state=42)

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train_encoded, y_test_encoded = y_integers[train_indices], y_integers[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# Define a range for the 'C' hyperparameter
param_grid = {'C': [0.1, 1, 10, 100, 1000]}
# Create a GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, cv=KFold(n_splits=5), refit=True, verbose=3)

# Train the model using the training sets
grid.fit(X_train, y_train_encoded)

# Print the best parameters
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

# Convert y_train and y_test to integer labels
y_train_integers = encoder.transform(y_train)
y_test_integers = encoder.transform(y_test)

# Train the model
svm = SVC(C=0.1)
svm.fit(X_train, y_train_integers)

# Make predictions
svm_pred_train = svm.predict(X_train)
svm_pred_test = svm.predict(X_test)

# Convert integer predictions back to original labels
svm_pred_train_words = encoder.inverse_transform(svm_pred_train)
svm_pred_test_words = encoder.inverse_transform(svm_pred_test)

# Create dataframes for the results
train_results = pd.DataFrame({'predicted_words': svm_pred_train_words, 'actual_words': y_train.values.flatten()}, index=y_train.index)

test_results = pd.DataFrame({'predicted_words': svm_pred_test_words, 'actual_words': y_test.values.flatten()}, index=y_test.index)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate accuracy
train_accuracy = accuracy_score(y_train.values, svm_pred_train_words)
test_accuracy = accuracy_score(y_test.values, svm_pred_test_words)
print(f'Train Accuracy: {train_accuracy*100:.2f}%')
print(f'Test Accuracy: {test_accuracy*100:.2f}%')

# Show confusion matrix
print('\nConfusion Matrix (Test Set):')
print(confusion_matrix(y_test.values, svm_pred_test_words))

# Show precision, recall, and F1 score
print('\nClassification Report (Test Set):')
print(classification_report(y_test.values, svm_pred_test_words))

"""!!! DOUBLE CHECK --> AT LEAST IT GIVES AS FIRST THE RISK SOURCE ITSELF !!!"""

my_vector = X[0:1].values[0]  # advil

# Calculate cosine similarities
similarities = {}
for word in model.index_to_key:  # Notice the change here
    word_vector = model[word]  # And here
    cosine_similarity = np.dot(my_vector, word_vector) / (np.linalg.norm(my_vector) * np.linalg.norm(word_vector))
    similarities[word] = cosine_similarity

# Get the top 10 most similar words to your vector
most_similar = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:10]

for word, similarity in most_similar:
    print(f'Word: {word}, Similarity: {similarity}')



"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Ridge_ = Ridge().set_params(alpha=4)
Ridge_.fit(X_train, y_train)
Ridge_pred_test = Ridge_.predict(X_test)
Ridge_pred_train = Ridge_.predict(X_train)

SVR_RBF = SVR(kernel='rbf').set_params(C=100)
SVR_RBF.fit(X_train, y_train)
SVR_RBF_pred_test = SVR_RBF.predict(X_test)
SVR_RBF_pred_train = SVR_RBF.predict(X_train)


data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_Bha = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)


# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_train)
SVR_RBF_r2_test2 = SVR_RBF.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF.score(X_train, y_train)

# Metrics for Ridge Regression
Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train)
Ridge_r2_test2 = Ridge_.score(X_test, y_test)
Ridge_r2_train2 = Ridge_.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_Bha = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_Bha.index = ['Ridge', 'SVR-RBF']"""
