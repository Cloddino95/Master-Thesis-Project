from Dataset import risk_ratings_2
from risk_rating_2_Bhatia import mean_rows
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


model = gensim.models.KeyedVectors.load(
    '/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]

# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

omitted_word_2 = [word for word in risk_source_name_2 if word not in valid_risk_source_name_2]

risk_source_vectors_2 = []
closest_word_vectors_2 = []
for word in valid_risk_source_name_2:
    risk_source_vector = model[word]
    closest_words = model.most_similar(word, topn=10)
    closest_word_vectors = [model[closest_word] for closest_word, _ in closest_words]
    risk_source_vectors_2.append(risk_source_vector)
    closest_word_vectors_2.extend(closest_word_vectors)

mat_closest_word_vectors_2 = np.array(closest_word_vectors_2).reshape(len(valid_risk_source_name_2), 3000)
mat_xi_300dim_2 = np.hstack((np.array(risk_source_vectors_2), mat_closest_word_vectors_2))
data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

data_300dim_2.insert(0, "mean_ratings", mean_rows)

X = data_300dim_2.drop('mean_ratings', axis=1)
y = data_300dim_2['mean_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameter_values = {'SVR-RBF': 100,
                    'Ridge Regression': 4}

models = {'SVR-RBF': SVR(kernel='rbf', C=parameter_values['SVR-RBF']),
          'Ridge Regression': Ridge(alpha=parameter_values['Ridge Regression'])}

residuals_df = pd.DataFrame()

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    shapiro_test = stats.shapiro(residuals)
    temp_df = pd.DataFrame({
        'model': [model_name],
        'parameter': [parameter_values[model_name]],
        'shapiro_statistic': [shapiro_test.statistic],
        'shapiro_p_value': [shapiro_test.pvalue]})
    residuals_df = pd.concat([residuals_df, temp_df], ignore_index=True)

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='-')  # this line represents no residuals
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted Values for {model_name} with Parameter={parameter_values[model_name]}')
    plt.savefig(f"R_10W_{model_name}_{parameter_values[model_name]}_residuals.png")
    plt.show()

# If you want to use the Ridge and SVR models later, they are stored in the models dictionary
Ridge_1W_R = models['Ridge Regression']
Ridge_pred_test1W_R = Ridge_1W_R.predict(X_test)
Ridge_pred_train1W_R = Ridge_1W_R.predict(X_train)

SVR_RBF_1W_R = models['SVR-RBF']
SVR_RBF_pred_test1W_R = SVR_RBF_1W_R.predict(X_test)
SVR_RBF_pred_trai1W_R = SVR_RBF_1W_R.predict(X_train)
