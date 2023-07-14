from risk_rating_2_Bhatia import valid_risk_source_name_2, y
from Dataset import risk_ratings_2
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from risk_rating_2_Bhatia import metrics_Bha
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')


# Below is an example for all 200 risk sources
tailor_words = ['medicine', 'travel', 'addiction', 'biological_warfare', 'mental_health',
                'medicine', 'education', 'space', 'weather', 'childbirth', 'sleep',
                'addiction', 'insect', 'transport', 'exercise', 'weather', 'terrorism',
                'mental_health', 'disease', 'food', 'accident', 'disease', 'vehicle',
                'cancer', 'home', 'animal', 'construction', 'deception', 'choking',
                'health', 'education', 'apparel', 'drug', 'cold', 'technology',
                'consumer', 'cooking', 'health', 'workplace', 'crash',
                'sport', 'disease', 'health', 'dispute', 'disease', 'pet', 'alcohol',
                'driving', 'weather', 'water', 'substance_abuse',
                'disaster', 'diet', 'disease', 'technology', 'transport',
                'exercise', 'radiation', 'failure', 'accident', 'hunger', 'politic',
                'health', 'violence', 'fire', 'weapon', 'disaster', 'health',
                'aviation', 'diet', 'social', 'gambling', 'fuel', 'education',
                'weapon', 'fitness', 'fitness', 'cybersecurity', 'weather',
                'tool', 'health', 'health', 'drug', 'person', 'tornado', 'cold', 'sickness', 'web', 'terrorism', 'isolation',
                'pedestrian', 'employment', 'crime', 'weapon', 'craft', 'weather',
                'illumination', 'alcohol', 'gambling', 'deception', 'drug', 'diet',
                'journalists', 'health', 'toxic', 'space', 'army', 'sport', 'crime',
                'homicide', 'crime', 'technology', 'dark', 'health', 'outdoor',
                'obesity', 'flu_pandemic', 'society', 'communication', 'medicine', 'aviation',
                'pollutants', 'PowerPoint_presentation', 'software', 'propaganda', 'pet',
                'politics', 'radiation', 'weather', 'literature', 'economy', 'protest',
                'water', 'crime', 'exercise', 'disease', 'scam', 'ocean', 'animal', 'footwear',
                'violence', 'theft', 'health', 'pedestrian', 'sport', 'sport', 'sport',
                'sleep', 'accident', 'tobacco', 'animal', 'weather', 'fast', 'espionage',
                'staircase', 'hunger', 'weather', 'road', 'mental_health', 'disease',
                'education', 'drug', 'diet', 'warm', 'burn', 'surgery', 'food',
                'water', 'furniture', 'media', 'terrorism', 'terrorism', 'texting', 'theft',
                'weather', 'tobacco', 'tool', 'disaster', 'torture', 'authoritarianism',
                'earthquake', 'accident', 'politics', 'disaster', 'medicine',
                'disaster', 'typing', 'unemployment', 'mental_health', 'diet',
                'volcano', 'walking', 'war', 'water', 'weapon', 'climate', 'drug',
                'internet', 'writing', 'yardwork']

# Each word is chosen to be related to the risk source in your list. For example, for "anxiety" the related word is "mental_health", and for "anthrax" the related word is "biological_warfare". Note that this list is quite arbitrary and may not perfectly capture the relationship between each risk source and its related word. It is recommended to evaluate the performance of this tailored word list and adjust as necessary to improve your model's performance.

"""# concatenate tailor_words and valid_risk_source_name_2
tailor = pd.DataFrame(tailor_words, columns=['tailor_words'])
valid = pd.DataFrame(valid_risk_source_name_2, columns=['valid_risk_source_name_2'])
tailor_valid = pd.concat([valid, tailor], axis=1)
len(tailor)
len(valid_risk_source_name_2)"""

# Ensure both lists are of the same length
assert len(valid_risk_source_name_2) == len(tailor_words)

modify_word_vectors_2 = []
for word, tailor_word in tqdm(zip(valid_risk_source_name_2, tailor_words)):
    modify_word = model[word] + model[tailor_word]
    modify_word_vectors_2.append(modify_word)

mat_modify_vec = np.array(modify_word_vectors_2)
modify_pd = pd.DataFrame(mat_modify_vec, index=valid_risk_source_name_2)

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]
risk_ratings_2_copy = risk_ratings_2.copy()
risk_ratings_2_copy.columns = risk_source_name_2
risk_ratings_2_copy = risk_ratings_2_copy.drop(risk_ratings_2_copy.index[0])
risk_ratings_2_copy = risk_ratings_2_copy.T
risk_ratings_2_copy = risk_ratings_2_copy.dropna(axis=1, how='all')
risk_ratings_2_copy = risk_ratings_2_copy.astype(float)
mean_rows = risk_ratings_2_copy.mean(axis=1)

modify_pd.insert(0, "mean_ratings", mean_rows)

X = modify_pd.drop('mean_ratings', axis=1)
y = modify_pd['mean_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# prediction Ridge
Ridge_norm = Ridge().set_params(alpha=4)
Ridge_norm.fit(X_train, y_train)
Ridge_pred_testnorm = Ridge_norm.predict(X_test)
Ridge_pred_trainnorm = Ridge_norm.predict(X_train)

# prediction SVR-RBF
SVR_RBF_norm = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_norm.fit(X_train, y_train)
SVR_RBF_pred_testnorm = SVR_RBF_norm.predict(X_test)
SVR_RBF_pred_trainorm = SVR_RBF_norm.predict(X_train)

# Dataframe for predictions
data_SVR = {"names": y_test.index.tolist(), "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_testnorm.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_testnorm.flatten(), index=y_test.index), "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test, "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_testnorm.flatten(), index=y_test.index), "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_testnorm.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_norm = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)

# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_testnorm)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_trainorm)
SVR_RBF_r2_test2 = SVR_RBF_norm.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_norm.score(X_train, y_train)

# Metrics for Ridge
Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_testnorm)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_trainnorm)
Ridge_r2_test2 = Ridge_norm.score(X_test, y_test)
Ridge_r2_train2 = Ridge_norm.score(X_train, y_train)

# Dataframe for metrics
results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2, 'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2), 'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_norm = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_norm.index = ['Ridge', 'SVR-RBF']

diff_bh_norm = metrics_Bha - metrics_norm
diff_bh_norm.reset_index(inplace=True)
diff_bh_norm.rename(columns={'index': 'Model'}, inplace=True)


"""for word in tailor_words:
    if word not in model.index_to_key:
        print(word)
"""


