from gensim.models import KeyedVectors
from Dataset import risk_ratings_2
import numpy as np
import pandas as pd
import gensim
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, make_scorer
from tqdm import tqdm
from risk_rating_2_Bhatia import tot_results_df_2_Bhatia, mean_rows, metrics_Bha
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve
import shap
from scipy.stats import pearsonr
from wordcloud import WordCloud

# REMOVE THE SMALL VERSION SINCE IN EXPERIMENTAL 2 THERE ARE NO MISSING WORDS!!! (changed in: risk_ratings_2_copy)
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

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
    closest_word = model.most_similar(word, topn=1)[0][0]
    closest_word_vector = model[closest_word]
    risk_source_vectors_2.append(risk_source_vector)
    closest_word_vectors_2.append(closest_word_vector)

mat_xi_300dim_2 = np.hstack((np.array(risk_source_vectors_2), np.array(closest_word_vectors_2)))
data_300dim_2 = mat_xi_300dim_2.copy()
data_300dim_2 = pd.DataFrame(data_300dim_2, index=valid_risk_source_name_2)

data_300dim_2.insert(0, "mean_ratings", mean_rows)

X = data_300dim_2.drop('mean_ratings', axis=1)
y = data_300dim_2['mean_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [4]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}

Ridge_1W_R = Ridge().set_params(alpha=4)
Ridge_1W_R.fit(X_train, y_train)
Ridge_pred_test1W_R = Ridge_1W_R.predict(X_test)
Ridge_pred_train1W_R = Ridge_1W_R.predict(X_train)

SVR_RBF_1W_R = SVR(kernel='rbf').set_params(C=100)
SVR_RBF_1W_R.fit(X_train, y_train)
SVR_RBF_pred_test1W_R = SVR_RBF_1W_R.predict(X_test)
SVR_RBF_pred_trai1W_R = SVR_RBF_1W_R.predict(X_train)


data_SVR = {"names": y_test.index.tolist(),
            "(SVR-RBF) predicted values": pd.Series(SVR_RBF_pred_test1W_R.flatten(), index=y_test.index)}
SVR_RBF_pred_test2_df = pd.DataFrame(data_SVR)
data_Ridge = {"(Ridge) predicted values": pd.Series(Ridge_pred_test1W_R.flatten(), index=y_test.index),
              "names": y_test.index.tolist()}
Ridge_pred_test2_df = pd.DataFrame(data_Ridge)
data_y = {"Actual values": y_test,
          "(Y - SVR-RBF) Residuals": y_test - pd.Series(SVR_RBF_pred_test1W_R.flatten(), index=y_test.index),
          "(Y - Ridge) Residuals": y_test - pd.Series(Ridge_pred_test1W_R.flatten(), index=y_test.index)}
data_y_df = pd.DataFrame(data_y)
prediction_1W_R = pd.concat([SVR_RBF_pred_test2_df, Ridge_pred_test2_df["(Ridge) predicted values"], data_y_df], axis=1)

# -----------------------CREATING THE RANKING SCHEME FOR THE PREDICTED VALUES (1 BLOCK = 10 RATINGS)--------------------
def assign_block(value):
    for i in range(-100, 100, 10):
        if i <= value < i + 10:
            return f"[{i}, {i + 10})"


prediction_1W_R['Actual Block'] = prediction_1W_R['Actual values'].apply(assign_block)
prediction_1W_R['(SVR-RBF) predicted Block'] = prediction_1W_R['(SVR-RBF) predicted values'].apply(assign_block)
prediction_1W_R['(Ridge) predicted Block'] = prediction_1W_R['(Ridge) predicted values'].apply(assign_block)

prediction_1W_R['(Ridge) predicted Block'] = prediction_1W_R['(Ridge) predicted Block'].where(
    prediction_1W_R['(Ridge) predicted Block'].notnull(), "[90, 100]")
only_ranking = prediction_1W_R.drop(
    ['Actual values', '(SVR-RBF) predicted values', '(Ridge) predicted values', '(Y - SVR-RBF) Residuals',
     '(Y - Ridge) Residuals'], axis=1)
# ----------------------------------------------------------------------------------------------------------------------

# Metrics for SVR-RBF
SVR_RBF_mse_test2 = mean_squared_error(y_test, SVR_RBF_pred_test1W_R)
SVR_RBF_mse_train2 = mean_squared_error(y_train, SVR_RBF_pred_trai1W_R)
SVR_RBF_r2_test2 = SVR_RBF_1W_R.score(X_test, y_test)
SVR_RBF_r2_train2 = SVR_RBF_1W_R.score(X_train, y_train)

# Metrics for Ridge Regression

Ridge_mse_test2 = mean_squared_error(y_test, Ridge_pred_test1W_R)
Ridge_mse_train2 = mean_squared_error(y_train, Ridge_pred_train1W_R)
Ridge_r2_test2 = Ridge_1W_R.score(X_test, y_test)
Ridge_r2_train2 = Ridge_1W_R.score(X_train, y_train)

results_list2_ridge = {'R2 test': Ridge_r2_test2, 'RMSE test': np.sqrt(Ridge_mse_test2), 'R2 training': Ridge_r2_train2,
                       'RMSE training': np.sqrt(Ridge_mse_train2)}
results_list2_SVR = {'R2 test': SVR_RBF_r2_test2, 'RMSE test': np.sqrt(SVR_RBF_mse_test2),
                     'R2 training': SVR_RBF_r2_train2, 'RMSE training': np.sqrt(SVR_RBF_mse_train2)}
metrics_1W_R = pd.DataFrame([results_list2_ridge, results_list2_SVR])
metrics_1W_R.index = ['Ridge', 'SVR-RBF']

diff_bh_1W_R = metrics_Bha - metrics_1W_R
diff_bh_1W_R.reset_index(inplace=True)
diff_bh_1W_R.rename(columns={'index': 'Model'}, inplace=True)

# --------------------------------------------LOOKING AT THE SUPPORT VECTORS USED---------------------------------------
support_vector_indices = SVR_RBF_1W_R.support_
support_vectors = X_train.iloc[support_vector_indices]  # SVR is using all the trainig instances as support vectors
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------SHAP ANALYSIS ON SVR-RBF-----------------------------------------------

# Create an explainer
explainer = shap.KernelExplainer(SVR_RBF_1W_R.predict, X_train)
# Compute SHAP values
shap_values = explainer.shap_values(X_test)
# Plot the SHAP values for the first prediction
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
# retain the first 300 columns in shap_values
shap300 = shap_values[:, :300]
# retain the second 300 columns in shap_values
shap300_600 = shap_values[:, 300:600]

from scipy.spatial.distance import cosine
closest_words_300 = []
for shap_vector in shap300:
    closest_word = None
    closest_dist = float('inf')
    for word in model.index_to_key:
        word_vector = model.get_vector(word)
        dist = cosine(word_vector, shap_vector)
        if dist < closest_dist:
            closest_word = word
            closest_dist = dist
    # Append the closest word to the list
    closest_words_300.append(closest_word)

closest_word300_df = pd.DataFrame(closest_words_300, columns=['closest_word'])
realword_shapword = pd.concat([pd.DataFrame(y_test.index.tolist()), closest_word300_df], axis=1)

# ------------------------------------------------------ CV Analysis ---------------------------------------------------
# CROSS VALIDATION NEEDED TO ENSURE GENERALISATION AND TO PERFROM THE BELOW HYPERPARAMETER TUNING

parameter_values2 = {'SVR-RBF': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                     'Ridge Regression': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

models2 = {'SVR-RBF': SVR(kernel='rbf'), 'Ridge Regression': Ridge()}

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

results_list_2 = []

for model_name, model in tqdm(models2.items()):
    if model_name == 'Ridge Regression':
        for k in parameter_values2[model_name]:
            model.set_params(alpha=k)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append({'model': model_name, 'parameter': k, 'mean_score_R2': np.mean(r2_scores),
                                   'std_score_R2': np.std(r2_scores),
                                   'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})
    else:
        for C in tqdm(parameter_values2[model_name]):
            model.set_params(C=C)
            r2_scores = cross_val_score(model, X=X, y=y, cv=kf, scoring='r2')
            mse_scores = - cross_val_score(model, X=X, y=y, cv=kf, scoring=mse_scorer)
            results_list_2.append(
                {'model': model_name, 'parameter': C, 'mean_score_R2': np.mean(r2_scores),
                 'std_score_R2': np.std(r2_scores),
                 'mean_score_RMSE': np.mean(np.sqrt(mse_scores)), 'std_score_RMSE': np.std(np.sqrt(mse_scores))})


results_df2_1Closest_R = pd.DataFrame(results_list_2, index=range(len(results_list_2)))

tot_results_df2_1Closest_R = results_df2_1Closest_R.drop('parameter', axis=1).groupby('model').mean()
difference_Bh_1C_R = tot_results_df_2_Bhatia - tot_results_df2_1Closest_R  # WHEN NEGATIVE THE NEW MODEL IS BETTER
difference_Bh_1C_R.reset_index(inplace=True)

# --------------------------------------------------Ridge---------------------------------------------------------------

"""The ***fill_between function*** is used to indicate the standard deviation of the RMSE score at each meta-parameter value, 
giving you an idea of the variability of the RMSE at each point."""
# -------------------------------------------ALPHA VALUES---------------------------------------------------------------
"""The penalization coefficient (alpha) in Ridge Regression serves to control the complexity of the model.
In practice, alpha is often chosen through cross-validation. This involves running Ridge Regression with different alpha values 
and seeing which one results in the lowest cross-validated error (or in your case, RMSE).
This is exactly what you are doing in your code below. Based on the plot generated from the code I provided, you would choose 
the alpha that gives the lowest mean RMSE.

!!!!In some cases, it might be better to choose a slightly higher alpha even if it doesn't give the absolute lowest RMSE if the 
standard deviation of the RMSE is significantly lower, as this would suggest that the model performs more consistently across 
different splits of the data.!!!!"""

# Select the Ridge Regression results
ridge_results = results_df2_1Closest_R[results_df2_1Closest_R['model'] == 'Ridge Regression']

# Set up the figure
plt.figure(figsize=(12, 8))
# Plot the mean RMSE score and fill the surrounding area with one standard deviation range
plt.plot(ridge_results['parameter'], ridge_results['mean_score_RMSE'], label='Mean RMSE', color='navy')
plt.fill_between(ridge_results['parameter'],
                 ridge_results['mean_score_RMSE'] - ridge_results['std_score_RMSE'],
                 ridge_results['mean_score_RMSE'] + ridge_results['std_score_RMSE'],
                 color='skyblue', alpha=0.5,
                 label='Mean RMSE ± 1 std. dev.')
# Define the x-axis ticks
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Set major ticks at increments of 0.2
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Set minor ticks at increments of 0.1
# Styling and labelling
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines for better reading
plt.title('RMSE as a function of Ridge Regression alpha parameter', fontsize=15)
plt.xlabel('Ridge Alpha Parameter', fontsize=12)
plt.ylabel('Mean RMSE', fontsize=12)
plt.legend(fontsize=10)
sns.despine()  # Remove the top and right spines for a cleaner look
# Show the plot
plt.savefig('Ridge_alpha_1close_R.png')
plt.show()


# --------------------------------------------PATH DIAGRAM-------------------------------------------------------------
# It shows how the coefficients of the Ridge regression model change as the penalty parameter alpha increases.

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# Initialize an array to hold the coefficients
coefs = []

# For each alpha value
for a in alphas:
    # Create and fit the Ridge model
    ridge = Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    # Store the coefficients
    coefs.append(ridge.coef_)

# Create the plot
plt.figure(figsize=(10, 6))
ax = plt.gca()
ax.plot(alphas, coefs)
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
xticks = np.arange(min(alphas), max(alphas)+0.1, 0.1)
plt.xticks(xticks)
plt.axis('tight')
plt.savefig('Ridge_Path_DiagramR1close.png')
plt.show()


# --------------------------------------------LEARNING CURVES-----------------------------------------------------------
"""This script below will first define the plot_learning_curve function. Then it will loop over the alpha values for the Ridge
 regression model specified in parameter_values2['Ridge Regression']. For each alpha value, it will create a Ridge regression 
 estimator with that alpha value, generate a learning curve, and display it. At the end, it will show all the learning curve plots."""

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend(loc="best")
    return plt


# Iterate over the Ridge alpha values
ridge_alphas = [1]

for alpha_value in ridge_alphas:
    # Title for the plot
    title = f"Learning Curves (Ridge Regression, alpha={alpha_value})"
    # Ridge estimator for the current alpha value
    estimator = Ridge(alpha=alpha_value)
    # Plot the learning curve
    plot_learning_curve(estimator, title, X, y, cv=kf)
    plt.savefig(f"LearnCurve_Ridge_R1close.png")
    plt.show()


# ------------------------------------- same analysis but for SVR-RBF --------------------------------------------------


# -------------------------------------------- C value -----------------------------------------------------------
"""With this plot, you can observe how the RMSE changes with different C values (the penalty parameter of the error term) for SVR 
with the RBF kernel. A smaller RMSE indicates a better fit of the model to the data."""


# Get SVR-RBF results
svr_results = results_df2_1Closest_R[results_df2_1Closest_R['model'] == 'SVR-RBF']

# Set up the figure
plt.figure(figsize=(12, 8))

# Plot the mean RMSE score and fill the surrounding area with one standard deviation range
plt.plot(svr_results['parameter'], svr_results['mean_score_RMSE'], label='Mean RMSE', color='navy')
plt.fill_between(svr_results['parameter'],
                 svr_results['mean_score_RMSE'] - svr_results['std_score_RMSE'],
                 svr_results['mean_score_RMSE'] + svr_results['std_score_RMSE'],
                 color='skyblue', alpha=0.5,
                 label='Mean RMSE ± 1 std. dev.')
# Define the x-axis ticks
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Set major ticks at increments of 100
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))  # Set minor ticks at increments of 50
# Styling and labelling
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Light grid lines for better reading
plt.title('RMSE as a function of SVR-RBF C parameter', fontsize=15)
plt.xlabel('SVR-RBF C Parameter', fontsize=12)
plt.ylabel('Mean RMSE', fontsize=12)
plt.legend(fontsize=10)
sns.despine()  # Remove the top and right spines for a cleaner look
plt.savefig('SVR_C.png')
plt.show()

# --------------------------------------------LEARNING CURVES-----------------------------------------------------------

# C values you're interested in
svr_C_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# For each C value
for C_value in svr_C_values:
    title = f"Learning Curves (SVR-RBF, C={C_value})"
    estimator = SVR(kernel='rbf', C=C_value)

    # Plot the learning curve using RMSE
    plot_learning_curve(estimator, title, X, y, cv=kf, n_jobs=None)
    plt.savefig(f"LearnCurve_SVR_C{C_value}.png")
    plt.show()

# --------------------------------------------WORD CLOUD-----------------------------------------------------------

result_name = valid_risk_source_name_2  #prediction_1W_R['names']
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')
risk_rating_result = y  # prediction_1W_R["Actual values"]

# Step 1: Calculate cosine similarity between words and risk sources
similarity_matrix = np.zeros((len(result_name), len(model.index_to_key)))
for i, risk_source in tqdm(enumerate(result_name)):
    if risk_source in model:
        risk_source_vector = model[risk_source]
        cosine_similarities = np.dot(model.vectors, risk_source_vector) / (np.linalg.norm(model.vectors, axis=1) * np.linalg.norm(risk_source_vector))
        similarity_matrix[i, :] = cosine_similarities

# Step 2: Calculate correlation between cosine similarity values and risk ratings
correlation_values = {}
p_values = {}
for j, word in tqdm(enumerate(model.index_to_key)):
    if word not in result_name:  # Exclude risk sources
        word_similarity_values = similarity_matrix[:, j]
        correlation, p_value = pearsonr(word_similarity_values, risk_rating_result)
        correlation_values[word] = correlation
        p_values[word] = p_value

# You can use the p_values in any way you like. For example, print out the words with p < 0.05 and correlation > 0
"""for word, correlation in correlation_values.items():
    p_value = p_values[word]
    if correlation > 0 and p_value < 0.05:
        print(f'{word}: correlation = {correlation}, p-value = {p_value}')"""

# Step 3: Determine words with the strongest association with risk
top_n = 30  # Specify the number of top words you want to select
strongest_associations = sorted(((word, correlation) for word, correlation in correlation_values.items() if correlation > 0), key=lambda x: x[1], reverse=True)[:top_n]

# Generate word cloud
word_cloud_data = {word: correlation for word, correlation in strongest_associations}
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_cloud_data)

# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Words Associated with Risk')
plt.show()
