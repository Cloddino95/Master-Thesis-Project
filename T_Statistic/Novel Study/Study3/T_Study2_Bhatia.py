import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from risk_rating_2_Bhatia import X as X_psy, y as y_psy

X_psy, y_psy = X_psy.copy(), y_psy.copy()

X1, X_temp, y1, y_temp = train_test_split(X_psy, y_psy, test_size=0.66, random_state=42)
X2, X3, y2, y3 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

parameter_values = {'SVR-RBF': [100],
                    'Ridge Regression': [4]}

models = {'SVR-RBF': SVR(kernel='rbf'),
          'Ridge Regression': Ridge()}


metrics_list = []
prediction_list = []
datasets_X = [X1, X2, X3]
datasets_y = [y1, y2, y3]
subset_names = ['Subset1', 'Subset2', 'Subset3']

for subset_name, X_subset, y_subset in zip(subset_names, datasets_X, datasets_y):
    subset_results = {}
    subset_predictions = {}
    for model_name, model in tqdm(models.items()):
        if model_name == 'SVR-RBF':
            model.set_params(C=100)
        elif model_name == 'Ridge Regression':
            model.set_params(alpha=4)

        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_subset)
        residuals = y_subset - y_pred
        subset_predictions[f'{model_name} Predictions'] = y_pred

        # Store metrics
        mse = mean_squared_error(y_subset, y_pred)
        r2 = model.score(X_subset, y_subset)
        metrics_list.append({'Subset': subset_name, 'Model': model_name, 'RMSE_psy': np.mean(np.sqrt(mse)), 'R2_psy': np.mean(r2)})
        prediction_list.append({'Subset': subset_name, 'Model': model_name, 'y_pred_psy': y_pred})


metrics_bha2 = pd.DataFrame(metrics_list)
prediction_bha2 = pd.DataFrame(prediction_list)













