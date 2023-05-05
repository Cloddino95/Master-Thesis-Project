import pandas as pd
import matplotlib.pyplot as plt

data2 = pd.read_csv("/results_df_1000_customized_2_AGG.csv")

data2_agg = {
    'category': ['Psychometric', 'SVR-RBF', 'SVR-Polynomial', 'SVR-Sigmoid', 'Lasso', 'Ridge', 'KNN'],
    'value': [0.73, 0.73531, 0.39081, 0.68672, 0.59906, 0.72123, 0.39191]}

df_2 = pd.DataFrame(data2_agg)

# set the x coordinates of the bars
x = [1, 3, 5, 7, 9, 11, 13]
color = ['red', 'green', 'blue', 'grey', 'orange', 'purple', 'tan']
# create the bar plot
plt.bar(x, height=df_2['value'], color=color, width=0.7, align='center', alpha=0.5)
# set the xticks to the category labels
plt.xticks(x, df_2['category'], rotation=15)
plt.ylabel("R2", rotation=0)
plt.title("R2 for aggregate results - Study 2")
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.ylim(0, 1)
plt.yticks([i/10 for i in range(11)])
plt.savefig('2_AGG.png')
# display the plot
plt.show()
plt.close()
