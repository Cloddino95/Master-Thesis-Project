import pandas as pd
import matplotlib.pyplot as plt

data1A_comb_agg = pd.read_csv("/results_df_1000_customized_1A_COMBINED_AGG.csv")

data1A_Comb_Agg = {
    'category': ['Psycho-only', 'SVM-best-only', 'SVR-RBF', 'SVR-Polynomial', 'SVR-Sigmoid', 'Lasso', 'Ridge', 'KNN'],
    'value': [0.46, 0.52520, 0.72508, 0.69024, 0.58447, 0.70375, 0.71656, 0.65874]}

df_1A_comb_agg = pd.DataFrame(data1A_Comb_Agg)

# set the x coordinates of the bars
x = [1, 3, 5, 7, 9, 11, 13, 15]
color = ['black', 'black', 'blue', 'green', 'orange', 'purple', 'tan', 'cyan']
# create the bar plot
plt.bar(x, height=df_1A_comb_agg['value'], color=color, width=0.7, align='center', alpha=0.5)
# set the xticks to the category labels
plt.xticks(x, df_1A_comb_agg['category'], rotation=15)
plt.ylabel("R2", rotation=0)
plt.title("R2 for aggregate results - Study 1A combined")
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.ylim(0, 1)
plt.yticks([i/10 for i in range(11)])
plt.savefig('1A_AGG_COMB.png')
# display the plot
plt.show()
plt.close()
