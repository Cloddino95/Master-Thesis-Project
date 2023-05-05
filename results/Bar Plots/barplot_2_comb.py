import pandas as pd
import matplotlib.pyplot as plt

data2_comb_agg = pd.read_csv('/results_df_1000_customized_2_COMBINED_AGG.csv')

data2_Comb_Agg = {
    'category': ['Psycho-only', 'SVM-best-only', 'SVR-RBF', 'SVR-Polynomial', 'SVR-Sigmoid', 'Lasso', 'Ridge', 'KNN'],
    'value': [0.73, 0.73531, 0.84185, 0.81664, 0.52700, 0.83643, 0.85120, 0.71145]}

df_2_comb_agg = pd.DataFrame(data2_Comb_Agg)

# set the x coordinates of the bars
x = [1, 3, 5, 7, 9, 11, 13, 15]
color = ['black', 'black', 'blue', 'green', 'orange', 'purple', 'tan', 'cyan']
# create the bar plot
plt.bar(x, height=df_2_comb_agg['value'], color=color, width=0.7, align='center', alpha=0.5)
# set the xticks to the category labels
plt.xticks(x, df_2_comb_agg['category'], rotation=15)
plt.ylabel("R2", rotation=0)
plt.title("R2 for aggregate results - Study 2 combined")
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.ylim(0, 1)
plt.yticks([i/10 for i in range(11)])
plt.savefig('2_AGG_COMB.png')
# display the plot
plt.show()
plt.close()
