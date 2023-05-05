import pandas as pd
import matplotlib.pyplot as plt

data_1A_agg = pd.read_csv('/results_df_1000_customized_1A_AGG.csv')

data1_A = {
    'category': ['Psychometric', 'SVR-RBF', 'SVR-Polynomial', 'SVR-Sigmoid', 'Lasso', 'Ridge', 'KNN'],
    'value': [0.46, 0.52520, 0.36666, 0.46015, 0.47338, 0.52215, 0.33836]}

df_A = pd.DataFrame(data1_A)

std = [0.46, 0.22716, 0.25028, 0.25043, 0.29111, 0.26698, 0.30220]


# set the x coordinates of the bars
x = [1, 3, 5, 7, 9, 11, 13]
color = ['red', 'green', 'blue', 'grey', 'orange', 'purple', 'tan']
# create the bar plot
plt.bar(x, height=df_A['value'], color=color, width=0.7, align='center', alpha=0.5)
# set the xticks to the category labels
plt.xticks(x, df_A['category'], rotation=15)
# add the error bars
#plt.errorbar(x, df_A['value'], yerr=std, fmt='none', color='black', capsize=4)
plt.ylabel("R2", rotation=0)
plt.title("R2 for aggregate results - Study 1A")
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.ylim(0, 1)
plt.yticks([i/10 for i in range(11)])
plt.savefig('1A_AGG.png')
# display the plot
plt.show()
plt.close()

# TODO: HOW TO ADD ERROR BARS TO THE PLOT

"""
import numpy as np

# set the values and standard deviations
values = [0.46, 0.52520, 0.36666, 0.46015, 0.47338, 0.52215, 0.33836]
std = [0.05, 0.03, 0.02, 0.03, 0.04, 0.02, 0.05]

# set the x coordinates of the bars
x = [1, 3, 5, 7, 9, 11, 13]
color = ['red', 'green', 'blue', 'grey', 'orange', 'purple', 'tan']

# create the bar plot
plt.bar(x, height=values, color=color, width=0.7, align='center', alpha=0.5)

# add the error bars
plt.errorbar(x, values, yerr=std, fmt='none', color='black', capsize=4)

# set the xticks to the category labels
plt.xticks(x, ['Psychometric', 'SVR-RBF', 'SVR-Polynomial', 'SVR-Sigmoid', 'Lasso', 'Ridge', 'KNN'], rotation=15)

# set the y axis limits and ticks
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.ylabel("R2", rotation=0)
plt.title("R2 for aggregate results - Study 1A")
plt.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
plt.savefig('1A_AGG.png')
plt.show()
plt.close()

"""