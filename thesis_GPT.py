"""CLO:
 I have a rating yi for a risk source i. I also have, for each source, the i have the word2vec-google-news-300 model
 provided by the Gensim-data library representing the source. For source i, I write this vector as xi,_ where xij is the
 value of the risk source on dimension j of the corresponding Word2Vec vector. My goal it to predict yi from the xi.
 How can I do it?
"""

"""CHATGPT:
To predict yi from xi, you can use a regression model that takes xi as input and outputs the predicted value of yi. 
Here are some steps to get you started:

1) Load the word2vec-google-news-300 model using Gensim.
2) For each risk source i, use the model to get the corresponding Word2Vec vector xi.
3) Load the yi values for each risk source i.
4) Split the data into training and testing sets.
5) Choose a regression model (e.g., linear regression, random forest regression, neural network regression) and fit it 
to the training data.
6) Evaluate the performance of the model on the testing data using metrics such as mean squared error or R-squared.
Below some example code showing how to load the Word2Vec model and get the corresponding vectors for each risk source:
"""

from Gensim import word2vec_googlenews_300
import numpy as np

model = word2vec_googlenews_300

# Get the Word2Vec vector for each risk source i
xi_vectors = []
for i in range(num_sources):  # Replace with the actual number of risk sources I have
    xi = get_vector_for_source_i(i)  # Replace with your own function to get the vector for source i
    xi_vectors.append(xi)

# Convert the list of vectors to a numpy array
X = np.array(xi_vectors)

"""Once you have X (the input features) and y (the target variable), you can split the data into training and testing 
sets and fit a regression model to the training data:"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model to the training data
model = LinearRegression()
model.fit(X_train, y_train)

"""Finally, you can evaluate the performance of the model on the testing data:"""

from sklearn.metrics import mean_squared_error, r2_score

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared:', r2)

# ---------------------------------------------------------------------------------------------------------------------

"""In the provided code,.

You would need to replace "num_sources" with the actual number of risk sources that you have, and replace 
"get_vector_for_source_i(i)" with your own function that retrieves the Word2Vec vector for the i-th risk source.

The line "xi = get_vector_for_source_i(i)" is calling a hypothetical function "get_vector_for_source_i" with the 
argument "i" to retrieve the Word2Vec vector for the i-th risk source. You would need to define this function yourself, 
or use a pre-existing one if available, to obtain the Word2Vec vector for each risk source in your specific dataset."""

# can you define for me the function "xi = get_vector_for_source_i(I)"?

"""CHATGPT:
Sure! The function get_vector_for_source_i(I) is a function that you need to implement yourself, as it will depend on 
the specific way you have of representing the sources.

In the context of the code snippet I provided earlier, the function get_vector_for_source_i(I) should return the 
Word2Vec vector for the source with index I.

For example, let's say you have a **list of source names** ''source_names'', and a dictionary source_vectors where the 
keys are the source names and the values are the corresponding Word2Vec vectors. In that case, you could define the 
get_vector_for_source_i(I) function as follows:"""


def get_vector_for_source_i(I):
    source_name = source_names[I]
    return source_vectors[source_name]


"""To create a dictionary source_vectors where the keys are the source names and the values are the corresponding 
Word2Vec vectors, you can follow these steps:"""
# 1) Load the Word2Vec model using Gensim: (did it)

# 2) Create a list of source names and a corresponding list of Word2Vec vectors for each source:
source_names = ['source1', 'source2', 'source3']  # qui devo mettere le risk sources di bathia.
source_vectors = [model['source1'], model['source2'], model['source3']]  # questa é la stessa cosa di 'king' fatta prima

# 3) Create a dictionary source_vectors where the keys are the source names and the values are the corresponding
# Word2Vec vectors using the dict() function:

source_vectors = dict(zip(source_names, source_vectors))

"""This will create a dictionary where the keys are the source names and 
the values are the corresponding Word2Vec vectors."""

# https://realpython.com/python-zip-function/#understanding-the-python-zip-function
# link per capire zip function come funziona

