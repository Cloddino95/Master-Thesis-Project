from risk_rating_2_Bhatia import valid_risk_source_name_2, y
from scipy.stats import pearsonr
from wordcloud import WordCloud
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import gensim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import learning_curve

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

# RETRIEVE THE MOST SIMILAR WORD
"""modify_word_vectors_2 = []
for word in tqdm(valid_risk_source_name_2[0:5]):
    modify_word = model[word] + model['U.S.'] + model["American"]
    modify_word_vectors_2.append(modify_word)

most_similar_words = []
for modify_word in tqdm(modify_word_vectors_2):
    max_cosine_similarity = -1  # Initialize maximum cosine similarity
    most_similar_word = None  # Initialize most similar word

    for vocab_word in tqdm(model.index_to_key):
        word_vector = model[vocab_word]
        cosine_similarity = np.dot(modify_word, word_vector) / (np.linalg.norm(modify_word) * np.linalg.norm(word_vector))

        if cosine_similarity > max_cosine_similarity:
            max_cosine_similarity = cosine_similarity
            most_similar_word = vocab_word

    most_similar_words.append(most_similar_word)

# now most_similar_words contains the most similar word for each word in valid_risk_source_name_2


most_similar_words_df = pd.DataFrame(most_similar_words, columns=['US risk sources'])

most_similar_words_df.insert(0, 'Risk sources', valid_risk_source_name_2[0:5])"""

# RETRIEVE THE SECOND MOST SIMILAR WORD
modify_word_vectors_2 = []
for word in tqdm(valid_risk_source_name_2):
    modify_word = model[word] + model['USA']
    modify_word_vectors_2.append(modify_word)

second_most_similar_words = []
for modify_word in tqdm(modify_word_vectors_2):
    similarities = []  # List of (cosine_similarity, word) tuples

    for vocab_word in tqdm(model.index_to_key):
        word_vector = model[vocab_word]
        cosine_similarity = np.dot(modify_word, word_vector) / (np.linalg.norm(modify_word) * np.linalg.norm(word_vector))
        similarities.append((cosine_similarity, vocab_word))

    # Sort the list of tuples by cosine similarity in descending order and pick the second one
    similarities.sort(reverse=True)
    second_most_similar_word = similarities[1][1]  # [1][1] because [0][1] would give the most similar word

    second_most_similar_words.append(second_most_similar_word)

second_most_similar_words_df = pd.DataFrame(second_most_similar_words, columns=['US risk sources'])

second_most_similar_words_df.insert(0, 'Risk sources', valid_risk_source_name_2)


valid_name_A = second_most_similar_words_df['US risk sources']
y2 = y.copy()

result_name = valid_name_A  #prediction_1W_R['names']
model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')
risk_rating_result = y2  # prediction_1W_R["Actual values"]

# Step 1: Calculate cosine similarity between words and risk sources
similarity_matrix = np.zeros((len(result_name), len(model.index_to_key)))
for i, risk_source in tqdm(enumerate(result_name)):
    if risk_source in model:
        risk_source_vector = model[risk_source]
        cosine_similarities = np.dot(model.vectors, risk_source_vector) / (np.linalg.norm(model.vectors, axis=1) * np.linalg.norm(risk_source_vector))
        similarity_matrix[i, :] = cosine_similarities

# Step 2: Calculate correlation between cosine similarity values and risk ratings
correlation_values = {}
for j, word in tqdm(enumerate(model.index_to_key)):
    word_similarity_values = similarity_matrix[:, j]
    correlation, _ = pearsonr(word_similarity_values, risk_rating_result)
    correlation_values[word] = correlation

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
plt.title('Words Associated with USA Risk perception')
plt.savefig('wordcloud_study_USA.png')
plt.show()


# to chec if words are in the model's vocabulary
"""words = ['US_citizen', 'US_regulations']
for word in words:
    if model.has_index_for(word):
        print(f"{word} is in the model's vocabulary.")
    else:
        print(f"{word} is NOT in the model's vocabulary.")"""

