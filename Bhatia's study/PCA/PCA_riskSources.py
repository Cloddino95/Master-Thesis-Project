import gensim
from gensim.models import KeyedVectors
from Dataset import risk_ratings_1A, risk_ratings_1B, risk_ratings_2
import collections
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

model = gensim.models.KeyedVectors.load('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell Bhatia/Embeddings_Risk_Perception/Word2Vec_downloaded/word2vec-google-news-300.bin')

risk_source_name_A = risk_ratings_1A.iloc[0].values.tolist()
risk_source_name_A = [word.replace(' ', '_') for word in risk_source_name_A]
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_A = [word for word in risk_source_name_A if word in vocab]

risk_source_name_B = risk_ratings_1B.iloc[0].values.tolist()
risk_source_name_B = [word.replace(' ', '_') for word in risk_source_name_B]
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_B = [word for word in risk_source_name_B if word in vocab]

risk_source_name_2 = risk_ratings_2.iloc[0].values.tolist()
risk_source_name_2 = [word.replace(' ', '_') for word in risk_source_name_2]
# noinspection PyUnresolvedReferences
vocab = model.key_to_index
valid_risk_source_name_2 = [word for word in risk_source_name_2 if word in vocab]

# tot risk sources
tot_risk_name = valid_risk_source_name_A + valid_risk_source_name_B + valid_risk_source_name_2
# check if there are duplicates
duplicates = [item for item, count in collections.Counter(tot_risk_name).items() if count > 1]
# remove duplicates
tot_risk_name = list(dict.fromkeys(tot_risk_name))


# Obtain vector representations of risk sources
vectors = [model[word] for word in tot_risk_name]

# Calculate pairwise cosine similarity
sim_matrix = cosine_similarity(vectors)

# Perform PCA to reduce the dimensions
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(sim_matrix)

# Create a scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1],  color='black')

# Annotate the points with the risk source names for a subset of points
# Here I'm considering the points in the first, second, third and fourth quartiles.
# You can change this condition to better suit your needs.

plt.title("PCA of risk sources")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig('PCA_riskSources.png')
plt.show()

# ---------------------DATASET: drugs & medicine------------------------------------------------------------------------

df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(len(vectors[0]))])
df['risk_source'] = tot_risk_name

plt.figure(figsize=(20, 20))
# Perform the PCA on the vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)
df["pca-one"] = pca_result[:, 0]
df["pca-two"] = pca_result[:, 1]

# Plot each point in the PCA
sns.scatterplot(x="pca-one", y="pca-two", data=df, color="black")

# set the limits
plt.xlim([0.9, 2])
plt.ylim([-0.2, 1.5])

# Annotate the points within the specified region
subset_df = df[(df['pca-one'] >= 0.9) & (df['pca-one'] <= 2) & (df['pca-two'] >= -0.2) & (df['pca-two'] <= 1.5)]
for i, point in subset_df.iterrows():
    plt.text(point['pca-one'] + 0.01, point['pca-two'] + 0.01, str(point['risk_source']), fontsize=13)
plt.savefig('PCA_region_drugMedicine.png')
plt.show()

# ---------------------DATASET: hobbies & sport ------------------------------------------------------------------------

df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(len(vectors[0]))])
df['risk_source'] = tot_risk_name

plt.figure(figsize=(20, 20))
# Perform the PCA on the vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)
df["pca-one"] = pca_result[:, 0]
df["pca-two"] = pca_result[:, 1]

# Plot each point in the PCA
sns.scatterplot(x="pca-one", y="pca-two", data=df, color="black")

# set the limits
plt.xlim([-1.5, 0.4])
plt.ylim([0.1, 1.4])

# Annotate the points within the specified region
subset_df = df[(df['pca-one'] >= -1.5) & (df['pca-one'] <= 0.4) & (df['pca-two'] >= 0.1) & (df['pca-two'] <= 1.4)]
for i, point in subset_df.iterrows():
    plt.text(point['pca-one'] + 0.01, point['pca-two'] + 0.01, str(point['risk_source']), fontsize=13)
plt.savefig('PCA_region_HobbiesSport.png')
plt.show()

# ---------------------DATASET: wide array risk sources (Study 2) -----------------------------------------------------

df = pd.DataFrame(vectors, columns=[f'feature_{i}' for i in range(len(vectors[0]))])
df['risk_source'] = tot_risk_name

plt.figure(figsize=(20, 20))
# Perform the PCA on the vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)
df["pca-one"] = pca_result[:, 0]
df["pca-two"] = pca_result[:, 1]

# Plot each point in the PCA
sns.scatterplot(x="pca-one", y="pca-two", data=df, color="black")

# set the limits
plt.xlim([-1.3, 0.3])
plt.ylim([-1.1, 0])

# Annotate the points within the specified region
subset_df = df[(df['pca-one'] >= -1.3) & (df['pca-one'] <= 0.3) & (df['pca-two'] >= -1.1) & (df['pca-two'] <= 0)]
for i, point in subset_df.iterrows():
    plt.text(point['pca-one'] + 0.01, point['pca-two'] + 0.01, str(point['risk_source']), fontsize=8)
plt.savefig('PCA_region_dataset2.png')
plt.show()


