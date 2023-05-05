""" https://radimrehurek.com/gensim/models/word2vec.html
website for the below code and explanation"""

""" Core tutorial and documentation:
https://radimrehurek.com/gensim/auto_examples/
ti da le basi su come usare gensim e come funziona (tra i vari modelli) word2vec"""

from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")

""" Embeddings with multiword ngrams:
There is a gensim.models.phrases module which lets you automatically detect phrases longer than one word, using 
collocation statistics. Using phrases, you can learn a word2vec model where “words” are actually multiword expressions, 
such as new_york_times or financial_crisis:"""

from gensim.models import Phrases

# Train a bigram detector.
bigram_transformer = Phrases(common_texts)

# Apply the trained MWE detector to a corpus, using the result to train a Word2vec model.
model = Word2Vec(bigram_transformer[common_texts], min_count=1)
# ---------------------------------------------------------------------------------------------------------------------


""" Pretrained models:
Gensim comes with several already pre-trained models, in the Gensim-data repository:"""

import gensim.downloader as api
# Show all available models in gensim-data
print(list(api.info()['models'].keys()))

""" output of the above code:
['fasttext-wiki-news-subwords-300', 'conceptnet-numberbatch-17-06-300', 'word2vec-ruscorpora-300', 
'word2vec-google-news-300', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-wiki-gigaword-200', 
'glove-wiki-gigaword-300', 'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200', 
'__testing_word2vec-matrix-synopsis']"""

# Download the "word2vec-google-news-300" embeddings
word2vec_googlenews_300 = api.load('word2vec-google-news-300')

""" terms of use of the model word2vec-google-news-300:
You can use Google-trained word embeddings in your own research or applications, under the CC BY-NC-SA 3.0 license. 
Note that the word embeddings are licensed for non-commercial use only. If you use the embeddings, please cite the 
original paper.
--->
Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. 
Proceedings of the Workshop at the International Conference on Learning Representations (ICLR).
"""
word2vec_googlenews_300.most_similar('cat', topn=5)
# ---------------------------------------------------------------------------------------------------------------------

""" WORD2VEC DEMO:
To see what Word2Vec can do, let’s download a pre-trained model and play around with it. We will fetch the Word2Vec
 model trained on part of the Google News dataset, covering approximately 3 million words and phrases. Such a model can 
 take hours to train, but since it’s already available, downloading and loading it with Gensim takes minutes."""

# Download the "word2vec-google-news-300" embeddings (already did it line 41 as word2vec_googlenews_300)

"""A common operation is to retrieve the vocabulary of a model. That is trivial:"""

for index, word in enumerate(word2vec_googlenews_300.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(word2vec_googlenews_300.index_to_key)} is {word}")

# output of the above code:
# word #0/3000000 is </s>
# word #1/3000000 is in
# word #2/3000000 is for
# word #3/3000000 is that
# word #4/3000000 is is
# word #5/3000000 is on
# word #6/3000000 is ##
# word #7/3000000 is The
# word #8/3000000 is with
# word #9/3000000 is said

""" We can easily obtain vectors for terms the model is familiar with: """

vec_king = word2vec_googlenews_300['king']

"""When you call word2vec_googlenews_300['king'], you're retrieving the vector representation of the word "king" from 
the word2vec_googlenews_300 model.
*Each element* in the list of numbers that you obtained represents the value of *one of the 300 dimensions* in the  
vector space. This vector is a high-dimensional representation of the word "king" that captures some of its semantic and 
syntactic properties.

The specific numbers in the list don't have any meaningful interpretation on their own, but they can be used to perform 
operations on words, such as finding words with similar meanings or analogies. For example, you could find the words 
most similar to "king" in the vector space using the most_similar method in gensim:"""

king_similar_words = word2vec_googlenews_300.most_similar('king', topn=10)
print(king_similar_words)
# [('kings', 0.7138046622276306), ('queen', 0.6510956287384033), ('monarch', 0.6413194537162781),
# ('crown_prince', 0.6204220056533813), ('prince', 0.6159993410110474), ('sultan', 0.5864824056625366),
# ('ruler', 0.5797566771507263), ('princes', 0.5646552443504333), ('Prince_Paras', 0.5432944297790527),
# ('throne', 0.5422104597091675)]

"""Unfortunately, the model is unable to infer vectors for unfamiliar words. This is one limitation of Word2Vec: if 
this limitation matters to you, check out the FastText model."""

try:
    vec_cameroon = word2vec_googlenews_300['clothes dryer']
except KeyError:
    print("The word 'cameroon' does not appear in this model")

# output of the above code: The word 'cameroon' does not appear in this model

"""Moving on, Word2Vec supports several word similarity tasks out of the box. You can see how the similarity intuitively 
decreases as the words get less and less similar."""

pairs = [('car', 'minivan'), ('car', 'bicycle'), ('car', 'airplane'), ('car', 'cereal'), ('car', 'communism')]

for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, word2vec_googlenews_300.similarity(w1, w2)))

# output of the above code:
# 'car'	'minivan'	0.69
# 'car'	'bicycle'	0.54
# 'car'	'airplane'	0.42
# 'car'	'cereal'	0.14
# 'car'	'communism'	0.06

""""in the print function the expressions [ '%r\t%r\t%.2f' % ] means:
This is a Python string formatting expression using the % operator, which allows for inserting values into a string. The
values to be inserted are specified after the string, and are passed as arguments to the % operator.

In this particular expression, the %r and %.2f are "format specifiers" that specify the TYPE and PRECISION of the values 
to be inserted into the string. The \t character represents a TAB character that separates the different values to be 
inserted. 
- %r is a format specifier that indicates that a value should be inserted into the string 
- %.2f is a format specifier that indicates that a floating-point number should be inserted into the string with 
2 digits after the decimal point.
"""

"""Print the 5 most similar words to “car” or “minivan”:"""

print(word2vec_googlenews_300.most_similar(positive=['car', 'minivan'], topn=5))

# [('SUV', 0.853219211101532), ('vehicle', 0.8175785541534424), ('pickup_truck', 0.7763689160346985),
# ('Jeep', 0.7567334175109863), ('Ford_Explorer', 0.756571888923645)]

"""the most_similar() function in Gensim's KeyedVectors class has both positive and negative parameters.
The positive parameter takes a list of words that are treated as having a positive association with the target word, 
while the negative parameter takes a list of words that are treated as having a negative association with the target 
word. This allows you to perform analogical reasoning tasks using the famous "king - man + woman = queen" example.

Here's an example of how to use the positive and negative parameters in the most_similar() function:"""

print(word2vec_googlenews_300.most_similar(positive=['woman', 'king'], negative=['man'], topn=5))

"""This will return a list of the five words that are most similar to the vector representation of the context of 
'woman' and 'king', but have a negative association with the context of 'man'."""

# [('queen', 0.7118193507194519), ('monarch', 0.6189674735069275), ('princess', 0.5902431011199951),
# ('crown_prince', 0.5499460697174072), ('prince', 0.5377322435379028)]

# --------------------------------------------------------------------------------
"""DIFFERENCE BETWEEN WORD2VEC OF GENSIM AND WORD2VEC OF GOOGLE NEWS DATASET"""

"""" the pre-trained word2vec model provided by Gensim is not trained on the full Google News dataset. As you mentioned, 
the Gensim model is trained on a vocabulary of around 3 million words and phrases, while the full Google News dataset 
has about 100 billion words.
The pre-trained word2vec model provided by Gensim is a smaller, more manageable version of the full Google News dataset, 
which can still be useful for many natural language processing tasks. However, if you need to work with the full Google
News dataset, you may need to train your own word2vec model or use a different pre-trained model that was trained on the 
full dataset.

There are some pre-trained models available that were trained on larger datasets than the Gensim model, such as the 
FastText pre-trained models provided by Facebook Research. These models were trained on larger datasets and may provide 
better performance for some tasks. However, they may also be larger and more difficult to work with than the 
Gensim model."""