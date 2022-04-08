import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Import the required libraries
import gensim
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

# Load the word2vec embedding
print("before work_vectors")
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
print("after work_vectors")
sent1 = "I like to eat mangoes"
sent2 = "I eat bananas"

# Create the vector representation for sent1
sent_v1 = np.zeros((300,))
count = 0
for word in sent1.split():
  if word in word_vectors:
    count += 1
    sent_v1 += word_vectors[word]

# Averaging the vector to obtain the sentence embedding
final_vector_1 = sent_v1 / count

# Create the vector representation for sent2
sent_v2 = np.zeros((300,))
count = 0
for word in sent2.split():
  if word in word_vectors:
    count += 1
    sent_v2 += word_vectors[word]

# Averaging the vector to obtain the sentence embedding
final_vector_2 = sent_v2 / count

# Calculate the similarity between the two sentences
print(cosine_similarity([final_vector_1],[final_vector_2]))