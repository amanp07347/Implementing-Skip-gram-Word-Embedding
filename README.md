# Implementing-Skip-gram-Word-Embeddings-in-PyTorch
Overview:
This notebook demonstrates the process of creating a skip-gram word embedding model using PyTorch. The skip-gram model aims to predict the context words (neighbors) of a given target word from the corpus. After training, the model's weight matrix can be used as word embeddings.

# Components:
# Imports:
PyTorch for neural networks
NLTK for text preprocessing
Scikit-learn for SVD and cosine similarity
Pandas and matplotlib for data handling and visualization
# Preprocessing:

Remove stopwords using NLTK's list.
Convert "-ing" verbs to their base forms using regex.
Tokenize the corpus.
# Vocabulary Creation:

Create a vocabulary dictionary where each unique word in the corpus is assigned a unique integer ID.
# Neighbor Set Preparation:

For each word in the corpus, identify its neighboring words. The number of neighbors can be defined by the n_gram parameter.
# Model Definition:

A basic two-layer linear model with no biases. The hidden layer represents the embeddings.
Training:

The model is trained using CrossEntropyLoss.
Gradient descent is used to update the weights.
# Visualization:

Use TruncatedSVD from Scikit-learn to reduce the dimensionality of the embeddings to 2D.
Plot the 2D embeddings using seaborn.
# Cosine Similarity:

Check the similarity between two word embeddings in both the original 4D space and the 2D space after SVD.

# Conclusion:
This notebook provides a basic understanding of how to create and visualize skip-gram word embeddings in PyTorch.
