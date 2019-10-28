pip install fasttext

import os
import pandas as pd
import json
import fasttext as fastText
from scipy.spatial import distance
import numpy as np
import networkx as nx
import re

def apply_transform(vec, transform):
        """
        Apply the given transformation to the vector space

        Right-multiplies given transform with embeddings E:
            E = E * transform

        Transform can either be a string with a filename to a
        text file containing a ndarray (compat. with np.loadtxt)
        or a numpy ndarray.
        """
        transmat = np.loadtxt(transform)# if isinstance(transform, str) else transform
        return np.matmul(vec, transmat)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        try:
            source = source.lower().split()
            sourceVector = np.zeros(300) + sum([source_dictionary[word] for word in source  if word in source_dictionary])/len(source)
            target = target.lower().split()
            targetVector = np.zeros(300) + sum([target_dictionary[word] for word in target  if word in target_dictionary])/len(target)
            if (sourceVector.all() !=0) and (targetVector.all() != 0):
                    source_matrix.append(sourceVector)
                    target_matrix.append(targetVector)
        except:
            pass
    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def apply_transform(vec, transform):
        """
        Apply the given transformation to the vector space

        Right-multiplies given transform with embeddings E:
            E = E * transform

        Transform can either be a string with a filename to a
        text file containing a ndarray (compat. with np.loadtxt)
        or a numpy ndarray.
        """
        transmat = np.loadtxt(transform)# if isinstance(transform, str) else transform
        return np.matmul(vec, transmat)



model = {}

# Here you want to replace /content/drive/My Drive/Capstone-Wiki/ with your folder
# The FastText files are going to be in a FastText folder after you run the other code that unzips it

model['en'] = fastText.load_model(os.path.abspath('/content/drive/My Drive/Capstone-Wiki/FastText/wiki.en.bin'))
model['es'] = fastText.load_model(os.path.abspath('/content/drive/My Drive/Capstone-Wiki/FastText/wiki.es.bin'))


filename = # Here just give it your descriptions file name
df = pd.read_csv(filename)
bilingual_dictionary = list(zip(df['target_content'],df['source_content']))
source_matrix, target_matrix = make_training_matrices(model['es'], model['en'], bilingual_dictionary)
transform = learn_transformation(source_matrix, target_matrix)
np.savetxt('descriptions_transform_en_es.txt', transform, fmt='%.2e')