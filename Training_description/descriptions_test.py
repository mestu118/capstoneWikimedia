import json
import fasttext
import pandas as pd
import numpy as np
import re
import os
import unicodedata
from tqdm import tqdm
from numpy.core.umath_tests import inner1d
from sklearn.model_selection import train_test_split
import sys 

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
    
    len_bd = len(bilingual_dictionary)

    for i, (source, target) in tqdm(enumerate(bilingual_dictionary)):
#         print(f'\r{i + 1}/{len_bd} | {100 * (i + 1) / len_bd:.3f} %', end = '', flush = True)
        sourceVector = source_dictionary.get_sentence_vector(source.lower().strip().replace('_',' '))
        targetVector = target_dictionary.get_sentence_vector(target.lower().strip().replace('_',' '))
        source_matrix.append(sourceVector)
        target_matrix.append(targetVector)
        
    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
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
    product = np.matmul(target_matrix.transpose(), source_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

def cleanData(data):
    """
    cleans the sentences in the data given. Possible HTML tags still in the sentences 
    """
    
    data_cleaned = []
    for source, target in zip(data['source_value'], data['target_value']):
        try:
            source_cleaned = unicodedata.normalize('NFD', source)
            target_cleaned = unicodedata.normalize('NFD', target)
            if not source == ''  and not target == '':
                data_cleaned.append([source_cleaned, target_cleaned])
        except:
            pass
    return data_cleaned

def joinData(path, source, target):
    """
    Function loads source.csv and target.csv and creates source2target.csv 
    
    """
    if check(source, target):
        return joinData(target, source)
    
    
    try:
        df1 = pd.read_csv(os.path.join(path, 'allLanguages', '{}.csv'.format(target)), error_bad_lines = False)
    except:
        raise Exception("{}.csv is not in the directory allLanguages".format(target))
        
    try:
        df2 = pd.read_csv(os.path.join(path, 'allLanguages', '{}.csv'.format(source)), error_bad_lines = False)
    except:
        raise Exception("{}.csv is not in the directory allLanguages".format(source))
        
        
    df1.columns = ['ID', 'source_lang', 'source_value']
    df2.columns = ['ID', 'target_lang', 'target_value']
    
    retVal = df1.join(df2.set_index('ID'), on = 'ID')
    
    retVal.to_csv(os.path.join(path, 'description_files', 'pair_sentences', '{}2{}.csv'.format(source, target)))
    
    
def check(source, target):
    """
    Function to check if souce and target are in sorted order 
    """
    
    val = [source, target]
    return val != sorted(val)

def loadData(source, target, recursed = False):
    """
    Function loads the file description_train_source2target.csv
    
    If file is not found, it create it by loading source2target.csv and cleans data and creates the file
    and saves it.
    
    If source2target.csv is not found. Loads source.csv and target.csv and creates and creates the file
    source2target.csv and recurses (recursed = True to avoid infinite recursion).
    """
    
    if check(source, target):
        return loadData(target, source)
    
    file = '{}2{}.csv'.format(source, target)
    
    path = os.getcwd()
    
    description_path = os.path.join(os.getcwd(), 'description_files')
    
    train_files = os.listdir(os.path.join(description_path, 'train_files'))
    
    pair_sentences = os.listdir(os.path.join(description_path, 'pair_sentences'))
    
    retVal = 'description_train_{}'.format(file)
    
    if retVal in train_files:
        return pd.read_csv(os.path.join(description_path, 'train_files', retVal))
    
    elif file in pair_sentences:
        print("Loading from {}".format(os.path.join(description_path, 'pair_sentences', file)))
        
        data = pd.read_csv(os.path.join(description_path, 'pair_sentences', file), error_bad_lines = False)
        
        data = data.drop(['ID'], axis = 1)
        data = data.drop_duplicates()
        df = pd.DataFrame(cleanData(data), columns = ['source', 'target'])
        print("Saving at {}".format(os.path.join(description_path, 'train_files', retVal)))
        
        df.to_csv(os.path.join(description_path, 'train_files', retVal))
        return pd.read_csv(os.path.join(description_path, 'train_files', retVal))
    
    else:
        if recursed:
            raise Exception("Entered infinite recursion. Check directories in function")
        else:
            joinData(path, source, target)
            return loadData(source, target, True)
        
def runTests(source, target):
    
    if check(source, target):
        return runTests(target, source)
    
    test = pd.read_csv(os.path.join('/scratch', 'ah3243', 'content_test.csv'))
    train = loadData(source, target)
    
    model = {}
    path = os.path.join('/scratch', 'dev241', 'capstone', 'fast')
    
    model[source] = fasttext.load_model(os.path.join(path, 'wiki.{}.bin'.format(source)))
    model[target] = fasttext.load_model(os.path.join(path, 'wiki.{}.bin'.format(target)))
    
    bilingual_dictionary = list(zip(train['source'],train['target']))
    
    source_matrix, target_matrix = make_training_matrices(model[source], model[target], bilingual_dictionary)
    
    transform = learn_transformation(source_matrix, target_matrix)
    
    print("Before trans:", np.mean(inner1d(target_matrix, source_matrix)))
    
    print("After trans:", np.mean(inner1d(normalized(target_matrix), np.matmul(transform, normalized(source_matrix).T).T)))
    
    bilingual_dictionary = list(zip(test['source'],test['target']))

    source_matrix_test, target_matrix_test = make_training_matrices(model[source], model[target], bilingual_dictionary)
    
    
    target_matrix_test = normalized(target_matrix_test)
    source_matrix_test = normalized(source_matrix_test)
    
    print("Before trans:",np.mean(inner1d(target_matrix_test, source_matrix_test)))
    #after
    print("After trans:", np.mean(inner1d(target_matrix_test, np.matmul(transform, source_matrix_test.T).T)))
    
    
if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage descriptions_test.py source_lang target_lang")
    else:
        runTests(sys.argv[1], sys.argv[2])
