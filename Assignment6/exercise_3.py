from typing import List
import matplotlib.pyplot as plt

special_characters = ['!','"','#','$','%','&','(',')','*','+','/',':',';','<','=','>','@','[','\\',']','^','`','{','|','}','~','\t','?','.', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 

def preprocess(text) -> List:
    text= text.lower()
    tokens = []
    text = text.replace('\n', ' ')
    for i in special_characters:
        text = text.replace(i, '')
    
    tokens = text.split()
    return tokens

def train_test_split_data(text: List[str], test_size=0.2):
    """ Splits the input corpus in a train and a test set
    :param text: input corpus
    :param test_size: size of the test set, in fractions of the original corpus
    :return: train and test set
    """

    total_size = len(text)
    train_size = 1 - test_size
    train_set = text[: int(train_size * total_size)]
    test_set = text[ int(train_size * total_size):]

    return train_set, test_set

def k_validation_folds(text: List[str], k_folds=10):
    """ Splits a corpus into k_folds cross-validation folds
    :param text: input corpus
    :param k_folds: number of cross-validation folds
    :return: the cross-validation folds
    """
    cv_folds = {}
    size = len(text)
    portion = int((1/k_folds)*size)
    
    for i in range(k_folds) :
      st = i*portion
      end = (i+1)*portion
      test = text[st:end]

      tr1 = text[0:st]
      tr2 = text[end:]
      train = tr1 + tr2

      cv_folds[i] = (train, test)

    return cv_folds

def plot_pp_vs_alpha(pps: List[float], alphas: List[float], lbl: str):
    """ Plots n-gram perplexity vs alpha
    :param pps: list of perplexity scores
    :param alphas: list of alphas
    """
    plt.figure()
    names = alphas
    freq = pps
    plt.plot([x for x in names],[x for x in freq] , '.-'
      ,label=lbl)
    plt.xlabel('alpha')
    plt.ylabel('perplexity')
    plt.legend()
    plt.show()