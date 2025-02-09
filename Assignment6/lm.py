from collections import Counter
from copy import deepcopy
import numpy as np
from typing import List, Tuple
import math


def get_n_grams_frequency(tokens: List[str], n: int):
    
    n_grams = get_n_grams(tokens, n)
    n_grams_freq = {}
    for sequence in n_grams:
            if sequence in n_grams_freq:
                n_grams_freq[sequence] += 1
            else:
                n_grams_freq[sequence] = 1

    return n_grams_freq

class LanguageModel:
    
    def __init__(self, train_tokens: List[str], test_tokens: List[str], N: int, alpha: float, epsilon=1.e-10):
        """ 
        :param train_tokens: list of tokens from the train section of your corpus
        :param test_tokens: list of tokens from the test section of your corpus
        :param N: n of the highest-order n-gram model
        :param alpha: pseudo count for lidstone smoothing
        :param epsilon: threshold for probability mass loss, defaults to 1.e-10
        """
        self.N = N
        self.alpha = alpha
        self.epsilon = epsilon

        self.train_counts = [Counter(self.get_n_grams(train_tokens, n)) for n in range(1, N+1)]
        #Vocabulary
        self.V = set(self.train_counts[0].keys())

    
    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        """ 
        :return: a list of n-mers
        """

        tokens_circular = tokens + tokens[:n-1]
        n_grams = []
        len_tokens = len(tokens_circular)
        for i in range(0, len_tokens-n+1):
            n_gram = tuple(tokens_circular[i:i+n])
            n_grams.append(n_gram)
  
        return n_grams


    def get_n_gram_count(self, n_gram: Tuple[str]) -> int:
        
        return self.train_counts[self.N-1][n_gram]


    def get_history_count(self, history: Tuple[str]) -> int:
        
        if len(history) == 0:
            return sum(self.train_counts[0].values())
        
        return self.train_counts[self.N-2][history]

    def perplexity(self, test_tokens: List[str]):
        """ returns the perplexity of the language model for n-grams with n=n """
        
        test_n_grams = self.get_n_grams(test_tokens, self.N)
        rel_freqs = {test_n_gram: count/len(test_n_grams) for test_n_gram, count in Counter(test_n_grams).items()}
        assert np.abs(1-sum(rel_freqs.values())) < self.epsilon, "Relative frequencies don't some up to 1!"
        H = -1 * sum([rel_freq * np.log2(self.lidstone_smoothing(test_n_gram)) for test_n_gram, rel_freq in rel_freqs.items()])
        return 2**H

        raise NotImplementedError


        


    def lidstone_smoothing(self, n_gram: Tuple[str]) -> float:
        """ applies lidstone smoothing on train counts

        :param alpha: the pseudo count ***change this***
        :return: the smoothed counts
        """

        history = n_gram[:self.N-1]
        return (self.get_n_gram_count(n_gram) + self.alpha) / (self.get_history_count(history) + self.alpha*len(self.V))

        raise NotImplementedError   
