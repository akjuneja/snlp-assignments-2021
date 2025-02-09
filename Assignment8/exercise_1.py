from collections import Counter
from pathlib import Path
import nltk
from nltk import RegexpTokenizer
nltk.download('reuters')
nltk.download('stopwords')
from nltk.corpus import reuters, stopwords

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import re
import math

def plot_category_frequencies(category_frequencies: Counter):
    plt.rcParams["figure.figsize"] = (30, 10)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("x-axis (Category)")
    plt.ylabel("y-axis (Absolute Frequency)")
    plt.title("Category v/s Absolute Frequency")

    x_axis = []
    y_axis = []
    n = 0
    for (word, freq) in category_frequencies.most_common():
        x_axis.append(word)
        y_axis.append(freq)
        n += 1

    plt.plot([x for x in range(1,n+1)], y_axis, label = "log-log plot")
    plt.xticks([x for x in range(1,n+1)], x_axis)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def plot_pmis(category: str, most_common: List[str], pmis: List[float]):
    plt.xlabel("x-axis (Words)")
    plt.ylabel("y-axis (pmi)")
    plt.title("Words v/s pmi")

    n = len(pmis)

    plt.plot([x for x in range(1,n+1)], pmis, label = category)
    plt.xticks([x for x in range(1,n+1)], most_common)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def plot_dfs(terms: List[str], dfs: List[int], category: str):
    plt.xlabel("x-axis (Word)")
    plt.ylabel("y-axis (Document Frequency)")
    plt.title("Category v/s Document Frequency")

    n = len(dfs)
    lbl = "category="+category
    plt.plot([x for x in range(1,n+1)], dfs, label = lbl)
    plt.xticks([x for x in range(1,n+1)], terms)
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        self.tokens = self.preprocess(text)

    def preprocess(self, text) -> List:
        '''
        params: text-text corpus
        return: tokens in the text
        '''
        text = text.lower()   #LOWERCASING
        tokens = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)     # Remove numbers
        tokens = re.sub(r'[^\w\s]', '', tokens)   #Removing the punctuations, special char
        tokens = re.sub('\s+', " ", tokens) ## remove \n\t ....
        tokens = tokens.split(' ')     #Tokenising

        filtered_tokens = [w for w in tokens if not w.lower() in self.stop_words]
 
        return filtered_tokens

class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = set(categories)

    def df(self, term: str, category=None) -> int:
        """
        :param category: None if df is calculated over all categories, else one of the reuters.categories
        """

        num_documents = 0
        tokens = []

        for document in self.documents:
            if category == None:
               tokens += document.tokens
               num_documents += 1
            elif category == document.category:
                tokens += document.tokens
                num_documents += 1

        tokens_count = Counter(tokens)
        df = tokens_count[term] / num_documents
        return df
        raise NotImplementedError

    def pmi(self, category: str, term: str) -> float:
        # A = co-occurence of term and category
        A = 0
        # B = times term occurs without category
        B = 0
        # C = times category occurs without term
        C = 0
        # D = num documents in C
        D = 0

        for document in self.documents:
            if document.category == category:
                D += 1
                if term in document.tokens:
                    A += 1
                else:
                    C += 1
            else:
                if term in document.tokens:
                    B += 1

        pmi = math.log((A * D)/((A+C)*(A+B)), 2)
        return pmi

        raise NotImplementedError
        
    def term_frequencies(self, category) -> Counter:
        documents = [document for document in self.documents if document.category == category]

        tokens = []
        for document in documents:
            tokens += document.tokens 

        return Counter(tokens)
        raise NotImplementedError

    def category_frequencies(self) -> Counter:
        return Counter([document.category for document in self.documents])
        raise NotImplementedError
