from collections import Counter

from typing import Dict, List

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import re
import math

class Document:
    def __init__(self, id:str, text: str, categories: str, stop_words: List[str]):
        self.id = id
        # Determines wheter the document belongs to the train and test set
        self.section = id.split("/")[0]
        # assume only 1 category per document for simplicity
        self.category = categories[0]
        self.stop_words = stop_words
        # TODO: tokenize!
        text = text.lower()   #LOWERCASING
        tokens = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)     # Remove numbers
        tokens = re.sub(r'[^\w\s]', '', tokens)   #Removing the punctuations, special char
        tokens = re.sub('\s+', " ", tokens) ## remove \n\t ....
        tokens = tokens.split(' ')     #Tokenising
 
        # TODO: remove stopwords!
        filtered_tokens = [w for w in tokens if not w.lower() in self.stop_words]
        # TODO: lemmatize
        wnl = WordNetLemmatizer()
        lemmatized = [wnl.lemmatize(w) for w in filtered_tokens]
        # count terms
        self._term_counts = Counter(lemmatized)

    def f(self, term: str) -> int:
        """ returns the frequency of a term in the document """
        return self.term_frequencies[term]

    @property
    def term_frequencies(self):
        return self._term_counts

    @property
    def terms(self):
        return set(self._term_counts.keys())


class Corpus:
    def __init__(self, documents: List[Document], categories: List[str]):
        self.documents = documents
        self.categories = sorted(list(set(categories)))

    def __len__(self):
        return len(self.documents)

    def _tf_idfs(self, document:Document, features:List[str], idfs: Dict[str, float]) -> List[float]:
        freq_reduced_words_in_doc = 0

        tf_idfs = []
        total_terms =  len(document.term_frequencies)   
        for feature in features:
            tf = document.term_frequencies[feature] / total_terms
            tf_idf = tf * idfs[feature]
            tf_idfs.append(tf_idf)

        return tf_idfs            
        raise NotImplementedError

    def _idfs(self, features: List[str]) -> Dict[str, float]:
        idfs = {}
        total_docs = len(self.documents)
        term_to_doc_count = Counter()
        for doc in self.documents:
            term_to_doc_count.update(set(doc.terms))

        for term, count in term_to_doc_count.items():
            idf = math.log((total_docs / count), 2)
            idfs[term] = idf
        return idfs
        raise NotImplementedError

    def _category2index(self, category:str) -> int:
        return self.categories.index(category)
        raise NotImplementedError

    def reduce_vocab(self, min_df: int, max_df: float) -> List[str]:
        dfs = {}
        total_docs = len(self.documents)
        term_to_doc_count = Counter()
        for doc in self.documents:
            term_to_doc_count.update(set(doc.terms))

        for term, count in term_to_doc_count.items():
            df = count / total_docs
            if df > min_df and df < max_df:
                dfs[term] = df

        print(dfs)
        return list(dfs.keys())
        raise NotImplementedError

    def compile_dataset(self, reduced_vocab: List[str], idfs: Dict[str, float]):
        train_idfs = []
        train_labels = []

        test_idfs = []
        test_labels = []

        for doc in self.documents:
            if doc.section == "training":
                train_idfs.append(self._tf_idfs(doc, reduced_vocab, idfs))
                train_labels.append(self._category2index(doc.category))
            elif doc.section == "test":
                test_idfs.append(self._tf_idfs(doc, reduced_vocab, idfs))
                test_labels.append(self._category2index(doc.category))

        return (train_idfs, train_labels), (test_idfs, test_labels)
        raise NotImplementedError

    def category_frequencies(self):
        return Counter([document.category for document in self.documents])

    def terms(self):
        terms = set()
        for document in self.documents:
            terms.update(document.terms)
        return sorted(list(terms))

