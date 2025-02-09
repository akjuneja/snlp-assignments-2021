import string
import re
from collections import Counter
from typing import List, Tuple


#TODO: Implement
def preprocess(text) -> List:
    '''
    params: text-text corpus
    return: tokens in the text
    '''
    text= text.lower()   #LOWERCASING
    tokens = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)     # Remove numbers
    tokens = re.sub(r'[^\w\s]', '', tokens)   #Removing the punctuations, special char
    tokens = re.sub('\s+', " ", tokens) ## remove \n\t ....
    tokens = tokens.split(' ')     #Tokenising
 
    return tokens

class KneserNey:
    def __init__(self, tokens: List[str], N: int, d: float):
        '''
        params: tokens - text corpus tokens
        N - highest order of the n-grams
        d - discounting paramater
        '''
        self.tokens = tokens
        self.N = N
        self.d = d

    def get_n_grams(self, tokens: List[str], n: int) -> List[Tuple[str]]:

        tokens_circular = tokens + tokens[:n-1]
        n_grams = []
        len_tokens = len(tokens_circular)
        for i in range(0, len_tokens-n+1):
            n_gram = tuple(tokens_circular[i:i+n])
            n_grams.append(n_gram)
  
        return n_grams
    
    def get_params(self, trigram) -> None:

        trigrams = trigram.split()

        trigram_tokens = self.get_n_grams(self.tokens, 3)
        bigram_tokens = self.get_n_grams(self.tokens, 2)
        unigram_tokens = self.get_n_grams(self.tokens, 1)

        trigram_tokens_count = Counter(trigram_tokens)
        bigram_tokens_count = Counter(bigram_tokens)
        unigram_tokens_count = Counter(unigram_tokens)

        # N+(∙∙)
        count_word_word = len(bigram_tokens_count)
        print(" N+(∙∙) = ", count_word_word)
        # **********************************

        # N+(∙w3)
        local_count = 0 
        for token in bigram_tokens_count.keys():
            if token[1] == trigrams[2]:     
                local_count += 1

        count_word_w3 = local_count
        print(" N+(∙w3) = ", count_word_w3)
        # **********************************
        
        # N+(∙w2∙)
        local_count = 0
        for token in trigram_tokens_count.keys():
            if token[1] == trigrams[1]:
                local_count += 1

        count_word_w2_word = local_count
        print(" N+(∙w2∙) = ", count_word_w2_word)
        # **********************************

        # N+(∙w2w3)
        local_count = 0
        for token in trigram_tokens_count.keys():
            if token[1] == trigrams[1] and token[2] == trigrams[2]: 
                local_count += 1

        count_word_w2w3 = local_count
        print(" N+(∙w2w3) = ", count_word_w2w3)
        # **********************************
        
        # N(w2)
        freq_w2 = unigram_tokens_count[(trigrams[1],)]
        print(" N(w2) = ", freq_w2)
        # **********************************

        # N+(w2∙) 
        local_count = 0
        for token in bigram_tokens_count:
            if token[0] == trigrams[1]:
                local_count += 1

        count_w2_word = local_count
        print(" N+(w2∙) = ", count_w2_word)
        # **********************************

        # N(w1w2)
        freq_w1w2 = bigram_tokens_count[(trigrams[0], trigrams[1])]
        print(" N(w1w2) = ", freq_w1w2)
        # ***********************************************************

        # N(w1w2w3)
        freq_w1w2w3 = trigram_tokens_count[(trigrams[0], trigrams[1], trigrams[2])]
        print(" N(w1w2w3) = ", freq_w1w2w3)
        # **************************************************************************
  
        # N+(w1w2∙)
        local_count = 0 
        for token in trigram_tokens:
            if token[0] == trigrams[0] and token[1] == trigrams[1]:
                local_count += 1

        count_w1w2_word = local_count
        print(" N+(w1w2∙) = ", count_w1w2_word)
        # **********************************

        # λ(w2)
        sigma_w2 = (self.d / freq_w2) * count_w2_word
        print(" λ(w2) = ", sigma_w2)
        # *********************************************

        # λ(w1w2)
        sigma_w1_w2 = (self.d / freq_w1w2) * count_w1w2_word
        print(" λ(w1w2) = ", sigma_w1_w2)
        # ***************************************************

        print(" \n Calculating the probabilities:")
        # PKN(w3)
        if trigrams[2] in unigram_tokens:
            prob_w3 = count_word_w3 / count_word_word
        else:
            prob_w3 = 1 / len(unigram_tokens_count)
        print(" PKN(w3) = ", prob_w3)
        # *********************************************

        # PKN(w3|w2)
        prob_w2_w3 = (max(count_word_w2w3 - self.d, 0) / count_word_w2_word) + (sigma_w2 * prob_w3)
        print(" PKN(w3|w2) = ", prob_w2_w3)
        # ************************************************************************************************

        # PKN(w3|w1,w2)
        prob_w1_w2_w3 = (max(freq_w1w2w3 - self.d, 0) / freq_w1w2) + (sigma_w1_w2 * prob_w2_w3)
        print(" PKN(w3|w1,w2) = ", prob_w1_w2_w3)
        # ********************************************************************************************
  