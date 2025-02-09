from collections import defaultdict
import math

from typing import Tuple
import numpy as np

class TrieNode(object):
    
    def __init__(self, char: str):
        self.char = char
        self.children = []
        # Is it the last character of the word.`
        self.word_finished = False
        # How many times this character appeared in the addition process
        self.counter = 1
    

def add(root, word: str):
    """
    Adding a word in the trie structure
    """
    node = root
    for char in word:
        found_in_child = False
        # Search for the character in the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found it, increase the counter by 1 to keep track that another
                # word has it as well
                child.counter += 1
                # And point the node to the child that contains this char
                node = child
                found_in_child = True
                break
        # We did not find it so add a new chlid
        if not found_in_child:
            new_node = TrieNode(char)
            node.children.append(new_node)
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a word.
    node.word_finished = True


def find_prefix(root, prefix: str) -> Tuple[bool, int]:
    """
    Check and return 
      1. If the prefix exsists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, 0
    for char in prefix:
        char_not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found the char existing in the child.
                char_not_found = False
                # Assign node as the child containing the char and break
                node = child
                break
            if (child.char == "$") :
                return True, child.counter
        # Return False anyway when we did not find a char.
        if char_not_found:
            return False, 0
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    return True, node.counter

def delete(root, k : int) :
    node = root
    count = 0
    del_child = []
    d_c = False

    for child in node.children :
        if child.counter < k : 
            del_child.append(child)
            count += child.counter
        else :
            delete(child, k)

    for child in del_child:
      node.children.remove(child)
      d_c = True

    if d_c :
      new_node = TrieNode("$")
      new_node.counter = count
      node.children.append(new_node)

def create_dic(ngrams) :
  f_dict = {}
  for ngram in ngrams : 
    text = ""
    for t in ngram :
      text += t + ","

    if text in f_dict :
      f_dict[text] += 1
    else :
      f_dict[text] = 1
  
  return f_dict 

class CountTree():
  def __init__(self, n=4):
    self.root = TrieNode("*")
  
  def add(self, ngram):   # add from last entries
    add(self.root, ngram[::-1])

  def get(self, ngram):
    val = find_prefix(self.root, ngram[::-1])
    return val[1]
    
  def perplexity(self, ngrams, vocab):
    ## for zerogram p = 1 / |vocab|
    ## assuming test data is ngrams and existing tree is trained model for 
    ## getting conditional probab
    cp0 = 1/len(vocab)
    freq_dic = create_dic(ngrams)
    siz = len(ngrams)

    sum = 0
    for ngram in ngrams : 
      text = ""
      for t in ngram :
        text += t + ","
      
      fq = freq_dic[text]/siz

      c1 = self.get(ngram)
      c2 = self.get(ngram[:-1])
      if c1 == 0 or c2 == 0 : 
        cp4 = 0
      else :
        cp4 = np.log(c1/c2)
      smoothed_cp4 = 0.75 * cp4 + 0.25 * cp0
      sum += fq * smoothed_cp4

    return np.power(2, -1*sum)
    
  def prune(self, k):
    delete(self.root, k)




