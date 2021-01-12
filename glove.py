import pandas as pd
import pickle
import os
from text_preprocessing import TextPreprocessor
import numpy as np
import torch

class Glove:
    
    def __init__(self, directory='./', dimension=300):
        
        assert dimension in [50, 100, 200, 300]
        
        self.dim = dimension
        
        if os.path.exists('./glove_dict'):
            with open('glove_dict', 'rb') as f:
                self.embedding_dict = pickle.load(f)
        else:
            self.get_glove(directory)
            
        self.create_embedding_matrix()
            
        
        
        
    def get_glove(self, directory):
        
        glove = pd.read_csv(directory + 'glove.6B.' + str(self.dim) +'d.txt', sep=' ', quoting=3, header=None, index_col=0)
        self.embedding_dict = {key: val.values for key, val in glove.T.items()}
        
        
        with open('glove_dict', 'wb') as f:
            pickle.dump(self.embedding_dict, f)
            
            
    def create_embedding_matrix(self):
        self.tp = TextPreprocessor('./flickr8k/annotations/annotations_image_id.csv', sep=';')            
        self.vocab = self.tp.vocab
        
        matrix = np.zeros((self.tp.vocab_size, self.dim))
        
        for index, word in enumerate(self.vocab):
            if word in self.embedding_dict:
                matrix[index] = self.embedding_dict[word]
                
        self.embedding_matrix = matrix
        
        
    def get_embedding_vector(self, word_ohe):
                
        #return self.embedding_matrix[torch.argmax(word_ohe)]
        return self.embedding_matrix[np.argmax(word_ohe)]
    
        
if __name__ == "__main__"; 
    g = Glove()
    man = g.tp.word_to_vect("man")
    emb_man = g.get_embedding_vector(man)

    assert (emb_man == g.embedding_dict["man"]).all()
        