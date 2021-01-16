import pandas as pd
import numpy as np
import torch

class TextPreprocessor:
    
    def __init__(self, annotation_file, sep):
        
        self.init_vocab(annotation_file, sep)
        
        
    def init_vocab(self, annotation_file, sep):

        df_vocab = pd.read_csv(annotation_file, sep=sep)

        # List of all the captions of the dataset
        raw_sentences = list(df_vocab.iloc[:, 1])
        
        self.raw_sentences = raw_sentences

        # String concatenating all the sentences
        self.raw_text = raw_sentences[0]
        
        # get the max length of the sentences of the dataset +2 for start and stop words
        self.max_len = max(list(map(lambda x: len(x.split()), self.raw_sentences))) + 2
        #print(self.max_len)

        # Build raw_text
        for i in range(1, len(raw_sentences)):
            self.raw_text += ' ' + raw_sentences[i]

        # Add special words start and stop
        self.raw_text += ' <start> <stop>'

        # Split text into words
        self.raw_text = self.raw_text.split()

        self.occurences = {}
        
        #print("lenght of raw_text : " + str(len(raw_text)))
        
        l = len(self.raw_text)
        c = 0
        for w in self.raw_text:
            #if c % 10000 == 0:
                #print(str(c) + " / " + str(l))
            c+=1
            if not w in self.occurences:
                self.occurences[w] = self.raw_text.count(w)
        
        # Get vocabulary
        vocab = np.array(self.raw_text)
        self.vocab = np.unique(vocab) 
        self.vocab_size = len(self.vocab)
        
        # Build words encoding matrix
        self.encoding_matrix = np.identity(len(self.vocab))
        
        # Keep in memory start and stop vectors
        self.start = self.encoding_matrix[np.where(self.vocab == '<start>')].flatten()
        self.stop = self.encoding_matrix[np.where(self.vocab == '<stop>')].flatten()


    def word_to_vect(self, word):
        
        assert word in self.vocab
        
        word_idx = np.searchsorted(self.vocab, word)
        return self.encoding_matrix[word_idx]
    

    def caption_to_vect(self, caption):
        '''
        Parameters : 
            caption : a string of a caption, starting with <start> and ending with <stop>
        Output :
            a vector of shape (nb_of_words, vocab_size) representing the caption
        '''

        words = np.array(caption.split())
        vects = []
        
        '''for w in words:
            if self.occurences[w] >= 5:
                vects.append(self.word_to_vect(w))'''
        
        vects = np.asarray(list(map(lambda x: self.word_to_vect(x), words)))
        
        #vects = np.asarray(vects)
        
        return vects

    
    def vect_to_word(self, vect):
        word_idx = np.argmax(vect)
        return self.vocab[word_idx]
    
    
    def vect_to_caption(self, vects):
        '''
        Parameters : 
            vect : a np array of shape (nb_of_words, vocab_size) that represents a caption starting with <start> ending with <stop>
        Output :
            a string caption (without <start> and <stop> symbols)
        '''
   
        if not np.equal(vects[0], self.start).all() or not np.equal(vects[-1], self.stop).all():
            raise ValueError
            
        return " ".join(list(map(lambda x: self.vect_to_word(x), vects)))



    def target_from_vect(self, words):
        
        if len(words) == 1:
            return torch.tensor([torch.argmax(words)])
        
        return torch.tensor([torch.argmax(w) for w in words])
    
    
    def word_to_idx(self, word):
        
        assert word in self.vocab
        
        return np.searchsorted(self.vocab, word)
    
    def pad_sentence(self, sentence):
        
        len_sentence = len(sentence.split())
        for i in range(self.max_len - len_sentence):
            sentence += ' <stop>'
            
        return sentence


        
        
    
