import pandas as pd
import numpy as np

df_vocab = pd.read_csv('./flickr8k/annotations/annotations_image_id.csv', sep=';')

# List of all the captions of the dataset
raw_sentences = list(df_vocab.iloc[:, 1])

# String concatenating all the sentences
raw_text = raw_sentences[0]

# Build raw_text
for i in range(1, len(raw_sentences)):
    raw_text += ' ' + raw_sentences[i]

# Add special words start and stop
raw_text += ' <start> <stop>'

# Split text into words
raw_text = raw_text.split()

# Get vocabulary
vocab = np.array(raw_text)
vocab = np.unique(vocab) # start is at index 67 and stop is at index 68

ohe = np.identity(vocab.shape[0])

def same_word(word1,word2):
    bool_arr = (word1 == word2)
    for b in bool_arr:
        if not b:
            return False
    return True

def word_to_vect(word):
    word_ind = np.searchsorted(vocab, word)
    return ohe[word_ind]

def caption_to_vect(caption):
    '''
    Parameters : 
        caption : a string of a caption, starting with <start> and ending with <stop>
    Output :
        a vector of shape (9631,nb_of_words) representing the caption
    '''
    c_list = caption.split()
    c_list = np.array(c_list)
    c_vect = np.zeros((len(c_list),vocab.shape[0]))
    for k in range(len(c_list)):
        print(word_to_vect(c_list[k]))
        c_vect[k] = np.array(word_to_vect(c_list[k]))
    return c_vect

def vect_to_caption(vect):
    '''
    Parameters : 
        vect : a np array of shape (9631,nb_of_words) that represents a caption starting with <start> ending with <stop>
    Output :
        a string caption
    '''
    caption = ""
    started = same_word(vect[0],word_to_vect('<start>'))
    if not started:
        raise ValueError
    for k in range(1,vect.shape[0]-1):
        wordx = np.argmax(vect[k])
        caption += vocab[wordx] + ' '
    stopped = same_word(vect[-1],word_to_vect('<stop>'))
    if not stopped:
        raise ValueError
    return caption
        
#wordtest = np.zeros((6,vocab.shape[0]))
#wordtest[0][67] = 1#<start>
#wordtest[1][310] = 1
#wordtest[2][4857] = 1
#wordtest[3][240] = 1
#wordtest[4][4687] = 1
#wordtest[5][68] = 1 #<stop>
#print(vect_to_caption(wordtest))
#print(word_to_vect('Canada'))
#vecttest = '<start> Crowd hurdle Canada herding <stop>'
#print(np.where((wordtest == caption_to_vect(vecttest)) == False))
print(vocab.shape[0])