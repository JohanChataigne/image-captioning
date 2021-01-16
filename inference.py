import torch
import copy

# Generate caption for given image
def sampling(cnn, embedding, rnn, image, tp, embedding_size, hidden_size, max_length=20):
          
    with torch.no_grad():

        caption = list()

        # Encode input image
        image_embedding = cnn(image).view(-1, 1, embedding_size).cuda()

        # Get first word prediction probabilities
        (hn, cn), probs = rnn(image_embedding)

        
        
        # Extract predicted word
        pred_idx = torch.argmax(probs)
        print(pred_idx)
        pred_word_vect = tp.encoding_matrix[pred_idx]
        predicted_word = tp.vect_to_word(pred_word_vect)

        caption.append(predicted_word)
        
        i = 0

        # Build caption until model outputs stop word
        while predicted_word != '<stop>' and i < max_length:

            word_embedding = embedding(pred_idx).view(1, 1, embedding_size).cuda()

            (hn, cn), probs = rnn(word_embedding, (hn, cn))


            pred_idx = torch.argmax(probs)
            print(pred_idx)
            pred_word_vect = tp.encoding_matrix[pred_idx]
            predicted_word = tp.vect_to_word(pred_word_vect)

            caption.append(predicted_word)
            
            i+=1


        caption = " ".join(caption)
        
        return caption


# Generate caption for the given image with a beam search approach
def beam_search(cnn, embedding, rnn, image, tp, embedding_size, hidden_size, beam_k=20, max_length=20):
    with torch.no_grad():
        
        captions = [[] for i in range(beam_k)]
        captions_over = []
        
        # get the best_k_words for the first iteration : [word0 = (index, (hn, cn)), word1 = ...]
        best_k_words = generate_first_k_words(cnn, embedding, rnn, image, tp, embedding_size, hidden_size, beam_k)
        
        # initialize each caption with a word
        for i in range(beam_k):
            captions[i].append(best_k_words[i][0])
        
        while not len(captions_over) == beam_k:
            
            # get the best_k_words for the next iteration : [word0 = (index, (hn, cn)), word1 = ...]
            best_k_words, captions = generate_next_k_words(best_k_words, captions, cnn, embedding, rnn, tp, embedding_size, hidden_size)
            
            captions, captions_over = update_captions(captions, captions_over, max_length)
            
        
        return list(map(lambda c: " ".join(list(map(lambda x: tp.vect_to_word(tp.encoding_matrix[int(x)]), c))), captions_over))

                           
# update captions by removing the finished captions and adding them to captions_over
def update_captions(captions, captions_over, max_length):
    
    for i, caption in enumerate(captions):
        if len(caption) >= max_length or caption[-1] == '<stop>':
            captions_over.append(captions.pop(i))
            
    return captions, captions_over
    
    
# Generate the first best k words predicted by the rnn from the image
# Return: best_k_words = [(index, (hn, cn)), ...]
def generate_first_k_words(cnn, embedding, rnn, image, tp, embedding_size, hidden_size, beam_k):
    
    # Encode input image
    image_embedding = cnn(image).view(-1, 1, embedding_size).cuda()

    # Get first word prediction probabilities
    (hn, cn), probs = rnn(image_embedding)

    # best_k_words = [(index, (hn, cn)), ...]
    best_k_words = []

    # get the indices of the best k probabilities of word from probs
    top_k_indices = torch.topk(probs, beam_k).indices

    # append the tuple (index, (hn, cn)) to best_k_words for each word 
    for i in top_k_indices[0]:

        #word_vect = tp.encoding_matrix[index]
        #word = tp.vect_to_word(word_vect)
        best_k_words.append([i, (hn, cn)])
        
    return best_k_words
        
        
# generate the next best k words predicted by the rnn from the previous best k words
# Return: new_best_k_words = [(index, (hn, cn)), ...], new_captions
def generate_next_k_words(previous_best_k_words, captions, cnn, embedding, rnn, tp, embedding_size, hidden_size):
    
    all_probs = torch.empty(0).cuda()      
    new_best_k_words = []   

    # get k lists of probabilities, one for each word of previous_best_k_words
    for i in range(len(captions)):

        index = previous_best_k_words[i][0]
        word_embedding = embedding(index).view(1, 1, embedding_size).cuda()

        (hn, cn), probs = rnn(word_embedding, previous_best_k_words[i][1])
        
        # concatenate all the probs lists
        all_probs = torch.cat((all_probs, probs))

        new_best_k_words.append([-1, (hn, cn)])
        
    # get indices for the best k words: (index of the list in which the word is, index of the word in this list = index of the word in the vocabulary)
    list_indices = get_k_best_indices(all_probs, len(captions), tp.vocab_size)
    
    # update the new_best_k_words and captions
    new_captions = []
    for i, indices in enumerate(list_indices):
        new_best_k_words[i][0] = indices[1]
        
        caption = copy.deepcopy(captions[indices[0]])
        caption.append(indices[1])
        new_captions.append(caption)
    
    assert [word[0] for word in new_best_k_words] == [caption[-1] for caption in new_captions]
        
    return new_best_k_words, new_captions
        
    
        
        
# get a list of tuples (index of the list, index in the list) for the best k probabilities from all_probs
def get_k_best_indices(all_probs, k, vocab_size):
    
    # get the indices of the best k probabilities of word from all_probs
    top_k_indices = torch.topk(all_probs, k).indices
    
    # get the list of tuples of indicess
    list_indices = list(map(lambda x: (torch.tensor(int(x) // vocab_size).cuda(), torch.tensor(int(x) % vocab_size).cuda()), top_k_indices[0]))
    return list_indices                 