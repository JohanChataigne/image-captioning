import torch 


# Generate caption for given image
def sampling(cnn, embedding, rnn, image, tp, embedding_size, hidden_size, max_length=20):
          
    with torch.no_grad():

        caption = list()

        # Random init the lstm state
        h0 = torch.rand((1, 1, hidden_size)).cuda()
        c0 = torch.rand((1, 1, hidden_size)).cuda()

        # Encode input image
        image_embedding = cnn(image).view(-1, 1, embedding_size).cuda()

        # Get first word prediction probabilities
        (hn, cn), probs = rnn(image_embedding, (h0, c0))

        # Extract predicted word
        pred_idx = torch.argmax(probs)
        pred_word_vect = tp.encoding_matrix[pred_idx]
        predicted_word = tp.vect_to_word(pred_word_vect)

        caption.append(predicted_word)
        
        i = 0

        # Build caption until model outputs stop word
        while predicted_word != '<stop>' and i < max_length:

            word_embedding = embedding(pred_idx).view(1, 1, embedding_size).cuda()

            (hn, cn), probs = rnn(word_embedding, (hn, cn))

            pred_idx = torch.argmax(probs)
            pred_word_vect = tp.encoding_matrix[pred_idx]
            predicted_word = tp.vect_to_word(pred_word_vect)

            caption.append(predicted_word)
            
            i+=1


        caption = " ".join(caption)
        
        return caption


def beamSearch(cnn, embedding, rnn, image, tp, embedding_size, beam_k=20, max_length=20):
    with torch.no_grad():
        
        captions = [[] for i in range(beam_k)]
        captions_over = [False for i in range(beam_k)]
        
        best_k_words = generate_first_k_words(cnn, embedding, rnn, image, tp, embedding_size, beam_k)
        previous_best_k_words = best_k_words.copy()
        
        while not all(captions_over):
            
            previous_best_k_words, captions = generate_next_k_words(previous_best_k_words, captions, 
                                                                    cnn, embedding, rnn, image, tp, embedding_size, beam_k)
            
            captions_over = update_captions_state(captions, captions_over)

                           
            
def update_captions_state(captions, captions_over):
    
    for i, caption in enumerate(captions):
        if len(caption > max_length) or caption[-1] == '<stop>':
            captions_over[i] = True
            
    return captions_over
    
    
            
def generate_first_k_words(cnn, embedding, rnn, image, tp, embedding_size, beam_k):
    
    # Random init the lstm state
    h0 = torch.rand((1, 1, hidden_size)).cuda()
    c0 = torch.rand((1, 1, hidden_size)).cuda()
    
    # Encode input image
    image_embedding = cnn(image).view(-1, 1, embedding_size).cuda()

    # Get first word prediction probabilities
    (hn, cn), probs = rnn(image_embedding, (h0, c0))

    # (index, (hn, cn)
    best_k_words = []

    top_k = torch.topk(probs, beam_k)

    for i in top_k.indices:

        index = int(i)
        word_vect = tp.encoding_matrix[index]
        word = tp.vect_to_word(word_vect)

        best_k_words.append((index, (hn, cn)))
        
    return best_k_words
        
        
                                
def generate_next_k_words(previous_best_k_words, captions, cnn, embedding, rnn, image, tp, embedding_size, beam_k):
    
    all_probs = torch.empty(0)
    list_probs = []        
    best_k_words = []   

    # get the k lists of probabilities
    for i in range(beam_k):

        index = previous_best_k_words[i][0]
        word_embedding = embedding(index).view(1, 1, embedding_size).cuda()

        (hn, cn), probs = rnn(word_embedding, previous_best_k_words[i][1])

        list_probs.append(probs)
        all_probs = torch.cat((all_probs, probs))

        best_k_words.append((-1, (hn, cn)))
        
        
    list_indices = get_k_best_indices(all_probs, beam_k, tp.vocab_size)
    
    for i, indices in enumerate(list_indices):
        best_k_words[i][0] = indices[1]
        
        # fix the previous word in the caption for this word
        word_vect = tp.encoding_matrix[previous_best_k_words[i][0]]
        word = tp.vect_to_word(word_vect)
        captions[i].append(word)
        
    return best_k_words, captions
        
    
        
        
# get a list of tuples (list index, index in the list) from all_probs (the concatenation of all probs list)
def get_k_best_indices(all_probs, k, vocab_size):
    
    best_k_indices = torch.topk(all_probs, k)
    
    list_indices = list(map(lambda x: (int(x) // vocab_size, int(x) % vocab_size), best_k_indices))
    return list_indices                 