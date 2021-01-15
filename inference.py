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


def beamSearch(cnn, rnn, image, tp, embedding_size, beam_k=20):
    pass
    
    