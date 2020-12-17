import os

base_file = './flickr8k/annotations/annotations_image_id.csv'

with open(base_file) as f:
    
    print(f.readline())
    print(f.readline())
    
    '''
    for line in f:
        im, caption = line.split(';')
        print(im)
        print(caption)
        '''