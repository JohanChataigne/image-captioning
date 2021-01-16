import os
import pandas as pd

# Get train and test images names
df_train_images = pd.read_csv('./flickr8k/annotations/trainImages.csv', names=['filename'])
df_test_images = pd.read_csv('./flickr8k/annotations/testImages.csv', names=['filename'])

print(f"Train shape: {df_train_images.shape}")
print(f"Test shape: {df_test_images.shape}")

# Split images in train and test folders
# 6000 train images and 1000 test images
imagesPath = './flickr8k/images/'
trainPath = imagesPath + 'train/'
testPath = imagesPath + 'test/' 
images = os.listdir(imagesPath)

for im in images[:10]:
    
    for file in df_train_images['filename']:
        if file == im:
            print(file + ' / ' + im)
            fullpath = imagesPath + im
            os.system('cp ' + fullpath + ' ' + trainPath)
            
    for file in df_test_images['filename']:
        if file == im:
            print(file + ' / ' + im)
            fullpath = imagesPath + im
            os.system('cp ' + fullpath + ' ' + testPath)
            
        
lines = list()

with open('./flickr8k/annotations/annotations_image_id_test.csv', 'r') as fr:
    
    lines = fr.readlines()
    print(len(lines))
    
    for line in lines[1:]:
        
        line_s = line.split(';')
        
        path = './flickr8k/images/test/' + line_s[0]
        
        if not os.path.exists(path):
            lines.remove(line)
    
    print(len(lines))
    
    
    
with open('./flickr8k/annotations/annotations_image_id_test.csv', 'w') as fw:
    
    for line in lines:
        
        fw.write(line)
        
        
        
print(len(lines))
    
