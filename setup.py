import os
import pandas as pd


### Install requirements and the images

#os.system('pip install -r requirements.txt')
#os.system('kaggle datasets download -d adityajn105/flickr8k')
#os.system('unzip flickr8k.zip')
#os.system('rm -f captions.txt')
os.system('mkdir flickr8k/images/')
os.system('mkdir flickr8k/images/train')
os.system('mkdir flickr8k/images/test')
os.system('mv Images/* flickr8k/images/ && rm -rf Images/')

### Setup the directory

# Get train and test images names
df_train_images = pd.read_csv('./flickr8k/annotations/trainImages.csv', names=['filename'])
df_test_images = pd.read_csv('./flickr8k/annotations/testImages.csv', names=['filename'])

imagesPath = 'flickr8k/images/'
trainPath = imagesPath + 'train/'
testPath = imagesPath + 'test/' 
images = os.listdir(imagesPath)

for im in images:
    
    for file in df_train_images['filename']:
        if file == im:
            fullpath = imagesPath + im
            os.system('mv ' + fullpath + ' ' + trainPath)
            
    for file in df_test_images['filename']:
        if file == im:
            fullpath = imagesPath + im
            os.system('mv ' + fullpath + ' ' + testPath)

            
            
### Clean test and train annotations

def clean_annotations(dataset_type):
    
    assert dataset_type in ['train', 'test']
    
    lines = list()

    with open(f'./flickr8k/annotations/annotations_image_id_{dataset_type}.csv', 'r') as fr:

        lines = fr.readlines()

        for line in lines[1:]:

            line_s = line.split(';')

            path = f'./flickr8k/images/{dataset_type}/' + line_s[0]

            if not os.path.exists(path):
                lines.remove(line)

    with open(f'./flickr8k/annotations/annotations_image_id_{dataset_type}.csv', 'w') as fw:

        for line in lines:

            fw.write(line)
        
        
clean_annotations('train')
clean_annotations('test')