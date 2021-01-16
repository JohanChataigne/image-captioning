# Computer Vision: Image Captioning

Authors: [Chataigner Johan](https://github.com/JohanChataigne), [Germon Paul](https://github.com/pgermon), and [Martin Hugo](https://github.com/ScarfZapdos).

This project deals with the problem of automatically describing the content of images. It implements several methods of computer vision and natural language processing in order to generate a textual description of a given image as much precise as possible.

## Content of the project

### Repository tree organization

📦image-captioning
 ┣ 📂figures  
 ┃ ┣ 📜Loss model random v2 training with lr=0.01.png  
 ┃ ┣ 📜Loss model random v2 training with lr=0.1.png  
 ┃ ┗ 📜Loss_model_random_v2_lr=0.01_LSTM_init_random.png  
 ┣ 📂flickr8k  
 ┃ ┣ 📂annotations  
 ┃ ┃ ┣ 📜annotations_image_id_test.csv  
 ┃ ┃ ┣ 📜annotations_image_id_train.csv    
 ┃ ┃ ┣ 📜testImages.csv  
 ┃ ┃ ┗ 📜trainImages.csv  
 ┃ ┣ 📂images  
 ┃ ┗ ┣ 📂train  
 ┃   ┗ 📂test  
 ┣ 📂models   
 ┃ ┣ 📜model_random_v2_init0_lstm3  
 ┃ ┣ 📜model_random_v2_initrandom_lstm3  
 ┃ ┗ 📜ngram_512_v1  
 ┣ 📜clean_up.py  
 ┣ 📜evaluate.ipynb  
 ┣ 📜image_captioning.ipynb  
 ┣ 📜inference.py  
 ┣ 📜model_v1_random.ipynb  
 ┣ 📜model_v1_repeat.ipynb  
 ┣ 📜model_v2_random.ipynb  
 ┣ 📜ngram.ipynb  
 ┣ 📜random_caption_dataset.py  
 ┣ 📜README.md  
 ┣ 📜repeat_image_dataset.py  
 ┣ 📜requirements.txt 
 ┣ 📜text_preprocessing.py  
 ┗ 📜transforms.py  


M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artifical Intellegence Research, Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html