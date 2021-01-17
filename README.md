# Computer Vision: Image Captioning

Authors: [Chataigner Johan](https://github.com/JohanChataigne), [Germon Paul](https://github.com/pgermon), and [Martin Hugo](https://github.com/ScarfZapdos).

This project deals with the problem of **automatically describing the content of images**. It implements several methods of **computer vision** and **natural language processing** in order to generate a **textual description** of a given image as much precise as possible.

## Content of the project

📦image-captioning  
 ┣ 📂figures // *contains diamgrams about the performance of the different models*  
 ┣ 📂flickr8k // *directory of the flickr8k dataset*  
 ┃ ┣ 📂annotations  
 ┃ ┃ ┣ 📜annotations_image_id_test.csv // *contains captions for the test images*  
 ┃ ┃ ┣ 📜annotations_image_id_train.csv  // *contains captions for the train images*    
 ┃ ┃ ┣ 📜testImages.csv // *contains the names of the images in the testing set*   
 ┃ ┃ ┗ 📜trainImages.csv  // *contains the names of the images in the training set*   
 ┃ ┣ 📂images  
 ┃ ┃ ┣ 📂train  
 ┃ ┗ ┗ 📂test  
 ┣ 📂models // *contains the saved trained models*  
 ┣ 📜evaluate.ipynb // *Computes the score of the different models*  
 ┣ 📜inference.py // *Contains inference methods for caption choice*  
 ┣ 📜model_v1_random.ipynb  
 ┣ 📜model_v1_repeat.ipynb  
 ┣ 📜model_v2_random.ipynb  
 ┣ 📜ngram.ipynb  
 ┣ 📜random_caption_dataset.py //  *Random Caption class for models*   
 ┣ 📜README.md  
 ┣ 📜repeat_image_dataset.py //  *Repeat Image class for models*  
 ┣ 📜requirements.txt  
 ┣ 📜setup.py // *To launch once the repository is cloned*   
 ┣ 📜text_preprocessing.py // *Preprocessor class*  
 ┗ 📜transforms.py // *Preprocessing file*  


## Launching the project

In order to use this project locally:
1) **Clone this repository** with `git clone https://github.com/JohanChataigne/image-captioning.git`.  
2) **Execute setup.py** with `python3 setup.py`. This will download the dataset's images, install the requirements and organize your repository.


After that, you should be able to use the notebooks without difficulties.  
