# Image Captioning Using Deep Neural Network Based Model

## Summary
The project combined two deep learning models for automatic image captioning using CNN (Convolutional Neural Network) and RNN (Recurrent Neural Network) or LSTM (Long Short Term Memory model). The proposed system was built using reliable python libraries such as TensorFlow, Keras, nltk, NumPy, etc. Three standard benchmark datasets, such as COCO, Flikr8k, Flikr30k, were used to train and test the program. The trained model was further tested on local data for evaluating the system's performance.

## The Proposed Model
The proposed system used two deep learning models, CNN that excels at remembering spatial details and identifying features in images, and an RNN model (generally, the LSTM model) that can process any sequential data such as word sequence generation. The encoder converts raw inputs into feature representations, which are then passed on to the decoder to generate final captions as output. The deep learning models combines two different models; CNN for processing images and LSTM or RNN model for text prediction. 

### CNN Encoder
Reads the photograph data and converts the output into a fixed-length vector using an internal representation.
### LSTM Decoder
Read the embedded image and generate a textual description (e.g., final caption).

## Python Programs/Libraries
### Tensorflow: A free and open-source software library for dataflow and differentiable programming across various tasks.
#### Keras: Open-source neural network library is written in Python. This project used this library to handle text data, tokenize, and create and modify relevant neural networks and predefined models. 
### OpenCV: Library of programming functions mainly aimed at real-time computer vision.
### NLTK: The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for English symbolic and statistical natural language processing. The program used these tools for evaluating the model through the BLEU score.
### Pillow: A python Imaging Library.
### Numpy: main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy, dimensions are called axes.
### h5py uses straightforward NumPy and Python metaphors, like dictionary and NumPy array syntax. So, for example, you can iterate over datasets in a file or check out the .shape or .dtype attributes of datasets.

## Images/Text Datasets
Three Benchmark Datasets, including MS COCO, Flikr8k, and Flikr30k, were used for training and testing the model. In addition, however, some local images have been used to confirm and validate the accuracy of the program.

## Read More
Read more for details: [Image_Caption_Generator_User_Manual](https://github.com/mijancse/image-captioning-using-deep-neural-network-based-model/blob/main/Image_Caption_Generator_User_Manual.pdf); [Image_Captioning_Process](https://github.com/mijancse/image-captioning-using-deep-neural-network-based-model/blob/main/Image_Captioning_Process.pdf). [Scripts & Datasets Download](https://drive.google.com/drive/folders/1h43vZTgNk-ulihv2jSf0QLrRI0ty3mOn?usp=sharing)

## Contact
Prof. Dr. Md. Mijanur Rahman, Department of Computer Science and Engineering, Jatiya Kabi Kazi Nazrul Islam University, Trishal, Mymensingh, Bangladesh. Email: mijan@jkkniu.edu.bd

## Citation
Md. Mijanur Rahman, Ashik Uzzaman, Sadia Islam Sami, "Image captioning using deep neural network based model", Department of Computer Science and Engineering, Jatiya Kabi Kazi Nazrul Islam University. Github Repository, 2021. https://github.com/mijancse/image-captioning-using-deep-neural-network-based-model
