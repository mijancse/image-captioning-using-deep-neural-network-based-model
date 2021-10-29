from os import listdir
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


# extract features from each photo in the directory
def extract_features(directory):
    model = VGG19()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
    return features


if __name__ == '__main__':
    print('Initializing Feature Extraction')
    directory = 'Flickr30k/flickr30k_images'
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))
    dump(features, open('Flickr30k/Features/features.pkl', 'wb'))
