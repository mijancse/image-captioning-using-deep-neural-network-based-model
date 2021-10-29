from numpy import argmax
from pickle import load
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features


# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    word_count_threshold = 10
    word_counts = {}

    for key in descriptions.keys():
        sentences = descriptions[key]
        for sent in sentences:
            for w in sent.split(' '):
                word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc, vocab


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    _,lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines,_ = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    file = open(directory+'/output/predicted_captions.txt','w')

    # step over the whole set
    count = 0
    for key, desc_list in descriptions.items():
        count += 1
        # generate description
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

        predicted_caption = ' '.join(yhat.split()[1:len(yhat.split())-1])
        entry = key + ' ' + predicted_caption + '\n'
        file.write(entry)

        #image = load_img('Flicker8k_Dataset/'+key+'.jpg')
        #plt.title(predicted_caption)
        #plt.imshow(image)
        #plt.axis('off')
        #plt.savefig('Output/Images/'+key+'.jpg')
        print('Processed:',count)

    file.close()
    # calculate BLEU score
    bleu_1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % bleu_1)
    print('BLEU-2: %f' % bleu_2)
    print('BLEU-3: %f' % bleu_3)
    print('BLEU-4: %f' % bleu_4)

    result = open(directory + '/output/results.txt', 'w')
    result.write('BLEU-1: %f\n' % bleu_1)
    result.write('BLEU-2: %f\n' % bleu_2)
    result.write('BLEU-3: %f\n' % bleu_3)
    result.write('BLEU-4: %f\n' % bleu_4)
    result.close()


# prepare tokenizer on train set

# load training dataset (6K)
directory = 'COCO'

filename = directory+'/text/train_images.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open(directory+'/features/tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# prepare test set

# load test set
filename = directory+'/text/test_images.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features(directory+'/features/features.pkl', test)
print('Photos: test=%d' % len(test_features))


# load the model
#filename = directory+'/models/best_model.h5'
filename = directory+'/models/model_0.h5'
model = load_model(filename)
print('Model Loaded')
# evaluate model
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)