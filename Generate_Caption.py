from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, END, NW, DISABLED, HORIZONTAL
from PIL import Image, ImageTk



wnd = tk.Tk()

dirLb = ttk.Label(wnd, text="Image Path: ")
outLb = ttk.Label(wnd, text="Output: ")

dirTxt = ttk.Entry(wnd, width=60)
detectedText = tk.Text(wnd, height=5, width=30)
detectedText.insert(END, "Please select an image file.")
detectedText.config(state=DISABLED)
objLb = ttk.Label(wnd, text="Feedback: ")
fbText = tk.Text(wnd, height=5, width=30)
fbText.insert(END, "Please write you feedback here.")
ratingLb = ttk.Label(wnd, text="Rate this Detection:")
w = tk.Scale(wnd, from_=0, to=5, orient=HORIZONTAL)

c = tk.Canvas(wnd, width=300, height=300)
original = Image.open("default.jpg").resize((300, 300), Image.ANTIALIAS)
picture = ImageTk.PhotoImage(original)
myimg = c.create_image((0,0),image=picture, anchor="nw")

def openDir():
    detectedText.config(state="normal")
    detectedText.delete(1.0, "end")
    detectedText.insert(END, "Detecting Caption, Please Wait.")
    detectedText.config(state=DISABLED)
    dirTxt.delete(0, END)
    dirTxt.insert(0,filedialog.askopenfilename(filetypes=[("Image File",'.jpg')]))
    global picture
    picture = ImageTk.PhotoImage(Image.open(dirTxt.get()).resize((300, 300), Image.ANTIALIAS))#
    c.itemconfigure(myimg, image=picture)
    
    
    global model
    global tokenizer
    global max_cap_size
    
    # load and prepare the photograph
    
    #img_filename = input("Image Filename with extension: ")
    #photo_name = 'Generator/'+img_filename
    photo = extract_features(dirTxt.get())
    # generate description
    description = generate_desc(model, tokenizer, photo, max_cap_size).split()
    description = description[1:len(description)-1]
    description = ' '.join(description)
    print(description)
    detectedText.config(state="normal")
    detectedText.delete(1.0, "end")
    detectedText.insert(END, "Detected Caption : "+description)
    detectedText.config(state=DISABLED)


dirBtn = ttk.Button(wnd, text="Select Image", width=20, command=openDir)


def refresh():
    dirTxt.insert(END,"")
    global picture
    picture = ImageTk.PhotoImage(Image.open("default.jpg").resize((300, 300), Image.ANTIALIAS))
    c.itemconfigure(myimg, image=picture)
    dirTxt.delete(0, 'end')
    detectedText.config(state="normal")
    detectedText.delete(1.0, "end")
    detectedText.insert(END, "Please select an image file.")
    detectedText.config(state=DISABLED)
    fbText.delete(1.0, "end")
    fbText.insert(END, "Please write you feedback here.")


refreshBtn = ttk.Button(wnd, text="Refresh", width=20, command=refresh)



def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


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


# calculate the length of the description with the most words
def max_length(descriptions):
    lines,_ = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# extract features from each photo in the directory
def extract_features(filename):
    # load the model
    model = VGG19()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


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





directory = 'COCO'
# load the tokenizer
filename = directory+'/text/train_images.txt'
train = load_set(filename)
train_descriptions = load_clean_descriptions(directory+'/features/descriptions.txt', train)

tokenizer = load(open(directory+'/features/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_cap_size = max_length(train_descriptions)
# load the model
model = load_model(directory+'/models/best_model.h5')



dirBtn.grid(row=0, column=0, sticky=tk.W+tk.E, padx=10)
dirLb.grid(row=0, column=3, sticky=tk.W+tk.E, padx=10, pady=0)
dirTxt.grid(row=1, column=3, columnspan=2, sticky=tk.N, padx=10, pady=0)
outLb.grid(row=2, column=3, sticky=tk.N+tk.W+tk.E, padx=10, pady=0)
detectedText.grid(row=2, column=3, columnspan=2, sticky=tk.N+tk.W+tk.E, padx=10, pady=20)
objLb.grid(row=2, column=3, columnspan=2, sticky=tk.N+tk.W+tk.E, padx=10, pady=120)
fbText.grid(row=2, column=3, columnspan=2, sticky=tk.N+tk.W+tk.E, padx=10, pady=150)
ratingLb.grid(row=2, column=3, sticky=tk.N+tk.W+tk.E, padx=10, pady=250)
w.grid(row=2, column=3, sticky=tk.N+tk.W+tk.E, padx=10, pady=270)
c.grid(row=2, column=0, sticky=tk.N,columnspan=1)
refreshBtn.grid(row=2, column=0, columnspan=3, padx=10, sticky=tk.N, pady=320)

wnd.title("Detection")

#center window
wnd.geometry("700x450")
wnd.update_idletasks()

screen_width = wnd.winfo_screenwidth()
screen_height = wnd.winfo_screenheight()


size = tuple(int(_) for _ in wnd.geometry().split('+')[0].split('x'))
x = screen_width/2 - size[0]/2
y = screen_height/2 - size[1]/2

wnd.geometry("+%d+%d" % (x, y-20))
wnd.resizable(width=False, height=False)
wnd.mainloop()
