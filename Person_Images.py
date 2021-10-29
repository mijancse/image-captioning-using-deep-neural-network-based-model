file = open('COCO/text/person_images.txt', 'r')
# read all text
text = file.read()
persons = text.split('\n')
persons = [int(i) for i in persons]
print(len(persons))
file.close()


'''
file = open('COCO/text/train_images.txt', 'r')
text = file.read()
current_trains = text.split('\n')
current_trains = [i.split('.')[0] for i in current_trains]
print(len(current_trains))

person_train = []
for img in current_trains:
    id = int(img)
    if id in persons:
        person_train.append(img+".jpg")

print(len(person_train))
file.close()

file = open('COCO/text/validation_images.txt', 'r')
text = file.read()
current_vals = text.split('\n')
current_vals = [i.split('.')[0] for i in current_vals]
print(len(current_vals))

person_val = []
for img in current_vals:
    id = int(img)
    if id in persons:
        person_val.append(img+".jpg")
    
print(len(person_val))
file.close()

file = open('COCO/text/p_train_images.txt', 'w')
text = '\n'.join(person_train)
file.write(text)
file.close()


file = open('COCO/text/p_validation_images.txt', 'w')
text = '\n'.join(person_val)
file.write(text)
file.close()
'''

file = open('COCO/text/test_images.txt', 'r')
text = file.read()
current_test = text.split('\n')
current_test = [i.split('.')[0] for i in current_test]
print(len(current_test))

person_test = []
for img in current_test:
    id = int(img)
    if id in persons:
        person_test.append(img + ".jpg")

print(len(person_test))
file.close()

file = open('COCO/text/p_test_images.txt', 'w')
text = '\n'.join(person_test)
file.write(text)
file.close()