from PIL import Image
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt

#Train DIR
DIR1 = os.getcwd()+'/train/forged/'
DIR2 = os.getcwd()+'/train/genuine/'

#Test DIR
DIR3=os.getcwd()+'/TestImage/forged/'
#DIR4=os.getcwd()+'/TestImage/genuine/'

#Validate DIR
DIR5=os.getcwd()+'/validate/forged/'
DIR6=os.getcwd()+'/validate/genuine/'


def label_img(name):
    for name in os.listdir(DIR1):
        label='forged'
    for name in os.listdir(DIR2):
        label='genuine'
    for name in os.listdir(DIR5):
        label='forged'
    for name in os.listdir(DIR6):
        label='genuine'        
    if label == 'forged': return np.array([1, 0])
    elif label == 'genuine' : return np.array([0, 1])

def load_training_data():
    train_data = []
    
    for img in os.listdir(DIR1):
        label = np.array([1,0])
        path = os.path.join(DIR1, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        train_data.append([np.array(img), label])

    for img in os.listdir(DIR2):
        label = np.array([0,1])
        path = os.path.join(DIR2, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        train_data.append([np.array(img), label])        
            
    shuffle(train_data)
    return train_data    

def load_validate_data():
    validate_data = []
    for img in os.listdir(DIR5):
        label = np.array([1,0])
        path = os.path.join(DIR5, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        validate_data.append([np.array(img), label])

    for img in os.listdir(DIR6):
        label = np.array([0,1])
        path = os.path.join(DIR6, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        validate_data.append([np.array(img), label])

    return validate_data    

def load_test_data():
    test_data = []
    for img in os.listdir(DIR3):
        label = np.array([1,0])
        path = os.path.join(DIR3, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        test_data.append([np.array(img), label])
    """
    for img in os.listdir(DIR4):
        label = np.array([0,1])
        path = os.path.join(DIR4, img)
        img = Image.open(path)
        img = img.convert('L')
        img = img.resize((500, 500), Image.ANTIALIAS)
        test_data.append([np.array(img), label])
	"""
    return test_data    

train_data = load_training_data()

valid_data = load_validate_data()

test_data = load_test_data()

trainImages = np.array([i[0][0] for i in train_data]).reshape(-1, 500)
trainLabels = np.array([i[1] for i in train_data])

validateImages = np.array([i[0][0] for i in valid_data]).reshape(-1, 500)
validateLabels = np.array([i[1] for i in valid_data])


testImages = np.array([i[0][0] for i in test_data]).reshape(-1, 500)
testLabels = np.array([i[1] for i in test_data])

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

model = Sequential()
model.add(Embedding(500, 100, input_shape=(500,)))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.01)))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.02))

model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

history = model.fit(trainImages,trainLabels, validation_data = (validateImages, validateLabels) , batch_size=100, epochs=20, verbose=1)

model.summary()

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
a = acc * 100
print("\nAccuracy = ",a)

img=Image.open('04402044.png')
img = img.convert('L')
img = img.resize((500, 500), Image.ANTIALIAS)
predict_data=[]
predict_data.append([np.array(img)])
predictImages = np.array([i for i in predict_data]).reshape(-1, 500)
predictions = model.predict(predictImages, batch_size=1)

if predictions[0][0]>predictions[0][1]:
	print('INPUT SIGNATURE IS GENUINE')
else:
	print('INPUT SIGNATURE IS FORGE')	

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
