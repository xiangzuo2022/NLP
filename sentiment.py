import os
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "7"



((XT,YT),(Xt,Yt)) = imdb.load_data(num_words=30000)
print('Dataset Loaded!')
word_idx = imdb.get_word_index()
idx_word = dict([value,key] for (key,value) in word_idx.items())
actual_review = ' '.join([idx_word.get(idx-3,'?') for idx in XT[0]])
print(actual_review)
print(len(actual_review.split()))


X_train = sequence.pad_sequences(XT,maxlen=500)
X_test = sequence.pad_sequences(Xt,maxlen=500)

print("Padding Completed!")

model = Sequential()
model.add(Embedding(30000,128))
model.add(SimpleRNN(64))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
print("Model Compiled Successfully!")

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
earlystop = EarlyStopping(monitor='val_acc',patience=1)

print("Callbacks Created Successfully!")

hist = model.fit(X_train,YT,validation_split=0.2,epochs=10,batch_size=128,callbacks=[checkpoint,earlystop])

# Evaluate the Model on Test Dataset
model.evaluate(X_test,Yt)


