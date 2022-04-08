# Import the required libraries
from tensorflow.keras.datasets import imdb
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.python.keras.models import Sequential

# Load the Dataset
((XT,YT),(Xt,Yt)) = imdb.load_data(num_words=30000)
print("The length of the Training Dataset is ", len(XT))
print("The length of the Testing Dataset is ", len(Xt))

# Perform the padding
X_train = sequence.pad_sequences(XT,maxlen=500)
X_test = sequence.pad_sequences(Xt,maxlen=500)

# Create the Model Architecture
model = Sequential()
model.add(Embedding(30000,128))
model.add(SimpleRNN(64))
model.add(Dense(1,activation='sigmoid'))

# Compile the Model
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

# Create the Callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
earlystop = EarlyStopping(monitor='val_acc',patience=1)

# Train the Model
hist = model.fit(X_train,YT,validation_split=0.2,epochs=10,batch_size=128,callbacks=[checkpoint,earlystop])

# Evaluate the Model on Test Dataset
model.evaluate(X_test,Yt)
sent = "This movie is really bad . I do not like this movie because the direction was horrible ."
inp = []

# Get the word:integer mapping
word_idx = imdb.get_word_index()

# Convert each word to integer
for word in sent.split():
  if word in word_idx.keys():
    inp.append(word_idx[word])
  else:
    inp.append(1)

print(inp) 

# Perform the padding
final_input = sequence.pad_sequences([inp],maxlen=500)

# Finally predict the sentiment
model.predict(final_input)