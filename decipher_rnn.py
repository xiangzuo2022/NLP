import os
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Input, Dense, TimeDistributed
from keras.models import Model
from keras.layers import Activation
from keras.optimizers import adam_v2
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
print("Imported Successfully!")
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def load_data(path):

    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

codes = load_data('cipher.txt')
plaintext = load_data('plaintext.txt')
print('Data Loaded Successfully!')



def tokenize(x):
    x_tk = Tokenizer(char_level=True)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)

print(text_tokenizer.word_index)

for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
    


def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post')

test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))
    
def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    print('Original Y: ', preprocess_y.shape)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    print('Reshaped Y: ', preprocess_y.shape)
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_code_sentences, preproc_plaintext_sentences, code_tokenizer, plaintext_tokenizer = preprocess(codes, plaintext)

print("Preprocessing Completed! ")

def simple_model(input_shape, output_sequence_length, code_vocab_size, plaintext_vocab_size):
    
    learning_rate = 1e-3

    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences=True)(input_seq)
    logits = TimeDistributed(Dense(plaintext_vocab_size))(rnn)

    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=adam_v2.Adam(learning_rate),
                  metrics=['accuracy'])
    return model

print("Function Created Successfully!")
tmp_x = pad(preproc_code_sentences, preproc_plaintext_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))
# Create the Model Object
simple_rnn_model = simple_model(
    tmp_x.shape,
    preproc_plaintext_sentences.shape[1],
    len(code_tokenizer.word_index)+1,
    len(plaintext_tokenizer.word_index)+1)

print("Model Object Created Successfully!")
simple_rnn_model.fit(tmp_x, preproc_plaintext_sentences, batch_size=64, epochs=6, validation_split=0.2)

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1) if index_to_words[prediction]!= '<PAD>'])

print("Function Created Successfully!")
for i in range(5):
    print('Original Sentence - {}'.format(plaintext[i]))
    print('Predicted Sentence - {}'.format(logits_to_text(simple_rnn_model.predict(tmp_x[i:i+1])[0], plaintext_tokenizer).upper()))
    print('*****************************************************************************')

x = ['N QNPJ LWFUJX FSI RFSLTJX .']
# Original Sentence - "I LIKE GRAPES AND MANGOES ."
print("Original Sentence - I LIKE GRAPES AND MANGOES .")
x = code_tokenizer.texts_to_sequences(x)
x = pad(x, preproc_plaintext_sentences.shape[1])
x = x.reshape((-1, preproc_plaintext_sentences.shape[-2], 1))
print("Predicted Sentence - ", logits_to_text(simple_rnn_model.predict(x[:1])[0], plaintext_tokenizer).upper())