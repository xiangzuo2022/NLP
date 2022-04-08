import numpy as np
import pandas as pd
import emoji as emoji
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
print('Imported Successfully!')
print(emoji.EMOJI_UNICODE)
print(len(emoji.EMOJI_UNICODE))
emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }
for e in emoji_dictionary.values():
  print(emoji.emojize(e))

# Load the Training and Testing dataset
train = pd.read_csv('dataset/train_emoji.csv',header=None)
test = pd.read_csv('dataset/test_emoji.csv',header=None)

data = train.values
for i in range(10):
    print(data[i][0],emoji.emojize(emoji_dictionary[str(data[i][1])]))
    
embeddings = {}
with open('/newDisk/users/zhoumu/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        embeddings[word] = coeffs

print(embeddings['eat'])
print(embeddings['play'])