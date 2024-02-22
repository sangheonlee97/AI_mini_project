from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

df = pd.read_csv('../resource/word.csv')
print(df)

label_k = df['한국어']
print(pd.value_counts(label_k))
token = Tokenizer()
token.fit_on_texts(label_k)
print(token.word_index)
label_real = token.texts_to_sequences(label_k)
label_real = pd.DataFrame(label_real)

df['한국어'] = label_real
print(df)

df.to_csv('../resource/tokend_word.csv', index=0)