import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts([d["level"] for d in data])
sequences = tokenizer.texts_to_sequences([d["level"] for d in data])
X = pad_sequences(sequences)

# Prepare labels
y = [d["score"] for d in data]

# Define a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10)
