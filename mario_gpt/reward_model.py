import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

def reward_model():
    # Define the input layers for prompt and response
    prompt_input = Input(shape=(None,), name='prompt_input')
    response_input = Input(shape=(None,), name='response_input')

    # Define the embedding layers for prompt and response
    prompt_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=None)(prompt_input)
    response_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=None)(response_input)

    # Concatenate the embeddings
    concatenated = Concatenate()([prompt_embedding, response_embedding])

    # Define the dense layers
    x = Dense(64, activation='relu')(concatenated)
    x = Dense(32, activation='relu')(x)

    # Define the output layer with 4 dimensions (playability, novelty, difficulty, aesthetics)
    output = Dense(4, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=[prompt_input, response_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def reward_trainer(prompts: list, levels: list, output_data: np.array):
    # Create a tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)

    # Fit the tokenizer to the prompts and responses
    tokenizer.fit_on_texts(prompts + levels)

    # Convert the prompts and responses to sequences
    prompt_sequences = tokenizer.texts_to_sequences(prompts)
    response_sequences = tokenizer.texts_to_sequences(levels)

    # Pad the sequences to have the same length
    max_length = 200
    prompt_padded = tf.keras.preprocessing.sequence.pad_sequences(prompt_sequences, maxlen=max_length)
    response_padded = tf.keras.preprocessing.sequence.pad_sequences(response_sequences, maxlen=max_length)

    # Create the input data
    input_data = [prompt_padded, response_padded]

    if tf.test.is_gpu_available():
        # Use the GPU
        device = '/gpu:0'
    else:
        # Use the CPU
        device = '/cpu:0'

    with tf.device(device):
        model = reward_model()
        model.fit(input_data, output_data, epochs=10, batch_size=32)

    model.save('mariogpt_reward_model.h5')


def train(dataset_path):
    df = pd.read_csv(dataset_path)
    prompts = df.iloc[:, 0].astype(str).tolist()
    levels = df.iloc[:, 4].astype(str).tolist()
    scores = df.iloc[:, 5:9].to_numpy()

    reward_trainer(prompts, levels, scores)


if __name__ == "__main__":
    train("../sampling/sampling_new.csv")