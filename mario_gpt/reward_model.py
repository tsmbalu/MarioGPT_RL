import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Embedding, Dropout, Conv1D, MaxPooling1D,\
    GlobalAveragePooling1D


def define_model(max_length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(max_length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    dense1 = Dense(32, activation='relu')(embedding1)
    drop1 = Dropout(0.5)(dense1)
    # pool1 = MaxPooling1D(pool_size=2)(drop1)
    pool1 = GlobalAveragePooling1D()(drop1)
    flat1 = Flatten()(pool1)
    # channel 2
    inputs2 = Input(shape=(max_length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    dense2 = Dense(32, activation='relu')(embedding2)
    drop2 = Dropout(0.5)(dense2)
    # pool2 = MaxPooling1D(pool_size=2)(drop2)
    pool2 = GlobalAveragePooling1D()(drop2)
    flat2 = Flatten()(pool2)
    # merge
    merged = Concatenate()([flat1, flat2])
    # interpretation
    dense1 = Dense(32, activation='relu')(merged)
    outputs = Dense(4, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # summarize
    print(model.summary())
    return model


def reward_model(max_length):
    # Define the input layers for prompt and response
    prompt_input = Input(shape=(200,), name='prompt_input')
    response_input = Input(shape=(200,), name='response_input')

    # Define the embedding layers for prompt and response
    prompt_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=None)(prompt_input)
    response_embedding = tf.keras.layers.Embedding(input_dim=10000, output_dim=128, input_length=None)(response_input)

    # Concatenate the embeddings
    concatenated = Concatenate()([prompt_embedding, response_embedding])

    # Define the dense layers
    x = Flatten()(concatenated)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Define the output layer with 4 dimensions (playability, novelty, difficulty, aesthetics)
    output = Dense(4, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=[prompt_input, response_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    return model


def to_sequence(prompts, levels, max_length=200):
    # Create a tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)

    # Fit the tokenizer to the prompts and responses
    tokenizer.fit_on_texts(prompts + levels)

    # Convert the prompts and responses to sequences
    prompt_sequences = tokenizer.texts_to_sequences(prompts)
    response_sequences = tokenizer.texts_to_sequences(levels)

    # Pad the sequences to have the same length
    prompt_padded = tf.keras.preprocessing.sequence.pad_sequences(prompt_sequences, maxlen=max_length)
    response_padded = tf.keras.preprocessing.sequence.pad_sequences(response_sequences, maxlen=max_length)

    # Create the input data
    sequence_data = [prompt_padded, response_padded]
    return sequence_data

def reward_trainer(train_prompts: list, train_levels: list, train_score: np.array, test_prompts: list, test_levels: list, test_score: np.array):
    max_length = 200
    train_input = to_sequence(train_prompts, train_levels, max_length)
    test_input = to_sequence(test_prompts, test_levels, max_length)
    if tf.test.is_gpu_available():
        # Use the GPU
        device = '/gpu:0'
    else:
        # Use the CPU
        device = '/cpu:0'

    with tf.device(device):
        model = define_model(max_length, 10000)
        model.fit(train_input, train_score, epochs=20, batch_size=32, validation_data=(test_input, test_score))

    model.save('mariogpt_reward_model.h5')


def trainer(dataset_path):
    df = pd.read_csv(dataset_path, sep=",")
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[['prompt']])

    print('Train Shape:', train_df.shape)
    print('Test Shape:', test_df.shape)

    train_prompts = train_df.iloc[:, 0].astype(str).tolist()
    train_levels = train_df.iloc[:, 4].astype(str).tolist()
    train_scores = train_df.iloc[:, 5:9].to_numpy()

    test_prompts = test_df.iloc[:, 0].astype(str).tolist()
    test_levels = test_df.iloc[:, 4].astype(str).tolist()
    test_scores = test_df.iloc[:, 5:9].to_numpy()

    reward_trainer(train_prompts, train_levels, train_scores, test_prompts, test_levels, test_scores)


if __name__ == "__main__":
    trainer("../sampling/sampling_score.csv")
