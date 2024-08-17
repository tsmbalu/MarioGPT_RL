import tensorflow as tf
import keras
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

from mario_gpt import SampleOutput

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"


class PreferenceModel(keras.Model):
    def __init__(self, input_dim=512, output_dim=3):
        super(PreferenceModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(input_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='linear')  # Use linear activation for regression
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = tf.keras.losses.MeanSquaredError()  # Use MSE for regression

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def train_step(self, data):
        inputs, scores = data
        with tf.GradientTape() as tape:
            predictions = self.call(inputs)
            loss = self.loss_fn(scores, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        inputs, scores = data
        predictions = self.call(inputs)
        loss = self.loss_fn(scores, predictions)
        return {'loss': loss, 'predictions': predictions}

    def fit(self, train_dataset, val_dataset, num_epochs=10):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Training loop
            train_loss = 0
            for step, data in enumerate(train_dataset):
                logs = self.train_step(data)
                train_loss += logs['loss'].numpy()

            # Validation loop
            val_loss = 0
            all_predictions = []
            all_true_scores = []
            for data in val_dataset:
                logs = self.test_step(data)
                val_loss += logs['loss'].numpy()
                all_predictions.extend(logs['predictions'].numpy().flatten())
                all_true_scores.extend(data[1].numpy().flatten())

            # Calculate regression metrics
            mse = mean_squared_error(all_true_scores, all_predictions)
            mae = mean_absolute_error(all_true_scores, all_predictions)

            print(f"Train Loss: {train_loss / len(train_dataset)}, Validation Loss: {val_loss / len(val_dataset)}")
            print(f"Validation MSE: {mse}, Validation MAE: {mae}")

    def save_model(self, path):
        self.save_weights(path)


def extract_level_text(levels: list):
    level_list = []
    for lvl in levels:
        generated_level = SampleOutput.load(lvl)
        level_txt = "\n".join(generated_level.level)
        level_list.append(level_txt)
    return level_list


def split_into_chunks(text, max_length):
    words = text.split()
    return [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]


def prepare_dataset(df):
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[['prompt']])
    print('Train Shape:', train_df.shape)
    print('Test Shape:', test_df.shape)
    train_prompts = train_df['prompt'].astype(str).tolist()
    train_levels = train_df['level_file_path'].astype(str).tolist()
    train_levels_txt = extract_level_text(train_levels)
    train_scores = train_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()
    val_prompts = test_df['prompt'].astype(str).tolist()
    val_levels = test_df['level_file_path'].astype(str).tolist()
    val_levels_txt = extract_level_text(val_levels)
    val_scores = test_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()

    # Concatenate the prompt and response
    train_input_texts = [f"{prompt} {response}" for prompt, response in zip(train_prompts, train_levels_txt)]
    val_input_texts = [f"{prompt} {response}" for prompt, response in zip(val_prompts, val_levels_txt)]

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Assuming train_input_texts and val_input_texts are lists of strings
    train_chunks = [split_into_chunks(text, max_length=512) for text in train_input_texts]
    val_chunks = [split_into_chunks(text, max_length=512) for text in val_input_texts]

    # Flatten the lists of chunks
    train_input_texts = [chunk for chunks in train_chunks for chunk in chunks]
    val_input_texts = [chunk for chunks in val_chunks for chunk in chunks]

    train_encodings = tokenizer(train_input_texts, truncation=False, padding=True, max_length=512)
    val_encodings = tokenizer(val_input_texts, truncation=False, padding=True, max_length=512)

    train_input_ids = tf.convert_to_tensor(train_encodings['input_ids'], dtype=tf.int32)
    val_input_ids = tf.convert_to_tensor(val_encodings['input_ids'], dtype=tf.int32)
    train_scores_tf = tf.convert_to_tensor(
        np.repeat(train_scores, [len(split_into_chunks(txt, max_length=512)) for txt in train_input_texts], axis=0),
        dtype=tf.float32)
    val_scores_tf = tf.convert_to_tensor(
        np.repeat(val_scores, [len(split_into_chunks(txt, max_length=512)) for txt in val_input_texts], axis=0),
        dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_ids, train_scores_tf)).batch(2)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_input_ids, val_scores_tf)).batch(2)

    return train_dataset, val_dataset


if __name__ == "__main__":
    dataset_path = "../sampling/sampling_score.csv"
    df = pd.read_csv(dataset_path, sep=",")
    train_dataset, val_dataset = prepare_dataset(df)

    # Initialize and train the preference model
    preference_model = PreferenceModel()
    preference_model.fit(train_dataset, val_dataset, num_epochs=50)

    # Save the trained model
    preference_model.save_model("preference_model.weights.h5")
