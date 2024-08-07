import tensorflow as tf
from transformers import TFAutoModelWithLMHead, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
import numpy as np
import pandas as pd

from mario_gpt import SampleOutput


class PreferenceModel(tf.keras.Model):
    def __init__(self, model_name="gpt2"):
        super(PreferenceModel, self).__init__()
        self.language_model = TFAutoModelWithLMHead.from_pretrained(model_name)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='linear')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def call(self, inputs):
        outputs = self.language_model(inputs).last_hidden_state
        pooled_output = tf.reduce_mean(outputs, axis=1)
        x = self.dense1(pooled_output)
        x = self.dense2(x)
        return x

    def train_step(self, input_txts, scores):
        input_ids = self.tokenizer(input_txts, return_tensors='tf', padding=True, truncation=True,
                                   max_length=512).input_ids
        with tf.GradientTape() as tape:
            predictions = self.call(input_ids)
            loss = self.loss_fn(scores, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def val_step(self, input_txts, scores):
        input_ids = self.tokenizer(input_txts, return_tensors='tf', padding=True, truncation=True,
                                   max_length=512).input_ids
        predictions = self.call(input_ids)
        loss = self.loss_fn(scores, predictions)
        return loss, predictions

    def fit(self, train_dataset, val_dataset, num_epochs=10):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Training loop
            train_loss = 0
            for step, (input_txts, scores) in enumerate(train_dataset):
                input_ids = self.tokenizer(input_txts, return_tensors='tf', padding=True, truncation=True,
                                           max_length=512).input_ids
                loss = self.train_step(input_ids, scores)
                train_loss += loss.numpy()

            # Validation loop
            val_loss = 0
            all_predictions = []
            all_true_scores = []
            for input_ids, scores in val_dataset:
                loss, predictions = self.val_step(input_ids, scores)
                val_loss += loss.numpy()
                all_predictions.extend(predictions.numpy().flatten())
                all_true_scores.extend(scores.numpy().flatten())

            # Calculate metrics
            accuracy = accuracy_score(np.round(all_true_scores), np.round(all_predictions))
            precision, recall, _ = precision_recall_curve(all_true_scores, all_predictions)
            pr_auc = auc(recall, precision)

            print(f"Train Loss: {train_loss / len(train_dataset)}, Validation Loss: {val_loss / len(val_dataset)}")
            print(f"Validation Accuracy: {accuracy}, Validation PR-AUC: {pr_auc}")

    def save_model(self, path):
        self.save_weights(path)


def extract_level_text(levels: list):
    level_list = []
    for lvl in levels:
        generated_level = SampleOutput.load(lvl)
        level_txt = "\n".join(generated_level.level)
        level_list.append(level_txt)
    return level_list


def prepare_dataset(df):
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df[['prompt']])
    print('Train Shape:', train_df.shape)
    print('Test Shape:', test_df.shape)
    train_prompts = train_df.loc['prompt'].astype(str).tolist()
    train_levels = train_df.iloc['generate_level_file_path'].astype(str).tolist()
    train_levels_txt = extract_level_text(train_levels)
    train_scores = train_df.loc['shannon_entropy', 'playability', 'aesthetic'].to_numpy()
    val_prompts = test_df.loc['prompt'].astype(str).tolist()
    val_levels = test_df.loc['generate_level_file_path'].astype(str).tolist()
    val_levels_txt = extract_level_text(val_levels)
    val_scores = test_df.loc['shannon_entropy', 'playability', 'aesthetic'].to_numpy()
    # Concatenate the prompt and response
    train_input_texts = [prompt + " " + response for prompt, response in zip(train_prompts, train_levels_txt)]
    val_input_texts = [prompt + " " + response for prompt, response in zip(val_prompts, val_levels_txt)]
    # Convert scores to tensor
    train_scores_tf = tf.convert_to_tensor(train_scores, dtype=tf.float32)
    val_scores_tf = tf.convert_to_tensor(val_scores, dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_texts, train_scores_tf)).batch(2)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_input_texts, val_scores_tf)).batch(2)
    print(train_dataset)
    return train_dataset, val_dataset


if __name__ == "__main__":
    PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"
    dataset_path = "../sampling/sampling_score.csv"
    df = pd.read_csv(dataset_path, sep=",")
    train_dataset, val_dataset = prepare_dataset(df)

    # Initialize and train the preference model
    preference_model = PreferenceModel(PRETRAINED_MODEL_PATH)
    preference_model.fit(train_dataset, val_dataset, num_epochs=10)

    # Save the trained model
    preference_model.save_model("preference_model_weights.h5")
