import math
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import torch

from sklearn.metrics import accuracy_score

from mario_gpt import MarioLM, SampleOutput

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"
MODEL_SAVE_PATH = '../checkpoints/pm'
SAVE_FREQ = 5


class PreferenceModel(keras.Model):
    def __init__(self, output_dim=3, name='preference_model'):
        super(PreferenceModel, self).__init__(name=name)
        self.dense1 = keras.layers.Dense(2168, activation='relu',
                                         kernel_regularizer=keras.regularizers.l2(0.01))
        self.dropout1 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(1084, activation='relu',
                                         kernel_regularizer=keras.regularizers.l2(0.01))
        self.dense3 = keras.layers.Dense(output_dim, activation='linear')  # Use linear activation for regression
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = keras.losses.MeanSquaredError()  # Use MSE for regression

        self.ckpt = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.ckpt, MODEL_SAVE_PATH, max_to_keep=5)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dense3(x)
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
        # Convert tensors to NumPy arrays
        scores_np = tf.squeeze(scores).numpy()
        predictions_np = tf.squeeze(predictions).numpy()
        r2score = r2_score(scores_np, predictions_np)
        return {'loss': loss, 'predictions': predictions, 'r2_score': r2score}

    def fit(self, train_dataset, val_dataset, num_epochs=10, steps_per_epoch=None, validation_steps=None):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Training loop
            train_loss = 0
            for step, (inputs, scores) in enumerate(train_dataset):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                data = (inputs, scores)
                logs = self.train_step(data)
                train_loss += logs['loss'].numpy()

            val_loss, mse, mae, val_metric = self.evaluate(val_dataset, validation_steps)

            print(
                f"Train Loss: {train_loss / steps_per_epoch}, Validation Loss: {val_loss}")
            print(f"Validation MSE: {mse}, Validation MAE: {mae}, Validation R2 Score: {val_metric}")

            if (epoch + 1) % SAVE_FREQ == 0:
                self.checkpoint_manager.save()
                print(f"Model saved at epoch {epoch + 1}")

    def evaluate(self, dataset, steps=None):
        """
        Evaluate the model on the provided dataset.
        :param dataset: tf.data.Dataset, the dataset for evaluation.
        :param steps: int, the number of steps to evaluate for. Default is None.
        :return: tuple of (mse, mae, accuracy)
        """
        val_loss = 0
        val_metric = 0
        all_predictions = []
        all_true_scores = []
        for step, (inputs, scores) in enumerate(dataset):
            if steps and step >= steps:
                break
            data = (inputs, scores)
            logs = self.test_step(data)
            val_loss += logs['loss'].numpy()
            val_metric += logs['r2_score']
            all_predictions.extend(logs['predictions'].numpy().flatten())
            all_true_scores.extend(data[1].numpy().flatten())

        mse = mean_squared_error(all_true_scores, all_predictions)
        mae = mean_absolute_error(all_true_scores, all_predictions)
        val_metric /= steps
        val_loss /= steps
        return val_loss, mse, mae, val_metric

    def predict(self, dataset):
        """
        Predict scores for the provided dataset.
        :param dataset: tf.data.Dataset, the dataset for prediction.
        :return: list of predictions
        """
        all_predictions = []
        for inputs in dataset:
            predictions = self.call(inputs)
            all_predictions.extend(predictions.numpy().flatten())
        return all_predictions

    def save_model(self):
        self.checkpoint_manager.save()
        print(f"Model saved at {self.checkpoint_manager.latest_checkpoint}")

    def load_model(self):
        latest_checkpoint = self.checkpoint_manager.latest_checkpoint
        if latest_checkpoint:
            print(f"Restoring model from {latest_checkpoint}")
            self.ckpt.restore(latest_checkpoint).expect_partial()
        else:
            print("Initializing from scratch.")

    def print_weights(self):
        for layer in self.layers:
            print(f"Layer: {layer.name}")
            for weight in layer.weights:
                print(weight.name, weight.numpy())


def extract_level_text(levels: list):
    level_list = []
    for lvl in levels:
        generated_level = SampleOutput.load(lvl)
        level_txt = "\n".join(generated_level.level)
        level_list.append(level_txt)
    return level_list


def convert_to_level_token(levels: list):
    level_list = []
    for lvl in levels:
        generated_level = SampleOutput.load(lvl)
        lst = generated_level.level
        if not lst:
            return ""

        # Initialize the result string
        result = ""

        list_length = len(lst) - 1
        # Get the length of the elements
        element_length = len(lst[0])
        # Iterate over the length of the elements
        for i in range(element_length):
            for j in range(list_length, -1, -1):
                result += lst[j][i]

        level_list.append(result)

    return level_list


def prepare_dataset(df, test_size, batch_size=32):
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[['prompt']])
    print('Train Shape:', train_df.shape)
    print('Test Shape:', test_df.shape)
    train_prompts = train_df['prompt'].astype(str).tolist()
    train_levels = train_df['level_file_path'].astype(str).tolist()
    train_levels_token = convert_to_level_token(train_levels)
    train_scores = train_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()
    val_prompts = test_df['prompt'].astype(str).tolist()
    val_levels = test_df['level_file_path'].astype(str).tolist()
    val_levels_token = convert_to_level_token(val_levels)
    val_scores = test_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()

    '''
    # Concatenate the prompt and response
    train_input_texts = [f"{prompt} {response}" for prompt, response in zip(train_prompts, train_levels_txt)]
    val_input_texts = [f"{prompt} {response}" for prompt, response in zip(val_prompts, val_levels_txt)]

    
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
    '''
    mario_lm = MarioLM()
    tokenized_train_prompts = [mario_lm.prompter.output_hidden(prompt) for prompt in zip(train_prompts)]
    tokenized_val_prompts = [mario_lm.prompter.output_hidden(prompt) for prompt in zip(val_prompts)]
    tokenized_train_levels = [
        mario_lm.tokenizer.encode(lvl_token, return_tensors="pt", truncation=True, max_length=1400) for lvl_token in
        train_levels_token]
    tokenized_val_levels = [mario_lm.tokenizer.encode(lvl_token, return_tensors="pt", truncation=True, max_length=1400)
                            for lvl_token in
                            val_levels_token]

    train_features_tf = [tf.convert_to_tensor(torch.concat((prompt, level), axis=1).numpy())
                         for prompt, level in zip(tokenized_train_prompts, tokenized_train_levels)]
    val_features_tf = [tf.convert_to_tensor(torch.concat((prompt, level), axis=1).numpy())
                       for prompt, level in zip(tokenized_val_prompts, tokenized_val_levels)]

    train_scores_tf = tf.convert_to_tensor(train_scores)
    val_scores_tf = tf.convert_to_tensor(val_scores)

    '''
    train_dataset = (train_features_tf, train_scores_tf)
    val_dataset = (val_features_tf, val_scores_tf)
    '''

    # Create TensorFlow datasets, batch them, and set them to repeat
    train_dataset = tf.data.Dataset.from_tensor_slices((train_features_tf, train_scores_tf)).batch(batch_size).repeat()
    val_dataset = tf.data.Dataset.from_tensor_slices((val_features_tf, val_scores_tf)).batch(batch_size)

    return train_dataset, val_dataset


def trainer(dataset_path, epochs, test_size, batch_size):
    df = pd.read_csv(dataset_path, sep=",")
    train_dataset, val_dataset = prepare_dataset(df, test_size, batch_size)

    train_dataset_size = math.ceil(len(df) * (1 - test_size))
    val_dataset_size = math.ceil(len(df) * test_size)

    steps_per_epoch = train_dataset_size // batch_size
    validation_steps = val_dataset_size // batch_size

    preference_model = PreferenceModel()
    preference_model.fit(train_dataset, val_dataset, num_epochs=1, steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps)
    preference_model.load_model()
    preference_model.summary()
    preference_model.print_weights()
    preference_model.fit(train_dataset, val_dataset, num_epochs=epochs, steps_per_epoch=steps_per_epoch,
                         validation_steps=validation_steps)
    preference_model.print_weights()


if __name__ == "__main__":
    DATASET_PATH = "../sampling/sampling_score.csv"
    NUM_EPOCHS = 100
    TEST_SIZE = 0.25
    BATCH_SIZE = 32

    # Ensure GPU availability
    device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device_name}")

    trainer(DATASET_PATH, NUM_EPOCHS, TEST_SIZE, BATCH_SIZE)
