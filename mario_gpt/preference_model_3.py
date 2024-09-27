import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import logging
import shutil
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split

from mario_gpt import MarioLM, SampleOutput
from checkpoint_manager import load_checkpoint, save_checkpoint

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"

global LOGGER


class PreferenceModel(nn.Module):
    def __init__(self, model_name=PRETRAINED_MODEL_PATH, context_len=700):
        super(PreferenceModel, self).__init__()
        self.mariolm = AutoModelForCausalLM.from_pretrained(model_name, **{"add_cross_attention": True,
                                                                           "output_hidden_states": True})
        self.chunk_size = context_len
        self.classification = nn.Sequential(
            nn.Linear(self.mariolm.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
        self.dropout = nn.Dropout(0.1)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, encoder_hidden_states):
        seq_len = input_ids.shape[-1]
        chunk_size = self.chunk_size
        num_chunks = seq_len // chunk_size

        all_hidden_states = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            chunk_input_ids = input_ids[:, :, start_idx:end_idx]
            chunk_attention_mask = torch.ones_like(chunk_input_ids)

            outputs = self.mariolm(input_ids=chunk_input_ids,
                                   attention_mask=chunk_attention_mask,
                                   encoder_hidden_states=encoder_hidden_states,
                                   token_type_ids=None,
                                   )
            hidden_states = outputs.hidden_states[-1]
            all_hidden_states.append(hidden_states)

            # Concatenate all hidden states
        hidden_states = torch.cat(all_hidden_states, dim=1)

        # Aggregate hidden states for the entire sequence
        sequence_representation = hidden_states.mean(dim=1)
        logits = self.classification(self.dropout(sequence_representation))
        logits = logits.permute(0, 2, 1)  # (batch_size, 3, full_seq_len)
        pooled_logits = self.pooling(logits)  # (batch_size, 3, 1)
        probabilities = pooled_logits.squeeze(-1)
        preference_score = self.sigmoid(probabilities)
        return preference_score

    def predict(self, input_ids, encoder_hidden_states, device=None):
        """
        Predict preference scores for given input_ids and encoder_hidden_states.

        :param input_ids: torch.Tensor, shape [batch_size, seq_len]
        :param encoder_hidden_states: torch.Tensor, shape [batch_size, seq_len, hidden_size]
        :param device: torch.device, device to run the prediction on
        :return: torch.Tensor, shape [batch_size, 3]
        """
        if device:
            self.to(device)
            input_ids = input_ids.to(device)
            encoder_hidden_states = encoder_hidden_states.to(device)

        self.eval()
        with torch.no_grad():
            probabilities = self(input_ids, encoder_hidden_states)
            predictions = (probabilities > 0.8).int()
        return predictions


class PreferenceDataset(Dataset):
    def __init__(self, input_ids: torch.LongTensor, encoder_hidden_states: torch.FloatTensor,
                 scores: torch.FloatTensor):
        self.input_ids = input_ids
        self.encoder_hidden_states = encoder_hidden_states
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'encoder_hidden_states': self.encoder_hidden_states[idx],
            'scores': self.scores[idx]
        }


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    # Wrap the dataloader with tqdm for a progress bar
    for batch in tqdm(dataloader, desc="Training", total=total_batches):
        input_ids = batch['input_ids'].to(device)
        encoder_hidden_states = batch['encoder_hidden_states'].to(device)
        scores = batch['scores'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, encoder_hidden_states)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * input_ids.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_batches = len(dataloader)

    # Wrap the dataloader with tqdm for a progress bar
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", total=total_batches):
            input_ids = batch['input_ids'].to(device)
            encoder_hidden_states = batch['encoder_hidden_states'].to(device)
            scores = batch['scores'].to(device)

            outputs = model(input_ids, encoder_hidden_states)
            loss = criterion(outputs, scores)
            running_loss += loss.item() * input_ids.size(0)

            preds = (outputs > 0.8).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(scores.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, mse, mae, accuracy


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


def tokenize_prompt(mario_lm, prompts: list, cache: dict) -> (list, dict):
    tokenized_prompts = []
    for prompt in prompts:
        if prompt not in cache:
            # If the prompt is not in the cache, process and store the result
            cache[prompt] = mario_lm.prompter.output_hidden(prompt)
        # Append the cached result to the list
        tokenized_prompts.append(cache[prompt])
    return tokenized_prompts, cache


def prepare_dataset(df, test_size):
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[['prompt']])
    LOGGER.info(f'Train Shape: {train_df.shape}')
    LOGGER.info(f'Test Shape: {test_df.shape}')
    train_prompts = train_df['prompt'].astype(str).tolist()
    train_levels = train_df['level_file_path'].astype(str).tolist()
    train_levels_token = convert_to_level_token(train_levels)
    train_scores = np.round(train_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']]).astype(int).to_numpy()

    val_prompts = test_df['prompt'].astype(str).tolist()
    val_levels = test_df['level_file_path'].astype(str).tolist()
    val_levels_token = convert_to_level_token(val_levels)
    val_scores = np.round(test_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']]).astype(int).to_numpy()

    mario_lm = MarioLM()
    prompt_cache = {}
    tokenized_train_prompts, prompt_cache = tokenize_prompt(mario_lm, train_prompts, prompt_cache)
    tokenized_val_prompts, prompt_cache = tokenize_prompt(mario_lm, val_prompts, prompt_cache)

    tokenized_train_levels = [
        mario_lm.tokenizer.encode(lvl_token, return_tensors="pt", truncation=True, max_length=1400) for lvl_token in
        train_levels_token]
    tokenized_val_levels = [mario_lm.tokenizer.encode(lvl_token, return_tensors="pt", truncation=True, max_length=1400)
                            for lvl_token in
                            val_levels_token]
    train_scores = [torch.tensor(row).float() for row in train_scores]
    val_scores = [torch.tensor(row).float() for row in val_scores]

    train_dataset = PreferenceDataset(tokenized_train_levels, tokenized_train_prompts, train_scores)
    val_dataset = PreferenceDataset(tokenized_val_levels, tokenized_val_prompts, val_scores)
    return train_dataset, val_dataset


def run_preference_trainer(data_set: str, checkpoint_dir: str, learning_rate: float, batch_size: int, epochs: int,
                           early_stop_patience: int, min_epochs: int):
    global device
    # Initialize model
    model = PreferenceModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv(data_set, sep=",")
    train_dataset, val_dataset = prepare_dataset(df, test_size=0.25)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.classification.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    # Log the start of training
    start_time = datetime.now()
    LOGGER.info(f"Training started at {start_time}")
    # Load the latest checkpoint if available
    start_epoch, best_val_loss = load_checkpoint(checkpoint_dir, model, optimizer)
    # Training loop
    epochs_no_improve = 0
    for epoch in range(start_epoch, epochs):
        curr_epoch = epoch + 1
        LOGGER.info(f"Epoch {curr_epoch}/{epochs}")
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_mse, val_mae, accuracy = validate(model, val_dataloader, criterion, device)

        LOGGER.info(f"Training Loss: {train_loss:.4f}")
        LOGGER.info(f"Validation Loss: {val_loss:.4f}")
        LOGGER.info(f"Validation MSE: {val_mse:.4f}")
        LOGGER.info(f"Validation MAE: {val_mae:.4f}")
        LOGGER.info(f"Validation Accuracy Score: {accuracy:.4f}")

        save_checkpoint(model, optimizer, curr_epoch, val_loss, checkpoint_dir=checkpoint_dir, prefix='pmv2_')
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            LOGGER.info(f'No improvement in loss for the last {epochs_no_improve} epochs')

        # Early stopping: only consider stopping after `min_epochs` have passed
        if curr_epoch >= min_epochs and epochs_no_improve >= early_stop_patience:
            LOGGER.info(f"Early stopping at epoch {curr_epoch}")
            break
    # Log the end of training
    end_time = datetime.now()
    LOGGER.info(f"Training ended at {end_time}")
    LOGGER.info(f"Total training time: {end_time - start_time}")


def find_lr(dataset_path, batch_size, beta=0.98):
    df = pd.read_csv(dataset_path, sep=",")
    train_dataset, val_dataset = prepare_dataset(df, test_size=0.25)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define learning rate increments
    increments = [
        0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1
    ]
    results = []
    losses = []
    log_lrs = []

    for lr in increments:
        model = PreferenceModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        optimizer.param_groups[0]['lr'] = lr

        loss = train(model, train_dataloader, optimizer, criterion, device)

        log_lrs.append(lr)  # Use the initial learning rate for the log scale
        losses.append(loss)

    # Save the plot
    plt.plot(log_lrs, losses)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Average Loss')
    plt.title(f'Learning Rate Finder')
    plt.savefig('../logs/lr_plot.png')
    plt.close()
    LOGGER.info(f"Plot saved at ../logs/lr_plot.png")

    results.extend(zip(log_lrs, losses))
    # Save results to CSV
    df_results = pd.DataFrame(results, columns=['Log Learning Rate', 'Average Loss'])
    csv_path = '../logs/lr_results.csv'
    df_results.to_csv(csv_path, index=False)
    LOGGER.info(f"Results saved to {csv_path}")

    return results


if __name__ == "__main__":
    LOG_FILE_PATH = '../logs/pmv3_training_log.log'
    # Configure logging
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    LOGGER = logging.getLogger('PreferenceModel')

    DATASET_PATH = "../sampling/sampling_score.csv"
    CHECKPOINT_DIR = "../checkpoints/pm_v3/"
    # Hyperparameters
    BATCH_SIZE = 4
    # LEARNING_RATE = 1e-5
    LEARNING_RATE = 0.0005
    EPOCHS = 500
    EARLY_STOP_PATIENCE = 10  # Stop if validation loss doesn't improve for 5 epochs
    MIN_EPOCHS = 50  # Minimum number of epochs before early stopping is considered

    # find_lr(DATASET_PATH, BATCH_SIZE)
    run_preference_trainer(DATASET_PATH, CHECKPOINT_DIR, LEARNING_RATE, BATCH_SIZE, EPOCHS, EARLY_STOP_PATIENCE,
                           MIN_EPOCHS)