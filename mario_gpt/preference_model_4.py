"""
Author: Balasubramani Murugan

This script is to train reward model. This reward model is
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from mario_gpt import MarioLM, SampleOutput
from checkpoint_manager import load_checkpoint, save_checkpoint

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"

global LOGGER


class PreferenceModel(nn.Module):
    def __init__(self, model_name=PRETRAINED_MODEL_PATH, context_len=700, lstm_hidden_size=256, num_lstm_layers=2):
        super(PreferenceModel, self).__init__()

        self.mariolm = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        self.chunk_size = context_len

        self.lstm = nn.LSTM(input_size=self.mariolm.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        self.regression = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

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
                                   encoder_hidden_states=encoder_hidden_states)

            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states.squeeze(1)
            all_hidden_states.append(hidden_states)

        hidden_states = torch.cat(all_hidden_states, dim=1)
        lstm_output, (hn, cn) = self.lstm(hidden_states)

        sequence_representation = lstm_output.mean(dim=1)
        logits = self.regression(sequence_representation)
        return logits

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
            predictions = self(input_ids, encoder_hidden_states)
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


def train(model, dataloader, optimizer, criterion, scheduler, device):
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # Update learning rate
        scheduler.step()

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

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(scores.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2score = r2_score(all_labels, all_preds)
    return epoch_loss, mse, mae, r2score


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
    train_scores = train_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()

    val_prompts = test_df['prompt'].astype(str).tolist()
    val_levels = test_df['level_file_path'].astype(str).tolist()
    val_levels_token = convert_to_level_token(val_levels)
    val_scores = test_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()

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

    # Set different learning rates for transformer and newly added layers
    param_optimizer = list(model.named_parameters())

    # Separate parameters for language model and LSTM/Regression layers
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if 'mariolm' in n and not any(nd in n for nd in no_decay)],
         'lr': 5e-5, 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if 'mariolm' in n and any(nd in n for nd in no_decay)],
         'lr': 5e-5, 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if 'lstm' in n],
         'lr': 1e-4, 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if 'regression' in n],
         'lr': 1e-4, 'weight_decay': 0.01}
    ]

    df = pd.read_csv(data_set, sep=",")
    train_dataset, val_dataset = prepare_dataset(df, test_size=0.25)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Define optimizer and loss function
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.MSELoss()

    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(num_training_steps * 0.1)  # Warmup for 10% of training
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

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
        train_loss = train(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss, val_mse, val_mae, r2score = validate(model, val_dataloader, criterion, device)

        LOGGER.info(f"Training Loss: {train_loss:.4f}")
        LOGGER.info(f"Validation Loss: {val_loss:.4f}")
        LOGGER.info(f"Validation MSE: {val_mse:.4f}")
        LOGGER.info(f"Validation MAE: {val_mae:.4f}")
        LOGGER.info(f"Validation R2 Score: {r2score:.4f}")

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


if __name__ == "__main__":
    LOG_FILE_PATH = '../logs/pmv4_training_log.log'
    # Configure logging
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    LOGGER = logging.getLogger('PreferenceModel')

    DATASET_PATH = "../sampling/sampling_score.csv"
    CHECKPOINT_DIR = "../checkpoints/pm_v4/"
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
