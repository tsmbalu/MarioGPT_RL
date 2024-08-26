import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import logging

from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from mario_gpt import MarioLM, SampleOutput

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"

LOG_FILE_PATH = '../logs/pmv2_training_log.log'
# Configure logging
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PreferenceModel(nn.Module):
    def __init__(self, model_name=PRETRAINED_MODEL_PATH, context_len=700):
        super(PreferenceModel, self).__init__()
        self.mariolm = AutoModelForCausalLM.from_pretrained(model_name, **{"add_cross_attention": True,
                                                                           "output_hidden_states": True})
        self.chunk_size = context_len
        self.regression = nn.Linear(self.mariolm.config.hidden_size, 3)
        self.dropout = nn.Dropout(0.1)
        self.pooling = nn.AdaptiveAvgPool1d(1)

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
        logits = self.regression(self.dropout(sequence_representation))
        logits = logits.permute(0, 2, 1)  # (batch_size, 3, full_seq_len)
        pooled_logits = self.pooling(logits)  # (batch_size, 3, 1)
        preference_score = pooled_logits.squeeze(-1)
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

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            preference_scores = self(input_ids, encoder_hidden_states)
        return preference_scores

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

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(scores.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2score = r2_score(all_labels, all_preds)
    return epoch_loss, mse, mae, r2score


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        logging.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
        return start_epoch, best_val_loss
    else:
        logging.info("No checkpoint found, starting from scratch.")
        return 0, float('inf')


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


def prepare_dataset(df, test_size):
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[['prompt']])
    logging.info(f'Train Shape: {train_df.shape}')
    logging.info(f'Test Shape: {test_df.shape}')
    train_prompts = train_df['prompt'].astype(str).tolist()
    train_levels = train_df['level_file_path'].astype(str).tolist()
    train_levels_token = convert_to_level_token(train_levels)
    train_scores = train_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()
    val_prompts = test_df['prompt'].astype(str).tolist()
    val_levels = test_df['level_file_path'].astype(str).tolist()
    val_levels_token = convert_to_level_token(val_levels)
    val_scores = test_df[['normalized_entropy', 'normalized_playability', 'normalized_aesthetic']].to_numpy()

    mario_lm = MarioLM()
    tokenized_train_prompts = [mario_lm.prompter.output_hidden(prompt) for prompt in zip(train_prompts)]
    tokenized_val_prompts = [mario_lm.prompter.output_hidden(prompt) for prompt in zip(val_prompts)]
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


if __name__ == "__main__":
    DATASET_PATH = "../sampling/sampling_score.csv"
    CHECKPOINT_PATH = "../checkpoints/pm_v2/pmv2_checkpoint.pth"
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-5
    epochs = 500
    early_stop_patience = 100  # Stop if validation loss doesn't improve for 10 epochs

    # Initialize model
    model = PreferenceModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    df = pd.read_csv(DATASET_PATH, sep=",")
    train_dataset, val_dataset = prepare_dataset(df, test_size=0.25)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Log the start of training
    start_time = datetime.now()
    logging.info(f"Training started at {start_time}")

    # Load checkpoint if available
    start_epoch, best_val_loss = load_checkpoint(CHECKPOINT_PATH, model, optimizer)

    # Training loop
    epochs_no_improve = 0
    for epoch in range(start_epoch, epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_mse, val_mae, r2score = validate(model, val_dataloader, criterion, device)

        logging.info(f"Training Loss: {train_loss:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}")
        logging.info(f"Validation MSE: {val_mse:.4f}")
        logging.info(f"Validation MAE: {val_mae:.4f}")
        logging.info(f"Validation R2 Score: {r2score:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path=CHECKPOINT_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Log the end of training
    end_time = datetime.now()
    logging.info(f"Training ended at {end_time}")
    logging.info(f"Total training time: {end_time - start_time}")
