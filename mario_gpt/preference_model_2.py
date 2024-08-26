import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from mario_gpt import MarioLM, SampleOutput

PRETRAINED_MODEL_PATH = "shyamsn97/Mario-GPT2-700-context-length"


class PreferenceModel(nn.Module):
    def __init__(self, model_name=PRETRAINED_MODEL_PATH, chunk_size=700):
        super(PreferenceModel, self).__init__()
        self.mariolm = AutoModelForCausalLM.from_pretrained(model_name, **{"add_cross_attention": True,
                                                                           "output_hidden_states": True})
        self.chunk_size = chunk_size
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


class PreferenceDataset(Dataset):
    def __init__(self, input_ids: torch.LongTensor, encoder_hidden_states: torch.FloatTensor, scores: torch.FloatTensor):
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
    for batch in dataloader:
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
    with torch.no_grad():
        for batch in dataloader:
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
    return epoch_loss, mse


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
    # Hyperparameters
    batch_size = 4
    learning_rate = 1e-5
    epochs = 3

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

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_mse = validate(model, val_dataloader, criterion, device)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")

