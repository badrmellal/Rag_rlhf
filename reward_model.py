"""
Reward model for Reinforcement Learning from Human Feedback (RLHF).
This module implements a model that predicts the quality of translations.
"""

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm


class RewardModel(nn.Module):
    """Neural network model that predicts the quality of translations"""

    def __init__(self, model_name="bert-base-multilingual-cased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        """Forward pass through the model"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        reward = self.reward_head(outputs.last_hidden_state[:, 0, :])
        return reward

    def get_reward(self, english_text, darija_text):
        """Calculate reward for a given translation pair"""
        # Combine texts with a separator
        combined = f"{english_text} [SEP] {darija_text}"

        # Tokenize
        inputs = self.tokenizer(combined, return_tensors="pt",
                                max_length=512, truncation=True, padding="max_length")

        # Get reward prediction
        with torch.no_grad():
            reward = self(inputs.input_ids, inputs.attention_mask)

        return reward.item()


def train_reward_model(train_data_path='data/train.parquet', val_data_path='data/val.parquet',
                       model_name="bert-base-multilingual-cased", num_epochs=3, batch_size=16):
    """
    Train a reward model using the quality field from the dataset

    Parameters:
    - train_data_path: Path to training data
    - val_data_path: Path to validation data
    - model_name: Base model to fine-tune
    - num_epochs: Number of training epochs
    - batch_size: Batch size for training

    Returns:
    - Trained reward model
    """
    # Load dataset
    train_df = pd.read_parquet(train_data_path)
    val_df = pd.read_parquet(val_data_path)

    # Initialize model and tokenizer
    model = RewardModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare training data
    train_inputs = []
    train_labels = []

    for _, row in train_df.iterrows():
        english = row['translation']
        darija = row['sentence']
        quality = row['quality']  # Assuming quality is in range 0-1

        combined = f"{english} [SEP] {darija}"
        train_inputs.append(combined)
        train_labels.append(quality)

    # Tokenize inputs
    train_encodings = tokenizer(train_inputs, truncation=True, padding="max_length",
                                max_length=512, return_tensors="pt")
    train_dataset = torch.utils.data.TensorDataset(
        train_encodings.input_ids,
        train_encodings.attention_mask,
        torch.tensor(train_labels, dtype=torch.float)
    )

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    val_inputs = []
    val_labels = []

    for _, row in val_df.iterrows():
        english = row['translation']
        darija = row['sentence']
        quality = row['quality']

        combined = f"{english} [SEP] {darija}"
        val_inputs.append(combined)
        val_labels.append(quality)

    val_encodings = tokenizer(val_inputs, truncation=True, padding="max_length",
                              max_length=512, return_tensors="pt")
    val_dataset = torch.utils.data.TensorDataset(
        val_encodings.input_ids,
        val_encodings.attention_mask,
        torch.tensor(val_labels, dtype=torch.float)
    )

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.MSELoss()

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Training reward model on {device}...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1} - Train loss: {avg_train_loss:.4f}, Val loss: {avg_val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "models/reward_model.pt")
    print("Reward model saved to 'models/reward_model.pt'")

    return model


if __name__ == "__main__":
    import os

    os.makedirs("models", exist_ok=True)
    train_reward_model(num_epochs=1)  # Use more epochs for full training