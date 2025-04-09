"""
Reinforcement Learning from Human Feedback (RLHF) module.
This module implements the RLHF training loop for fine-tuning the translation model.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from reward_model import RewardModel
from generator import Generator
from retriever import Retriever
from rag_pipeline import RAGPipeline


class RLHFTrainer:
    """Trainer class for RLHF fine-tuning"""

    def __init__(self, rag_pipeline=None, reward_model_path="models/reward_model.pt"):
        # Initialize RAG pipeline
        self.rag_pipeline = rag_pipeline if rag_pipeline else RAGPipeline()

        # Load the reward model
        self.reward_model = RewardModel()
        self.reward_model.load_state_dict(torch.load(reward_model_path))
        self.reward_model.eval()  # Set to evaluation mode

        # Get the generator from the RAG pipeline
        self.generator = self.rag_pipeline.generator

        # Original model for reference (to calculate KL divergence)
        self.reference_model = AutoModelForSeq2SeqLM.from_pretrained(self.generator.model_name)
        self.reference_model.eval()

        print("RLHF Trainer initialized")

    def generate_candidate_translations(self, text, examples, num_samples=5):
        """Generate multiple candidate translations for a given text"""
        candidates = []

        for _ in range(num_samples):
            # Generate with slight variations (temperature, top_p, etc.)
            inputs = self.generator.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

            outputs = self.generator.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=128,
                do_sample=True,
                temperature=0.8 + np.random.rand() * 0.4,  # temperature between 0.8 and 1.2
                top_p=0.9 + np.random.rand() * 0.1,  # top_p between 0.9 and 1.0
                num_return_sequences=1
            )

            translation = self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(translation)

        return candidates

    def calculate_rewards(self, english_texts, darija_translations):
        """Calculate rewards for translation pairs"""
        rewards = []

        for english, darija in zip(english_texts, darija_translations):
            reward = self.reward_model.get_reward(english, darija)
            rewards.append(reward)

        return rewards

    def train_with_ppo(self, train_data_path='data/train.parquet',
                       num_iterations=100, batch_size=8, lr=1e-5):
        """Train the generator using PPO algorithm"""
        # Load training data
        train_df = pd.read_parquet(train_data_path)
        train_data = train_df.sample(min(len(train_df), num_iterations * batch_size), random_state=42)

        # Setup optimizer
        optimizer = torch.optim.AdamW(self.generator.model.parameters(), lr=lr)

        # PPO parameters
        clip_param = 0.2
        value_loss_coef = 0.1
        entropy_coef = 0.01

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.model.to(device)
        self.reference_model.to(device)
        self.reward_model.to(device)

        print(f"Training with PPO for {num_iterations} iterations on {device}...")

        for iteration in range(num_iterations):
            # Sample batch
            batch = train_data.iloc[iteration * batch_size:(iteration + 1) * batch_size]

            # Process each example
            for _, row in tqdm(batch.iterrows(), total=len(batch),
                               desc=f"Iteration {iteration + 1}/{num_iterations}"):
                english_text = row['translation']
                reference_darija = row['sentence']

                # Get retrieved examples from RAG
                retrieved_docs = self.rag_pipeline.retriever.retrieve(
                    english_text, index_type='combined', top_k=3
                )

                # Generate candidate translations
                candidates = self.generate_candidate_translations(
                    english_text, retrieved_docs, num_samples=3
                )

                # Calculate rewards for all candidates
                candidate_rewards = self.calculate_rewards(
                    [english_text] * len(candidates), candidates
                )

                # Calculate reward for reference translation
                reference_reward = self.reward_model.get_reward(english_text, reference_darija)

                # Find best candidate
                best_idx = np.argmax(candidate_rewards)
                best_candidate = candidates[best_idx]
                best_reward = candidate_rewards[best_idx]

                # PPO training step (if candidate is better than reference)
                if best_reward > reference_reward:
                    # Process the winning candidate
                    inputs = self.generator.tokenizer(english_text, return_tensors="pt").to(device)
                    target_ids = self.generator.tokenizer(best_candidate, return_tensors="pt").input_ids.to(device)

                    # Forward pass
                    outputs = self.generator.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        labels=target_ids
                    )

                    # Calculate policy loss
                    policy_loss = outputs.loss

                    # Get reference model loss for KL divergence
                    with torch.no_grad():
                        ref_outputs = self.reference_model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            labels=target_ids
                        )
                        ref_loss = ref_outputs.loss

                    # KL divergence between current and reference policy
                    kl_div = policy_loss - ref_loss

                    # Clipped objective
                    ratio = torch.exp(ref_loss - policy_loss)
                    clipped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

                    # Calculate final loss
                    advantage = torch.tensor(best_reward - reference_reward).to(device)
                    policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

                    # Add entropy bonus
                    entropy = torch.distributions.Categorical(
                        logits=outputs.logits
                    ).entropy().mean()

                    total_loss = policy_loss - entropy_coef * entropy

                    # Update model
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            # Validate and save periodically
            if (iteration + 1) % 10 == 0 or iteration == num_iterations - 1:
                self.save_model(f"models/rlhf_model_iter{iteration + 1}.pt")
                print(f"Model saved at iteration {iteration + 1}")

        # Save final model
        self.save_model("models/rlhf_final_model.pt")
        print("RLHF training completed. Final model saved.")

    def save_model(self, path):
        """Save the fine-tuned generator model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.generator.model.state_dict(), path)


if __name__ == "__main__":
    # Make sure reward model is trained first
    if not os.path.exists("models/reward_model.pt"):
        print("Training reward model first...")
        from reward_model import train_reward_model

        train_reward_model(num_epochs=1)  # Use more for full training

    # Train with RLHF
    rlhf_trainer = RLHFTrainer()
    rlhf_trainer.train_with_ppo(num_iterations=5)  # Use more for full training