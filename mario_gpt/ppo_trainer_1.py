import torch
from torch.optim import AdamW
import torch.nn.functional as F
from mario_gpt import MarioLM, SampleOutput
from transformers import AutoModelForCausalLM, AutoTokenizer

from value_head import ValueHead
from preference_model_1 import PreferenceModel


class PPOTrainer:
    def __init__(self, initial_mario_lm, finetune_mario_lm, preference_model, checkpoint_dir, beta=0.01, clip_ratio=0.2,
                 learning_rate=1e-5):
        self.initial_mario_lm = initial_mario_lm
        self.finetune_mario_lm = finetune_mario_lm
        self.preference_model = preference_model
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.optimizer = AdamW(self.finetune_mario_lm.lm.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir

    def compute_kl_divergence(self, current_probs, initial_probs):
        p = torch.clamp(current_probs, min=1e-10, max=1.0)
        q = torch.clamp(initial_probs, min=1e-10, max=1.0)
        kl_div = torch.sum(p * torch.log(p / q), dim=-1)
        return kl_div

    def compute_reward(self, preferability, initial_logits, current_logits):
        current_probs = F.softmax(current_logits, dim=-1)
        initial_probs = F.softmax(initial_logits, dim=-1)
        kl_div = self.compute_kl_divergence(current_probs, initial_probs)

        reward = preferability - self.beta * kl_div
        return reward

    def ppo_loss(self, advantages, rewards, old_values):
        def loss_fn(y_true, y_pred):
            prob_ratio = torch.exp(torch.log(y_pred + 1e-10) - torch.log(y_true + 1e-10))
            clipped_ratio = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = prob_ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            critic_loss = torch.mean((rewards - old_values) ** 2)
            return actor_loss + critic_loss

        return loss_fn

    def convert_to_level_token(self, lst: list):
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

        return result

    def train(self, prompts, num_epochs=10, save_freq=5):
        self.finetune_mario_lm.train()
        for epoch in range(num_epochs):
            total_loss = 0
            all_rewards, all_values, all_advantages = [], [], []

            for prompt in prompts:
                # Generate level with the current policy
                response = self.finetune_mario_lm.sample(prompts=[prompt], num_steps=1400,
                                                         temperature=2.5, use_tqdm=True)

                # prompt_tensor = self.initial_mario_lm.prompter.output_hidden(prompt)
                # response_tensor = response.level_tensor
                # prompt_tensor = prompt_tensor.to(self.finetune_mario_lm.device)
                # response_tensor = response_tensor.to(self.finetune_mario_lm.device)

                combined_input = prompt + ' ' + self.convert_to_level_token(response.level)

                # Combine the prompt and response tensors for input
                input_ids = self.finetune_mario_lm.tokenizer.encode(combined_input, return_tensors="pt",
                                                                    truncation=True, max_length=700)
                input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(self.finetune_mario_lm.device)

                attention_mask = torch.ones(input_ids.shape, device=self.finetune_mario_lm.device)

                # Forward pass through the initial and fine-tuned models
                with torch.no_grad():  # No gradient computation for the initial model
                    initial_outputs = self.initial_mario_lm.lm(input_ids=input_ids, attention_mask=attention_mask)
                    initial_logits = initial_outputs.logits  # Get the logits for the initial model

                    # Forward pass through the fine-tuned model
                    current_outputs = self.finetune_mario_lm.lm(input_ids=input_ids, attention_mask=attention_mask)
                    current_logits = current_outputs.logits  # Get the logits for the fine-tuned model
                    last_hidden_states = current_outputs.hidden_states[-1]
                    last_hidden_states = last_hidden_states.squeeze(0)
                    value_head = ValueHead(last_hidden_states.shape[2])
                    value_head.to(self.finetune_mario_lm.device)
                    values = value_head(last_hidden_states)

                preferability = 1.0

                reward = self.compute_reward(preferability, initial_logits, current_logits)

                all_rewards.append(reward)
                all_values.append(values)

            advantages = self.get_advantages(all_rewards, all_values)
            all_advantages.extend(advantages)

            for input_ids, (reward, advantage, value) in zip(dataset, zip(all_rewards, all_advantages, all_values)):
                input_ids = input_ids.to(self.finetune_mario_lm.device)
                advantage = advantage.to(self.finetune_mario_lm.device)
                reward = reward.to(self.finetune_mario_lm.device)

                self.optimizer.zero_grad()
                logits = self.finetune_mario_lm(input_ids).logits
                loss_fn = self.ppo_loss(advantage, reward, value)
                loss = loss_fn(logits, logits)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")

            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
                print(f"Model saved at epoch {epoch + 1}")

    def get_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
            gae = delta
            gamma * lam * gae

            advantages.insert(0, gae)
        return advantages

    def save_checkpoint(self, epoch):
        model_save_path = f"{self.checkpoint_dir}/model_epoch_{epoch}.pt"
        torch.save(self.finetune_mario_lm.state_dict(), model_save_path)
        print(f"Model checkpoint saved at {model_save_path}")


def trainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_mario_lm = MarioLM()
    finetune_mario_lm = MarioLM()
    initial_mario_lm.to(device)
    finetune_mario_lm.to(device)
    preference_model = PreferenceModel()
    checkpoint_dir = '../checkpoints/ppo'
    ppo_trainer = PPOTrainer(initial_mario_lm, finetune_mario_lm, preference_model, checkpoint_dir, 0.01, 0.2)
    ppo_trainer.train(['many pipes many enemies many blocks high elevation'], 1)


if __name__ == "__main__":
    trainer()
