import torch
from torch.optim import AdamW
import torch.nn.functional as F
from mario_gpt import MarioLM, SampleOutput

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

    def ppo_loss(self, advantages, rewards, values):
        advantages = advantages.view(1, -1, 1)

        def loss_fn(old_logits, current_logits):
            prob_ratio = torch.exp(torch.log(current_logits + 1e-10) - torch.log(old_logits + 1e-10))
            clipped_ratio = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = prob_ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))
            critic_loss = torch.mean((rewards - values) ** 2)
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

            for prompt in prompts:
                # Generate level with the current policy
                response, response_tensor, current_logits, value = self.finetune_mario_lm.sample(prompts=[prompt],
                                                                                                 num_steps=1400,
                                                                                                 temperature=2.5,
                                                                                                 use_tqdm=True,
                                                                                                 return_tensor=True,
                                                                                                 return_logits=True,
                                                                                                 return_values=True)

                prompt_tensor = self.initial_mario_lm.prompter.output_hidden(prompt)
                prompt_tensor = prompt_tensor.view(prompt_tensor.shape[0], 1, prompt_tensor.shape[1])
                prompt_tensor = prompt_tensor.to(self.initial_mario_lm.device)
                response_tensor = response_tensor[:, 1:]
                response_tensor = response_tensor.to(self.initial_mario_lm.device)

                # Forward pass through the initial models
                initial_logits, _ = self.forward_pass(self.initial_mario_lm, prompt_tensor, response_tensor)

                preferability = 1.0

                reward = self.compute_reward(preferability, initial_logits, current_logits)

                advantage = self.get_advantages(reward, value)

                self.optimizer.zero_grad()
                logits, _ = self.forward_pass(self.finetune_mario_lm, prompt_tensor, response_tensor)
                loss_fn = self.ppo_loss(advantage[0], reward, value)
                loss = loss_fn(initial_logits, logits)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(prompts)}")

            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
                print(f"Model saved at epoch {epoch + 1}")

    def forward_pass(self, model, prompt_tensor, response_tensor):
        with torch.no_grad():  # No gradient computation for the initial model
            context_len = 672
            step_size = 14

            diff = response_tensor.shape[-1] % step_size
            ctx = context_len + diff
            start_idx = 0

            logits = torch.tensor([], device=model.device)
            last_hidden_states = torch.tensor([], device=model.device)

            while start_idx + ctx <= response_tensor.shape[-1]:
                # Extract the portion from the tensor
                chunk_response_tensor = response_tensor[:, start_idx:start_idx + ctx]
                attention_mask = torch.ones(chunk_response_tensor.shape, device=model.device)

                outputs = model.lm(input_ids=chunk_response_tensor,
                                   attention_mask=attention_mask,
                                   encoder_hidden_states=prompt_tensor,
                                   token_type_ids=None,
                                   )

                ilogits = outputs.logits.detach()
                ilogits = ilogits.squeeze(1)
                lhidden_state = outputs.hidden_states[-1]

                if start_idx == 0:
                    logits = torch.cat((logits, ilogits[:, start_idx:, :]), dim=1)
                    last_hidden_states = torch.cat((last_hidden_states, lhidden_state[:, start_idx:, :]), dim=1)

                else:
                    logits = torch.cat((logits, ilogits[:, -step_size:, :]), dim=1)
                    last_hidden_states = torch.cat((last_hidden_states, lhidden_state[:, -step_size:, :]), dim=1)

                start_idx += step_size
        return logits, last_hidden_states

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
    ppo_trainer.train(['many pipes many enemies many blocks high elevation'], 10)


if __name__ == "__main__":
    trainer()
