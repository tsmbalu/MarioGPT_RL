import torch
from torch.optim import AdamW
import torch.nn.functional as F
from mario_gpt import MarioLM, SampleOutput

from value_head import ValueHead


class PPOTrainer:
    def __init__(self, frozen_lm, finetune_lm, preference_model, checkpoint_dir, beta=0.01, clip_ratio=0.2,
                 learning_rate=1e-5):
        self.frozen_mario_lm = frozen_lm
        self.finetune_mario_lm = finetune_lm
        self.preference_model = preference_model
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.optimizer = AdamW(self.finetune_mario_lm.lm.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.value_head = ValueHead(self.finetune_mario_lm.lm.config.hidden_size)

    def compute_kl_divergence(self, current_probs, initial_probs):
        p = torch.clamp(current_probs, min=1e-10, max=1.0)
        q = torch.clamp(initial_probs, min=1e-10, max=1.0)
        kl_div = torch.sum(p * torch.log(p / q), dim=-1)
        return kl_div

    def compute_reward(self, preferability, initial_logits, current_logits, input_ids):
        current_probs = F.softmax(current_logits, dim=-1)
        initial_probs = F.softmax(initial_logits, dim=-1)
        kl_div = self.compute_kl_divergence(current_probs, initial_probs)

        reward = preferability - self.beta * kl_div
        return reward

    def get_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        gae = 0
        advantages_reversed = []
        gen_len = values.shape[-1]

        for i in reversed(range(gen_len)):
            next_values = values[:, i + 1] if i < gen_len - 1 else 0.0
            delta = rewards[:, i] + gamma * next_values - values[:, i]
            gae = delta + gamma * lam * gae
            advantages_reversed.append(gae)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        return advantages

    def logprobs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=2)

        if not gather:
            return logp
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy

    def ppo_loss(self, old_logprobs, current_logprobs, rewards, advantages, values):
        # Compute the probability ratio
        prob_ratio = torch.exp(current_logprobs - old_logprobs)  # shape (batch_size, sequence_length)

        # Compute the clipped ratio
        clipped_ratio = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Compute surrogate losses
        surrogate1 = prob_ratio * advantages  # shape (batch_size, sequence_length)
        surrogate2 = clipped_ratio * advantages  # shape (batch_size, sequence_length)

        # Actor loss: take the minimum of the two surrogates and mean over the batch
        actor_loss = -torch.mean(torch.min(surrogate1, surrogate2))

        # Critic loss: mean squared error between rewards and values
        critic_loss = torch.mean((rewards - values) ** 2)

        return actor_loss + critic_loss

    def train_step(self, prompt):
        # Generate level with the current policy
        response, response_tensor, _, _ = self.finetune_mario_lm.sample(
            prompts=[prompt],
            num_steps=700,
            temperature=2.5,
            use_tqdm=True,
            return_tensor=True,
            return_logits=True,
            return_values=True)

        prompt_tensor = self.frozen_mario_lm.prompter.output_hidden(prompt)
        prompt_tensor = prompt_tensor.view(-1, 1, self.frozen_mario_lm.lm.config.hidden_size)
        prompt_tensor = prompt_tensor.to(self.frozen_mario_lm.device)
        response_tensor = response_tensor[:, 1:].to(self.frozen_mario_lm.device)

        # Forward pass through the initial model
        initial_logits, _ = self.forward_pass(self.frozen_mario_lm, prompt_tensor, response_tensor)

        # Forward pass through the finetune model
        current_logits, values = self.forward_pass(self.finetune_mario_lm, prompt_tensor,
                                                   response_tensor)

        preferability = 0.5
        rewards = self.compute_reward(preferability, initial_logits, current_logits, response_tensor)
        advantages = self.get_advantages(rewards, values)
        #returns = advantages + values

        self.optimizer.zero_grad()
        '''new_logits, v_preds = self.forward_pass(self.finetune_mario_lm, prompt_tensor, response_tensor)
        current_logprobs = self.logprobs_from_logits(current_logits, response_tensor)
        new_logprobs = self.logprobs_from_logits(new_logits, response_tensor)'''
        old_logprobs = self.logprobs_from_logits(initial_logits, response_tensor)
        current_logprobs = self.logprobs_from_logits(current_logits, response_tensor)
        loss = self.ppo_loss(current_logprobs, old_logprobs, advantages, rewards, values)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, prompts, num_epochs=10, save_freq=5):
        self.finetune_mario_lm.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for prompt in prompts:
                loss = self.train_step(prompt)
                total_loss += loss

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(prompts)}")

            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1)
                print(f"Model saved at epoch {epoch + 1}")

    def forward_pass(self, model, prompt_tensor, response_tensor):
        with torch.no_grad():
            context_len = 672
            step_size = 14

            diff = response_tensor.shape[-1] % step_size
            ctx = context_len + diff
            start_idx = 0

            logits = torch.tensor([], device=model.device)
            last_hidden_states = torch.tensor([], device=model.device)

            while start_idx + ctx <= response_tensor.shape[-1]:
                chunk_response_tensor = response_tensor[:, start_idx:start_idx + ctx]
                attention_mask = torch.ones(chunk_response_tensor.shape, device=model.device)

                outputs = model.lm(input_ids=chunk_response_tensor,
                                   attention_mask=attention_mask,
                                   encoder_hidden_states=prompt_tensor,
                                   token_type_ids=None)

                ilogits = outputs.logits.detach().squeeze(1)
                lhidden_state = outputs.hidden_states[-1]

                if start_idx == 0:
                    logits = torch.cat((logits, ilogits[:, start_idx:, :]), dim=1)
                    last_hidden_states = torch.cat((last_hidden_states, lhidden_state[:, start_idx:, :]), dim=1)
                else:
                    logits = torch.cat((logits, ilogits[:, -step_size:, :]), dim=1)
                    last_hidden_states = torch.cat((last_hidden_states, lhidden_state[:, -step_size:, :]), dim=1)

                start_idx += step_size

        self.value_head.to(self.finetune_mario_lm.device)
        values = self.value_head(last_hidden_states)
        values = values.squeeze(-1)

        return logits, values

    def convert_to_level_token(self, lst: list):
        if not lst:
            return ""

        result = ""
        list_length = len(lst) - 1
        element_length = len(lst[0])

        for i in range(element_length):
            for j in range(list_length, -1, -1):
                result += lst[j][i]

        return result

    def save_checkpoint(self, epoch):
        model_save_path = f"{self.checkpoint_dir}/model_epoch_{epoch}.pt"
        torch.save({
            'model_state_dict': self.finetune_mario_lm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, model_save_path)
        print(f"Model checkpoint saved at {model_save_path}")


def trainer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_mario_lm = MarioLM().to(device)
    finetune_mario_lm = MarioLM().to(device)
    preference_model = ''
    checkpoint_dir = '../checkpoints/ppo'
    ppo_trainer = PPOTrainer(initial_mario_lm, finetune_mario_lm, preference_model, checkpoint_dir, 0.01, 0.2)
    ppo_trainer.train(['many pipes many enemies many blocks high elevation'], 10)


if __name__ == "__main__":
    trainer()
