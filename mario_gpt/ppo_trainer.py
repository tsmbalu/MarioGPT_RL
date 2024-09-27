"""
Author: Balasubramani Murugan

This script is to fine tune the MarioGPT model using the PPO.
"""
import torch
from torch.optim import AdamW
import torch.nn.functional as F
import logging

from tqdm import tqdm
from mario_gpt import MarioLM, SampleOutput
from mario_gpt import ValueHead
from preference_model_4 import PreferenceModel
from checkpoint_manager import load_model, save_checkpoint, load_checkpoint
from playability_measure import measure_playability, run_astar_agent_simulator, normalize_playability_score
from shannon_entropy import rate_novelty_of_level, normalize_score as vnormalize_score
from aesthetic_evaluator import evaluate_aesthetic, normalize_score as anormalize_score

global LOGGER


class PPOTrainer:
    def __init__(self, frozen_lm, finetune_lm, preference_model, checkpoint_dir, beta=0.01, clip_ratio=0.2,
                 learning_rate=1e-5):
        """

        @param frozen_lm:
        @param finetune_lm:
        @param preference_model:
        @param checkpoint_dir:
        @param beta:
        @param clip_ratio:
        @param learning_rate:
        """
        self.frozen_mario_lm = frozen_lm
        self.finetune_mario_lm = finetune_lm
        self.preference_model = preference_model
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.optimizer = AdamW(self.finetune_mario_lm.lm.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir
        self.value_head = ValueHead(self.finetune_mario_lm.lm.config.hidden_size).to(self.finetune_mario_lm.lm.device)

    def forward_pass(self, model, prompt_tensor, response_tensor):
        """

        @param model:
        @param prompt_tensor:
        @param response_tensor:
        @return:
        """
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

        values = self.value_head(last_hidden_states).squeeze(-1)
        return logits, values

    def compute_reward(self, preferability, initial_logits, current_logits):
        """

        @param preferability:
        @param initial_logits:
        @param current_logits:
        @return:
        """
        current_probs = F.softmax(current_logits, dim=-1)
        initial_probs = F.softmax(initial_logits, dim=-1)
        kl_div = compute_kl_divergence(current_probs, initial_probs)
        reward = preferability - self.beta * kl_div
        return reward

    def get_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """

        @param rewards:
        @param values:
        @param gamma:
        @param lam:
        @return:
        """
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

    def ppo_loss(self, old_logprobs, current_logprobs, rewards, advantages, values):
        """

        @param old_logprobs:
        @param current_logprobs:
        @param rewards:
        @param advantages:
        @param values:
        @return:
        """
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

    def train_step(self, prompts):
        """

        @param prompts:
        @return:
        """
        total_reward = 0
        LOGGER.info(f"Starting train step with {len(prompts)} prompts.")
        # Generate level with the current policy
        response, response_tensor, _, _ = self.finetune_mario_lm.sample(
            prompts=prompts,
            num_steps=1400,
            temperature=2.5,
            use_tqdm=True,
            return_tensor=True,
            return_logits=True,
            return_values=True)

        response_tensor = response_tensor[:, 1:].to(self.frozen_mario_lm.device)

        all_frozen_logits = []
        all_current_logits = []
        all_values = []
        preference_scores = None

        for i, prompt in enumerate(prompts):
            LOGGER.info(f"Processing prompt {i + 1}/{len(prompts)}.")
            # Prepare the prompt tensor for each individual prompts
            tokenized_prompt = self.frozen_mario_lm.prompter.output_hidden(prompt)
            prompt_tensor = tokenized_prompt.unsqueeze(1)
            prompt_tensor = prompt_tensor.to(self.frozen_mario_lm.device)

            # Forward pass through the initial model
            frozen_logits, _ = self.forward_pass(self.frozen_mario_lm, prompt_tensor, response_tensor[i].unsqueeze(0))
            all_frozen_logits.append(frozen_logits)

            # Forward pass through the fine-tuned model
            current_logits, values = self.forward_pass(self.finetune_mario_lm, prompt_tensor,
                                                       response_tensor[i].unsqueeze(0))
            all_current_logits.append(current_logits)
            all_values.append(values)

            '''
            # Compute the scores for novelty, playability and aesthetic using the evaluator function 
            level_txt = "\n".join(response[i].level)
            tiles_to_consider = '?SQ[]E'
            v_score = rate_novelty_of_level(level_txt, tiles_to_consider)
            v_score = vnormalize_score(v_score)
            simulator_resp = get_a_star_response(response[i])
            p_score = measure_playability(simulator_resp)
            p_score = normalize_playability_score(p_score)
            a_score = evaluate_aesthetic(level_txt)
            a_score = anormalize_score(a_score)

            pref_score = torch.tensor([v_score, p_score, a_score])
            pref_score = pref_score.unsqueeze(0)
            if preference_scores is None:
                preference_scores = pref_score
            else:
                preference_scores = torch.cat((preference_scores, pref_score), dim=0)
            '''

        # Combine all logits and values into batch tensors
        frozen_logits = torch.cat(all_frozen_logits, dim=0)
        current_logits = torch.cat(all_current_logits, dim=0)
        values = torch.cat(all_values, dim=0)

        preference_scores = self.preference_model.predict(response_tensor.unsqueeze(1), prompt_tensor,
                                                          self.finetune_mario_lm.device)

        preference_scores = preference_scores.to(self.finetune_mario_lm.device)
        preference_weights = torch.tensor([0.2, 0.6, 0.2]).to(self.finetune_mario_lm.device)
        # Multiply each score by its corresponding weight
        weighted_score = preference_scores * preference_weights
        preferability = weighted_score.sum(dim=1, keepdim=True)

        rewards = self.compute_reward(preferability, frozen_logits, current_logits)
        advantages = self.get_advantages(rewards, values)

        avg_rewards = rewards.mean(dim=1)  # Average over sequence length
        avg_reward_per_batch = avg_rewards.mean().item()  # Scalar value for logging
        total_reward += avg_reward_per_batch

        LOGGER.info(f"Average Reward per Batch: {avg_reward_per_batch}")

        self.optimizer.zero_grad()
        old_logprobs = logprobs_from_logits(frozen_logits, response_tensor)
        current_logprobs = logprobs_from_logits(current_logits, response_tensor)
        loss = self.ppo_loss(current_logprobs, old_logprobs, advantages, rewards, values)
        loss.backward()
        self.optimizer.step()

        LOGGER.info(f"Train step completed with loss: {loss.item()}")
        return loss.item(), total_reward

    def train(self, prompts, num_epochs=10, batch_size=4, save_freq=5):
        """
        This method performs multiple training epochs over a set of prompts, optimizing the model
        based on rewards and loss values. The training process is checkpointed at regular intervals
        based on the `save_freq` parameter, ensuring progress can be saved and resumed.

        @param prompts: List of prompts
        @param num_epochs: Number of epochs
        @param batch_size: Batch Size
        @param save_freq: Frequence interval in which the model as to be saved
        @return:
        """
        start_epoch, reward = load_checkpoint(self.checkpoint_dir, self.finetune_mario_lm.lm, self.optimizer)
        LOGGER.info("Starting training.....")
        self.finetune_mario_lm.train()
        if reward == float('inf'):
            total_reward = 0
        else:
            total_reward = reward

        for epoch in tqdm(range(start_epoch, num_epochs), desc='Training Epochs', unit='epoch'):
            curr_epoch = epoch + 1
            total_loss = 0
            LOGGER.info(f"Starting epoch {curr_epoch}/{num_epochs}.")
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                loss, reward = self.train_step(batch_prompts)
                total_reward += reward
                total_loss += loss

            loss = total_loss / len(prompts)
            reward = total_reward / len(prompts)
            LOGGER.info(f"Epoch {curr_epoch}, Loss: {loss}: Reward: {reward}")

            if curr_epoch % save_freq == 0:
                save_checkpoint(self.finetune_mario_lm.lm, self.optimizer, curr_epoch, reward, max_to_keep=5,
                                checkpoint_dir=self.checkpoint_dir, best_model_metric='high')


def compute_kl_divergence(current_probs, initial_probs):
    """
    Compute KL Divergence

    @param current_probs: Finetuning MarioGPT model probs
    @param initial_probs: Existing MarioGPT model probs
    @return:
    """
    p = torch.clamp(current_probs, min=1e-10, max=1.0)
    q = torch.clamp(initial_probs, min=1e-10, max=1.0)
    kl_div = torch.sum(p * torch.log(p / q), dim=-1)
    return kl_div


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    This function is to calculate the log probabilities from the logits

    @param logits:
    @param labels:
    @param gather:
    @return:
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def run_ppo_trainer(preference_model_path, mario_model_checkpoint_dir, training_prompts, num_of_epochs):
    """
    This function is the starting point of the ppo training of MarioGPT model

    @param preference_model_path: Path of the reward model checkpoint
    @param mario_model_checkpoint_dir: Path of the fine-tuned MarioGPT model checkpoint
    @param training_prompts: List of prompts that will be used for PPO training
    @param num_of_epochs: Number of epochs
    @return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_mario_lm = MarioLM().to(device)
    finetune_mario_lm = MarioLM().to(device)
    preference_model = PreferenceModel()
    preference_model.to(device)
    preference_model = load_model(preference_model, preference_model_path)

    print(preference_model)
    '''
    for name, param in finetune_mario_lm.lm.named_parameters():
        print(f"Name: {name}, Size: {param.size()}, Requires Grad: {param.requires_grad}")
    '''

    # Freeze all parameters
    for param in initial_mario_lm.lm.parameters():
        param.requires_grad = False

    # Freeze all parameters
    for param in finetune_mario_lm.lm.parameters():
        param.requires_grad = False

    # Fine-tune specific parameters (e.g., last 2 transformer layers and layer normalization)
    for name, param in finetune_mario_lm.lm.named_parameters():
        if 'h.4' in name or 'h.5' in name or 'ln_f' in name:
            param.requires_grad = True

    ppo_trainer = PPOTrainer(initial_mario_lm, finetune_mario_lm, preference_model, mario_model_checkpoint_dir, 0.01,
                             0.2)
    ppo_trainer.train(training_prompts, num_of_epochs)


if __name__ == "__main__":
    LOG_FILE_PATH = '../logs/ppo_directscore_3_1_training_log.log'
    # Configure logging
    logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    LOGGER = logging.getLogger('PPOTrainer')

    PREFERENCE_MODEL_PATH = '../checkpoints/pm_v2/best_pmv2_checkpoint_epoch_51.pth'
    CHECKPOINT_DIR = '../checkpoints/ppo_directscore_3'
    DATASET_PATH = '../sampling/input_prompts.txt'

    prompts = []

    with open(DATASET_PATH, 'r') as file:
        for line in file:
            prompt = line.strip()
            prompts.append(prompt)

    run_ppo_trainer(PREFERENCE_MODEL_PATH, CHECKPOINT_DIR, prompts, 102)
