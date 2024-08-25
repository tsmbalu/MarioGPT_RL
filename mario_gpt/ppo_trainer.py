import tensorflow as tf
import keras
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelWithLMHead
from mario_gpt import MarioLM, SampleOutput


class PPOTrainer:
    def __init__(self, model, preference_model, tokenizer, checkpoint_dir, beta=0.01, clip_ratio=0.2):
        self.initial_model = model
        self.preference_model = preference_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.clip_ratio = clip_ratio
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-5)
        self.model = model
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=5)

    def compute_kl_divergence(self, current_probs, initial_probs):
        p = tf.clip_by_value(current_probs, 1e-10, 1.0)
        q = tf.clip_by_value(initial_probs, 1e-10, 1.0)
        return tf.reduce_sum(p * tf.math.log(p / q))
        # return tf.reduce_mean(tf.reduce_sum(current_probs * tf.math.log(current_probs / initial_probs), axis=-1))

    def compute_reward(self, x, y):
        input_text = x + " " + y
        input_ids = self.tokenizer(input_text, return_tensors="tf", padding=True, truncation=True,
                                   max_length=512).input_ids

        preferability = self.preference_model(input_ids, training=False)

        initial_logits = self.initial_model(input_ids).logits
        current_logits = self.model(input_ids).logits

        current_probs = tf.nn.softmax(current_logits, axis=-1)
        initial_probs = tf.nn.softmax(initial_logits, axis=-1)
        kl_div = self.compute_kl_divergence(current_probs, initial_probs)

        reward = preferability - self.beta * kl_div
        return reward

    def ppo_loss(self, advantages, rewards, old_values):
        def loss_fn(y_true, y_pred):
            prob_ratio = tf.exp(tf.math.log(y_pred + 1e-10) - tf.math.log(y_true + 1e-10))
            clipped_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = prob_ratio * advantages
            surrogate2 = clipped_ratio * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            critic_loss = tf.reduce_mean(tf.square(rewards - old_values))
            return actor_loss + critic_loss

        return loss_fn

    def train(self, dataset, num_epochs=10, save_freq=5):
        for epoch in range(num_epochs):
            total_loss = 0
            all_rewards, all_values, all_advantages = [], [], []

            for input_ids, _ in dataset:
                # Generate level with the current policy
                responses = self.model.generate(input_ids, max_length=50)
                x = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                y = self.tokenizer.batch_decode(responses, skip_special_tokens=True)

                for prompt, response in zip(x, y):
                    reward = self.compute_reward(prompt, response)
                    value = self.model(input_ids)[0]
                    all_rewards.append(reward)
                    all_values.append(value)

            advantages = self.get_advantages(all_rewards, all_values)
            all_advantages.extend(advantages)

            for input_ids, (reward, advantage, value) in zip(dataset, zip(all_rewards, all_advantages, all_values)):
                with tf.GradientTape() as tape:
                    logits = self.model(input_ids).logits
                    loss = self.ppo_loss(advantage, reward, value)
                    loss_value = loss(logits, logits)

                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                total_loss += loss_value.numpy()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")

            if (epoch + 1) % save_freq == 0:
                self.checkpoint_manager.save()
                print(f"Model saved at epoch {epoch + 1}")

    def get_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        # Placeholder method for calculating advantages
        # return [r - v for r, v in zip(rewards, values)]
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * (values[i + 1] if i + 1 < len(values) else 0) - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
        return advantages


if __name__ == "__main__":
    """
    # Initialize the PPO trainer
    ppo_trainer = PPOTrainer(model, preference_model, tokenizer)
    
    # Train the model using PPO
    ppo_trainer.next(dataset)
    """
