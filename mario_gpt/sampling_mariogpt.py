import torch
import time
import csv

from mario_gpt import MarioLM, SampleOutput


def generate_sampling_levels(prompts: [str], num_levels: int, mini_batch_size: int, output_dir: str):
    """
    Generate multiple sampling levels for a batch of prompts.
    @param prompts: A list of prompts describing the levels' features.
    @param num_levels: Number of levels to generate for each prompt.
    @param mini_batch_size: Mini batch size to generate multiple  prompt at a time.
    """
    num_steps = 1400
    temperature = 2.5

    # Instantiate the MarioGPT model
    mario_lm = MarioLM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mario_lm = mario_lm.to(device)

    for _ in range(num_levels):
        # Generate levels for the entire batch of prompts
        batch_outputs = mario_lm.sample(
            prompts=prompts,
            num_steps=num_steps,
            temperature=temperature,
            use_tqdm=True
        )

        for _ in range(num_levels):
            for start_idx in range(0, len(prompts), mini_batch_size):
                # Create a mini-batch of prompts
                mini_batch_prompts = prompts[start_idx:start_idx + mini_batch_size]

                # Generate levels for the mini-batch of prompts
                batch_outputs = mario_lm.sample(
                    prompts=mini_batch_prompts,
                    num_steps=num_steps,
                    temperature=temperature,
                    use_tqdm=True
                )

                for i, sample_output in enumerate(batch_outputs):
                    current_time_millis = int(time.time() * 1000)
                    prompt = mini_batch_prompts[i]
                    output_file = f"{output_dir}/generated_level/generated_level_{current_time_millis}_{start_idx + i}.txt "
                    sample_output.save(output_file)

                    sampling_data = f'{output_dir}/sampling_1.csv'
                    # Open the file in append mode and create a writer object
                    with open(sampling_data, mode='a', newline='') as level_file:
                        writer = csv.writer(level_file)
                        # Write the row of data
                        writer.writerow([prompt, num_steps, temperature, current_time_millis, output_file])


if __name__ == "__main__":
    file_path = '../sampling/input_prompts.txt'
    prompts = []

    with open(file_path, 'r') as file:
        for line in file:
            prompt = line.strip()
            prompts.append(prompt)

    # Set the number of levels to generate for each prompt
    num_levels_to_generate = 5
    mini_batch = 32
    output_dir = '../sampling/'

    # Generate levels for all prompts in a batch
    generate_sampling_levels(prompts, num_levels_to_generate, mini_batch, output_dir)
