import torch
import time
import csv

from mario_gpt import MarioLM, SampleOutput


def generate_sampling_levels(prompt: str, num_levels: int) -> [SampleOutput]:
    """
    Generate multiple sampling levels for given prompt
    @param prompt: The prompt describing the level's features
    @param num_levels: Number of levels to generate
    @return: A list of generated levels
    """
    levels = []
    num_steps = 1400
    temperature = 2.5

    # Instantiate the MarioGPT model
    mario_lm = MarioLM()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mario_lm = mario_lm.to(device)

    for index in range(num_levels):
        sample_output = mario_lm.sample(
            prompts=[prompt],
            num_steps=num_steps,
            temperature=temperature,
            use_tqdm=True
        )
        current_time_millis = int(time.time() * 1000)
        output_file = f"../sampling/generated_level/generated_level_{current_time_millis}_{index}.txt"
        sample_output.save(output_file)

        sampling_data = '../sampling_1.csv'
        # Open the file in append mode and create a writer object
        with open(sampling_data, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the row of data
            writer.writerow([prompt, num_steps, temperature, current_time_millis, output_file])

        levels.append(sample_output)

    return levels


if __name__ == "__main__":
    file_path = '../sampling/input_prompts.txt'
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            prompt = (line.strip())
            # Generate levels for the given prompt
            num_levels_to_generate = 10
            levels = generate_sampling_levels(prompt, num_levels_to_generate)


