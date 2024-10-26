"""
Author: Balasubramani Murugan

This script evaluates the aesthetic quality of Mario game levels based on text representation
and calculates scores, including normalization of those scores. The results are saved to a new CSV file.
"""
from mario_gpt import SampleOutput
import csv


def evaluate_aesthetic(level_text):
    """
    Evaluates the aesthetic quality of a Mario game level based on its text representation.

    @param level_text: The text representation of the Mario game level.
    @return A reward score reflecting the aesthetic quality of the level.
             A positive score indicates good aesthetics (no mismatched brackets),
             while a negative score indicates structural problems (mismatched brackets).
    """
    reward = 0
    stack = []

    # Iterate through each character in the level text
    for char in level_text:
        if char == '[':
            # Opening bracket, push onto the stack
            stack.append(char)
        elif char == ']':
            # Closing bracket, check if there's a matching opening bracket
            if not stack:
                # If no opening bracket, penalize the reward
                reward -= 10  # Negative reward for missing opening bracket
            else:
                # Otherwise, pop the matched opening bracket
                stack.pop()

    # Penalize for any unmatched opening brackets left in the stack
    reward -= len(stack) * 10

    # If no mismatches (reward is 0), assign a positive reward for perfect aesthetics
    if reward == 0:
        reward = 10

    return reward


def normalize_score(score):
    """
    Normalizes the aesthetic score to a range of 0 or 1.

    If the score is positive, it returns 1 (indicating a good aesthetic), and if the score
    is non-positive, it returns 0.

    @param score: The raw aesthetic score.
    @return Normalized score. 1.0 for positive score, 0.0 for non-positive score.
    """
    if score > 0:
        return 1.0
    else:
        return 0.0


def compute_aesthetic(input_csv_path, output_csv_path):
    """
    Computes the aesthetic score for each level in the CSV file, normalizes the score,
    and writes the updated data to a new CSV file.

    @param: input_csv_path (str): The path to the input CSV file containing level data.
    @param: output_csv_path (str): The path to the output CSV file to store the updated data.
    """
    # Read the input CSV file
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # If the file is not empty, process each row
    if rows:
        for i, row in enumerate(rows):
            # Load the generated Mario level from the 5th column of the row (index 4)
            generated_level = SampleOutput.load(row[4])
            level_txt = "\n".join(generated_level.level)

            # Compute the aesthetic score for the level text
            aesthetic_score = evaluate_aesthetic(level_txt)

            # Normalize the aesthetic score
            normalize_aesthetic_score = normalize_score(aesthetic_score)

            # Append both the raw aesthetic score and normalized score to the row
            row.append(str(aesthetic_score))
            row.append(str(normalize_aesthetic_score))

    # Write the updated rows back to a new CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    input_csv_path = '../sampling/sampling_1.csv'
    output_csv_path = '../sampling/sampling_1_score.csv'
    # Run the aesthetic evaluation and scoring process
    compute_aesthetic(input_csv_path, output_csv_path)
