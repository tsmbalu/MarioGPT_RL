"""
Author: Balasubramani Murugan

This script evaluates the novelty of Mario game levels based on text representation
and calculates scores, including normalization of those scores. The results are saved to a new CSV file.
"""
from mario_gpt import SampleOutput

from collections import Counter
import numpy as np
import csv


def filter_tiles(level_text, tiles):
    return ''.join([char for char in level_text if char in tiles or char == '\n'])


def calculate_shannon_entropy(filtered_level_text):
    """
    Calculate the Shannon entropy of the entire level, giving variability of overall level

    @param level_text: Text representation of the level
    @return: entropy of the entire level
    """
    # Remove newlines and count tile occurrences
    tile_counts = Counter(filtered_level_text.replace('\n', ''))
    total_tiles = sum(tile_counts.values())

    # Calculate the probability of each tile type
    probabilities = [count / total_tiles for count in tile_counts.values()]

    # Calculate Shannon entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy


def segment_entropy(filtered_level_text, segment_size=10):
    """
    Calculate the entropy of the small segment

    @param filtered_level_text: Text representation of the level
    @param segment_size: Maximum of tiles in a segment
    @return: entropy of the segment
    """
    # Split the level into lines
    level_lines = filtered_level_text.split('\n')
    num_lines = len(level_lines)
    segment_entropies = []
    concentration_penalties = []

    # Iterate over the level in segments
    for start in range(0, num_lines, segment_size):
        segment = ''.join(level_lines[start:start + segment_size])
        if segment.strip():
            # Calculate entropy for the segment
            entropy = calculate_shannon_entropy(segment)
            segment_entropies.append(entropy)

            # Apply concentration penalty
            tile_counts = Counter(segment.replace('\n', ''))
            max_count = max(tile_counts.values())
            penalty = max_count / len(segment.replace('\n', ''))
            concentration_penalties.append(penalty)

    return segment_entropies, concentration_penalties


def rate_novelty_of_level(level_text, tiles_to_consider):
    """
    Score the novelty of the level design.

    @param level_text: Textual representation of the level
    @param tiles_to_consider: Tiles to be considered
    @return: novelty score of the level
    """
    # Filter the level text to include only the specified tiles
    filtered_level_text = filter_tiles(level_text, tiles_to_consider)

    # Calculate overall entropy
    overall_entropy = calculate_shannon_entropy(filtered_level_text)

    # Calculate segment entropies and penalties
    segment_entropies, concentration_penalties = segment_entropy(filtered_level_text)

    if len(segment_entropies) == 0:
        avg_segment_entropy = 0
        avg_concentration_penalty = 0
    else:
        # Average segment entropy
        avg_segment_entropy = sum(segment_entropies) / len(segment_entropies)
        # Average concentration penalty
        avg_concentration_penalty = sum(concentration_penalties) / len(concentration_penalties)

    print(overall_entropy)

    # Final score: overall entropy adjusted by segment variability and concentration penalties
    final_score = overall_entropy + avg_segment_entropy - avg_concentration_penalty
    return final_score


def normalize_score(score):
    """
    Normalizes the score based on the given conditions.

    @param score:  The score to be normalized.
    @return The normalized score.
    """
    if score > 3:
        return 1.0
    elif 0 <= score <= 3:
        return round(0.9 * (score / 3), 2)
    else:
        return 0


def compute_novelty_score(input_csv_path, output_csv_path):
    """
    Computes the novelty score for each level in the CSV file, normalizes the score,
    and writes the updated data to a new CSV file.

    @param: input_csv_path (str): The path to the input CSV file containing level data.
    @param: output_csv_path (str): The path to the output CSV file to store the updated data.
    """
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        for i, row in enumerate(rows):
            generated_level = SampleOutput.load(row[4])
            level_txt = "\n".join(generated_level.level)
            # Filter the most common block tiles i.e. X and S while calculating the novelty
            tiles_to_consider = '?SQ[]E'
            novelty_score = rate_novelty_of_level(level_txt, tiles_to_consider)
            normalize_novelty_score = normalize_score(novelty_score)
            row.append(str(novelty_score))
            row.append(normalize_novelty_score)

    # Write the updated content back to a new CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    input_csv_path = '../sampling/sampling_1.csv'
    output_csv_path = '../sampling/sampling_1_score.csv'
    compute_novelty_score(input_csv_path, output_csv_path)
