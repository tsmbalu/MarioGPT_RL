from mario_gpt import SampleOutput


def evaluate_aesthetic(level_text):
    """
    Evaluate the aesthetic of a Mario game level text corpus.

    Args:
        level_text (str): The text representation of the Mario game level.

    Returns:
        int: A reward value indicating the aesthetic quality of the level.
             Positive values indicate a perfect aesthetic, while negative values
             indicate a imperfect aesthetic.
    """
    reward = 0
    stack = []

    for char in level_text:
        if char == '[':
            stack.append(char)
        elif char == ']':
            if not stack:
                reward -= 10  # Negative reward for missing opening bracket
            else:
                stack.pop()

    # Negative reward for missing closing brackets
    reward -= len(stack) * 10

    # If no missing brackets, give positive reward
    if reward == 0:
        reward = 10

    return reward


def normalize_score(score):
    """
    Normalizes the score based on the given conditions.

    Parameters:
    score (float): The score to be normalized.

    Returns:
    float: The normalized score.
    """
    if score > 0:
        return 1.0
    elif score <= 0:
        return 0.0


import csv

if __name__ == "__main__":
    input_csv_path = '../sampling/sampling_1.csv'
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        for i, row in enumerate(rows):
            generated_level = SampleOutput.load(row[4])
            level_txt = "\n".join(generated_level.level)
            aesthetic_score = evaluate_aesthetic(level_txt)
            normalize_aesthetic_score = normalize_score(aesthetic_score)
            row.append(str(aesthetic_score))
            row.append(str(normalize_aesthetic_score))

    output_csv_path = '../sampling/sampling_1_score.csv'
    # Write the updated content back to a new CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
