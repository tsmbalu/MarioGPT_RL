import csv
import json

from mario_gpt import SampleOutput


def get_a_star_response(level_file_path):
    g_level = SampleOutput.load(level_file_path)
    output = g_level.run_astar_evaluate()
    json_resp = output.stdout
    return json.loads(json_resp)


def measure_playability(simulator_response):
    response = simulator_response
    game_status = response.get('gameStatus')
    print(game_status)
    playability_reward = -1

    if game_status == "WIN":
        playability_reward = 1
    elif game_status == "LOSE" or game_status == "TIME_OUT":
        playability_reward = -100

    return playability_reward


def measure_difficulty(simulator_response):
    response = simulator_response
    max_x_jump = response.get("maxXJump")
    max_jump_air_time = response.get("maxJumpAirTime")
    num_jumps = response.get("numJumps")

    weight_max_x_jump = 0.5
    weight_max_jump_air_time = 0.3
    weight_num_jumps = 0.2

    # Calculate weighted difficulty score
    difficulty_score = (
            weight_max_x_jump * max_x_jump +
            weight_max_jump_air_time * max_jump_air_time +
            weight_num_jumps * num_jumps
    )

    return difficulty_score


def normalize_playability_score(score):
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


if __name__ == "__main__":
    input_csv_path = '../sampling/sampling_score.csv'
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        for i, row in enumerate(rows):
            response = get_a_star_response(row[4])
            playability_score = measure_playability(response)
            difficulty_score = measure_difficulty(response)
            normalized_playability_score = normalize_playability_score(playability_score)
            row.append(str(playability_score))
            row.append(str(normalized_playability_score))
            row.append(str(difficulty_score))

    output_csv_path = '../sampling/sampling_score_1.csv'
    # Write the updated content back to a new CSV file
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
