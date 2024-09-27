"""
Author: Balasubramani Murugan

This script evaluates the playability and difficulty of Mario game levels based on text representation
and calculates scores, including normalization of those scores. The results are saved to a new CSV file.
"""
import csv
import json
from mario_gpt import SampleOutput
from concurrent.futures import ThreadPoolExecutor


def run_astar_agent_simulator(game_level):
    """
    Runs the A* agent simulator on the given game level and returns the simulator response.

    @param game_level: text representation of the level
    @return: the response of the simulator
    """
    output = game_level.run_astar_evaluate()
    if output.stderr.strip() != '':
        if 'class file version' in output.stderr:
            raise Exception("Jar and Java Version Mismatch Error")
        else:
            json_resp = '{"gameEvents":[],"agentEvents":[],"remainingTime":0,"killsByStomp":0,"killsTotal":0,' \
                        '"killsByFire":0,"killsByFall":0,"marioMode":0,"gameStatus":"LOSE","killsByShell":0,' \
                        '"numCollectedMushrooms":0,"completionPercentage":1.0,"numCollectedFireflower":0,' \
                        '"numDestroyedBricks":0,"marioNumHurts":0,"currentCoins":0,"maxJumpAirTime":0,' \
                        '"numBumpBrick":0,"numJumps":0,"maxXJump":0,"currentLives":0,"numBumpQuestionBlock":0,' \
                        '"numCollectedTileCoins":0}'
    else:
        json_resp = output.stdout

    return json.loads(json_resp)


def measure_playability(simulator_response):
    """
    Score the playability of the game level based on the simulator's response if the status is WIN then playable
    otherwise not playable.

    @param simulator_response: Response from the simulator
    @return: the playability score
    """
    game_status = simulator_response.get('gameStatus')
    playability_reward = -1

    if game_status == "WIN":
        playability_reward = 1
    elif game_status == "LOSE" or game_status == "TIME_OUT":
        playability_reward = -100

    return playability_reward


def measure_difficulty(simulator_response):
    """
    Measures the difficulty of the game level using features like jump distances, jump counts and air time.

    @param simulator_response: Response from the simulator
    @return: the difficulty score
    """
    max_x_jump = simulator_response.get("maxXJump")
    max_jump_air_time = simulator_response.get("maxJumpAirTime")
    num_jumps = simulator_response.get("numJumps")

    weight_max_x_jump = 0.5
    weight_max_jump_air_time = 0.3
    weight_num_jumps = 0.2

    difficulty_score = (
            weight_max_x_jump * max_x_jump +
            weight_max_jump_air_time * max_jump_air_time +
            weight_num_jumps * num_jumps
    )

    return difficulty_score


def normalize_playability_score(score):
    """
    Normalizes the playability score. Returns 1.0 for positive scores, 0.0 for non-positive.

    @param score: Playability score
    @return: normalized playability score
    """
    if score > 0:
        return 1.0
    else:
        return 0.0


def process_level(row):
    """
    Processes a single game level, running the simulator and calculating playability and difficulty.

    @param row: A row from the input CSV file, representing a game level.
    @return: The updated row with additional columns for playability, normalized playability, and difficulty scores.
    """
    try:
        g_level = SampleOutput.load(row[4])
        response = run_astar_agent_simulator(g_level)
        playability_score = measure_playability(response)
        difficulty_score = measure_difficulty(response)
        normalized_playability_score = normalize_playability_score(playability_score)

        # Append the calculated scores to the row
        row.append(str(playability_score))
        row.append(str(normalized_playability_score))
        row.append(str(difficulty_score))

        return row
    except Exception as e:
        print(f"Error processing level: {e}")
        return None


def compute_playability_and_difficulty(input_csv_path, output_csv_path):
    """
    Computes playability and difficulty for all levels in the input CSV file and writes the results to a new CSV file.

    @param input_csv_path: Path to the input CSV file containing game levels.
    @param output_csv_path: Path to the output CSV file where results will be saved.
    @return:
    """
    # Load the input CSV file
    with open(input_csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if rows:
        with ThreadPoolExecutor(max_workers=4) as executor:  # Use 4 threads; adjust based on your machine's capacity
            # Submit the processing of each level to the executor
            future_to_row = {executor.submit(process_level, row): row for row in rows}

            # Collect the processed results
            processed_rows = []
            for future in future_to_row:
                result = future.result()
                if result:
                    processed_rows.append(result)

        # Write the updated content back to a new CSV file
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(processed_rows)


if __name__ == "__main__":
    input_csv_path = '../sampling/sampling_1.csv'
    output_csv_path = '../sampling/sampling_1_score.csv'
    compute_playability_and_difficulty(input_csv_path, output_csv_path)
