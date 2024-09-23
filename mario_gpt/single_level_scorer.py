import csv
import math


def single_scorer(input_file, output_file):
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        next(reader, None)
        rows = list(reader)

    if rows:
        for i, row in enumerate(rows):
            score = 0
            novelty = float(row[6])
            playability = int(float(row[8]))
            aesthetic = int(float(row[11]))
            # If Novelty is greater than 0.6 and the level playable and no fault tile, give score 1 else 0
            # if novelty >= 0.6 and playability == 1 and  aesthetic == 1:
            score = (0.2 * novelty) + (0.6 * playability) + (0.2 * aesthetic)
            score = round(score, 2)
            row.append(score)

    # Write the updated content back to a new CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == "__main__":
    single_scorer("../sampling/sampling_score.csv", "../sampling/sampling_score_one.csv")
