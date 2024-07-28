from mario_gpt import SampleOutput


def count_elements(grid, element):
    return sum(row.count(element) for row in grid)


def calculate_density(grid, element):
    total_tiles = len(grid) * len(grid[0])
    element_count = count_elements(grid, element)
    return element_count / total_tiles


def analyze_enemy_density(grid):
    return calculate_density(grid, 'E')


def analyze_jump_complexity(grid):
    max_jump_height = 4
    max_gap_width = 3
    jump_difficulty = 0

    # Analyze vertical jumps (height)
    for r in range(len(grid) - max_jump_height):
        for c in range(len(grid[0])):
            if grid[r][c] == '-' and all(grid[r + i][c] == ' ' for i in range(1, max_jump_height)):
                jump_difficulty += 1

    # Analyze horizontal jumps (width)
    for r in range(len(grid)):
        for c in range(len(grid[0]) - max_gap_width):
            if grid[r][c] == '-' and all(grid[r][c + i] == ' ' for i in range(1, max_gap_width + 1)):
                jump_difficulty += 1

    return jump_difficulty


def neighbors(grid, row, col):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    result = []
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            result.append((r, c))
    return result


def analyze_path_complexity(grid):
    path_length = 0
    current_pos = (len(grid) - 1, 0)
    goal_pos = (0, len(grid[0]) - 1)

    open_set = [current_pos]
    came_from = {}

    while open_set:
        current = open_set.pop(0)
        if current == goal_pos:
            break
        for neighbor in neighbors(grid, *current):
            if neighbor not in came_from:
                came_from[neighbor] = current
                open_set.append(neighbor)
                path_length += 1

    return path_length


def measure_difficulty(vglc_text):
    enemy_density = analyze_enemy_density(vglc_text)
    jump_complexity = analyze_jump_complexity(vglc_text)
    path_complexity = analyze_path_complexity(vglc_text)

    difficulty_score = (enemy_density * 2 +
                        jump_complexity * 3 +
                        path_complexity * 1)

    return difficulty_score


if __name__ == "__main__":
    generated_level = SampleOutput.load("../lvl-2.txt")
    print(measure_difficulty(generated_level.level))