from mario_gpt import SampleOutput
from mario_gpt.sampler import convert_level_to_png
from PIL import Image, ImageTk
import tkinter as tk
import csv


# Function to load and filter levels
def load_and_filter_levels(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        levels = list(reader)[1:]  # Skip the first row (header)
        return [
            row for row in levels
            if float(row[6]) >= 0.6 and float(row[8]) == 1.0 and float(row[11]) == 1.0
        ]


# Function to get unique levels based on column 1 and highest column 4 value
def get_unique_levels(levels, common_column_1):
    unique_levels = {}
    for row in levels:
        column_1_value = row[0]
        column_7_value = float(row[6])
        if (column_1_value in common_column_1) and (
                column_1_value not in unique_levels or column_7_value > float(unique_levels[column_1_value][3])):
            unique_levels[column_1_value] = row
    return unique_levels


def show_level_images(img1, img2, border_size=5, border_color=(0, 0, 0)):
    # Add border to img1
    img1_with_border = Image.new('RGB',
                                 (img1.width + 2 * border_size, img1.height + 2 * border_size),
                                 border_color)
    img1_with_border.paste(img1, (border_size, border_size))

    # Add border to img2
    img2_with_border = Image.new('RGB',
                                 (img2.width + 2 * border_size, img2.height + 2 * border_size),
                                 border_color)
    img2_with_border.paste(img2, (border_size, border_size))

    # Calculate the max width and total height for the combined image
    max_width = max(img1_with_border.width, img2_with_border.width)
    total_height = img1_with_border.height + img2_with_border.height

    # Create a new image that combines the two vertically (up and down)
    combined_image = Image.new('RGB', (max_width, total_height))

    # Paste both images with borders onto the combined image (one above the other)
    combined_image.paste(img1_with_border, (0, 0))
    combined_image.paste(img2_with_border, (0, img1_with_border.height))

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Level Images")

    # Convert the combined image to ImageTk format for Tkinter
    combined_image_tk = ImageTk.PhotoImage(combined_image)

    # Create a label to display the combined image
    label = tk.Label(root, image=combined_image_tk)
    label.pack()

    # Run the Tkinter main loop to display the window
    root.mainloop()


# Main function
if __name__ == "__main__":
    # File paths
    mariogpt_sampling_file = '../sampling/sampling_original_model_7.csv'
    finetuned_mariogpt_sampling_file = '../sampling/sampling_finetuned_16.csv'

    # Load and filter levels from both files
    mariogpt_levels = load_and_filter_levels(mariogpt_sampling_file)
    finetuned_mariogpt_levels = load_and_filter_levels(finetuned_mariogpt_sampling_file)

    # Find the intersection of column 1 (zero-indexed as row[0])
    common_column_1 = {row[0] for row in mariogpt_levels}.intersection(
        {row[0] for row in finetuned_mariogpt_levels}
    )

    # Filter and get unique levels based on the highest column 4 value
    unique_mariogpt_levels = get_unique_levels(mariogpt_levels, common_column_1)
    unique_finetuned_mariogpt_levels = get_unique_levels(finetuned_mariogpt_levels, common_column_1)

    while 1:
        # List the keys of unique_mariogpt_levels
        print("Level prompts:")
        mariogpt_keys = list(unique_mariogpt_levels.keys())
        for idx, key in enumerate(mariogpt_keys):
            print(f"{idx + 1}. {key}")

        # User selection
        user_choice = int(input(f"\nPlease select a prompt by entering its number (1-{len(mariogpt_keys)}): "))

        # Retrieve and display the selected row
        if 1 <= user_choice <= len(mariogpt_keys):
            selected_key = mariogpt_keys[user_choice - 1]
            selected_mariogpt_level = unique_mariogpt_levels[selected_key]
            selected_finetuned_mariogpt_level = unique_finetuned_mariogpt_levels[selected_key]

            print(f"\nYou selected the level prompt: {selected_key}")

            # Load and play the selected levels
            generated_level1 = SampleOutput.load(selected_mariogpt_level[4])
            generated_level1.play()
            generated_level1.img = convert_level_to_png(generated_level1.level)[0]

            generated_level2 = SampleOutput.load(selected_finetuned_mariogpt_level[4])
            generated_level2.play()
            generated_level2.img = convert_level_to_png(generated_level2.level)[0]

            show_level_images(generated_level1.img, generated_level2.img)
        else:
            print("Invalid selection. Please try again.")
