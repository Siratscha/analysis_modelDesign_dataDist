from PIL import Image
import os

# Define the dimensions of the grid (6 rows and 4 columns)
rows, cols = 6, 4
image_width, image_height = 400, 400  # Adjust these dimensions as needed

# Folder containing your images
folder_path = r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\images\SD_examples\SD_examples\female"  # Replace with the actual folder path

# Create a blank canvas for the composite image
composite = Image.new('RGB', (cols * image_width, rows * image_height))

# List all image files in the folder
image_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg', '.gif'))]

# Iterate over the image files and paste them onto the canvas
for i, image_file in enumerate(image_files):
    image = Image.open(image_file)
    x = (i % cols) * image_width
    y = (i // cols) * image_height
    composite.paste(image, (x, y))

# Save the composite image
composite.save(r"C:\Users\rankl\Documents\uni\Thesis\Development\modelDesign_bias_CXR\images\SD_examples\SD_examples\composite_imageFemale.jpg")

# Display the composite image (optional)
composite.show()