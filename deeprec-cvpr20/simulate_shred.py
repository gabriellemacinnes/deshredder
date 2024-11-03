from PIL import Image
import os
import argparse
from io import BytesIO
import matplotlib.pyplot as plt

def pad_image(img, target_height=3000, color=(255, 255, 255)):
    current_height = img.height
    # Calculate the padding required for top and bottom
    if current_height < target_height:
        total_padding = target_height - current_height
        top_padding = total_padding // 2
        bottom_padding = total_padding - top_padding  # ensure total padding is met
    else:
        # No padding needed if image height is already >= target height
        return img

    # Create a new padded image
    padded_img = Image.new('RGB', (img.width, target_height), color)
    # Paste the original image centered on the new canvas
    padded_img.paste(img, (0, top_padding))
    return padded_img

# Function to split an image into vertical strips of 3000x32 pixels
def shred_image_fixed_size(image_path, output_folder, strip_height=3000, strip_width=32):
    with Image.open(image_path) as img:
        # Convert to RGB if the image has different color modes (e.g., grayscale or CMYK)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = Image.open(image_path)


        # Pad the image to 3000x960 if necessary
        img = pad_image(img)
        plt.imshow(img)
        plt.show()

        width, height = img.size
        print(width, height)

        # Calculate the number of strips
        num_strips = width // strip_width + (1 if width % strip_width > 0 else 0)

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        dir = os.path.join(output_folder, 'strips')
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Iterate and save each strip
        for i in range(num_strips):
            left = i * strip_width
            right = min(left + strip_width, width)  # Ensure we don't exceed the image width

            # Crop each strip with the full height and specified width
            strip = img.crop((left, 0, right, height))
            output_path = os.path.join(dir, f'strip_{i + 1}.png')
            strip.save(output_path)
            print(f'Saved {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shred an image into 32-pixel wide strips")
    parser.add_argument('input_image_path', help="Path to the input image.")
    parser.add_argument('output_folder', help="Path to the output folder where strips will be saved.")

    args = parser.parse_args()

    shred_image_fixed_size(image_path=args.input_image_path, output_folder=args.output_folder)
