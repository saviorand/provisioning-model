import erdantic as erd
from PIL import Image


def save_erdantic_diagram(diagram: erd.EntityRelationshipDiagram, img_title: str):
    output_path = f'../../output/{img_title}.png'
    diagram.draw(out=output_path)

    return output_path


#
# def concatenate_images_horizontally(img_paths, output_path):
#     images = [Image.open(img) for img in img_paths]
#
#     # Get the max height among all images
#     max_height = max(img.height for img in images)
#
#     # Concatenate images
#     total_width = sum(img.width for img in images)
#     concatenated_img = Image.new('RGB', (total_width, max_height))
#
#     x_offset = 0
#     for img in images:
#         # Calculate the vertical position to center the image
#         y_offset = (max_height - img.height) // 2
#         concatenated_img.paste(img, (x_offset, y_offset))
#         x_offset += img.width
#
#     # Save the result
#     concatenated_img.save(output_path)

def concatenate_images_with_centered_below(img_paths, bottom_img_path, output_path):
    images = [Image.open(img) for img in img_paths]

    # Get the max height among all images
    max_height = max(img.height for img in images)

    # Concatenate images
    total_width = sum(img.width for img in images)
    concatenated_img = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for img in images:
        # Calculate the vertical position to center the image
        y_offset = (max_height - img.height) // 2
        concatenated_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    # Load the image to be placed at the bottom
    bottom_image = Image.open(bottom_img_path)

    # Create a new canvas that can fit the horizontally concatenated images and the bottom image
    new_total_height = max_height + bottom_image.height
    final_img = Image.new('RGBA', (total_width, new_total_height))
    final_img.paste(concatenated_img, (0, 0))

    # Center the bottom image horizontally and paste it below the concatenated images
    bottom_x_offset = (total_width - bottom_image.width) // 2
    final_img.paste(bottom_image, (bottom_x_offset, max_height))

    # Save the result
    final_img.save(output_path)
