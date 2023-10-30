import erdantic as erd
from PIL import Image


def save_erdantic_diagram(diagram: erd.EntityRelationshipDiagram, img_title: str):
    output_path = f'../../output/{img_title}.png'
    diagram.draw(out=output_path)

    return output_path


def concatenate_images_horizontally(img_paths, output_path):
    images = [Image.open(img) for img in img_paths]

    # Get the max height among all images
    max_height = max(img.height for img in images)

    # Concatenate images
    total_width = sum(img.width for img in images)
    concatenated_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        # Calculate the vertical position to center the image
        y_offset = (max_height - img.height) // 2
        concatenated_img.paste(img, (x_offset, y_offset))
        x_offset += img.width

    # Save the result
    concatenated_img.save(output_path)
