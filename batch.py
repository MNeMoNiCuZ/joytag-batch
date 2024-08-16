from Models import VisionModel
from PIL import Image
import torch.amp.autocast_mode
from pathlib import Path
import torch
import torchvision.transforms.functional as TVF
import os
import tiktoken  # For token counting
from tqdm import tqdm  # For progress tracking

# Options
input_dir = 'input'  # Directory containing the images to process
path = 'models/'  # Change this to where you downloaded the model
save_in_same_directory = True  # Set to False to save in a separate 'output' directory
extension = 'txt'  # Extension for the saved tag files
overwrite_existing = False  # Set to False to skip processing images with existing 

certainty_threshold = 0.4  # Threshold for considering tags during prediction
save_threshold = 0.2  # Saves all tags above this value

token_length = 100  # Maximum number of tokens, None to disable
string_length = None  # Maximum string length, None to disable
word_count = None  # Maximum word count, None to disableoutput files

# Load model and tags
model = VisionModel.load_model(path)
model.eval()
model = model.to('cuda')

with open(Path(path) / 'top_tags.txt', 'r') as f:
    top_tags = [line.strip() for line in f.readlines() if line.strip()]

def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
    # Pad image to square
    image_shape = image.size
    max_dim = max(image_shape)
    pad_left = (max_dim - image_shape[0]) // 2
    pad_top = (max_dim - image_shape[1]) // 2

    padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    # Resize image
    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
    
    # Convert to tensor
    image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

    # Normalize
    image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    return image_tensor

@torch.no_grad()
def predict(image: Image.Image):
    image_tensor = prepare_image(image, model.image_size)
    batch = {
        'image': image_tensor.unsqueeze(0).to('cuda'),
    }

    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        preds = model(batch)
        tag_preds = preds['tags'].sigmoid().cpu()
    
    scores = {top_tags[i]: tag_preds[0][i] for i in range(len(top_tags))}
    predicted_tags = [tag for tag, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > certainty_threshold]
    tag_string = ', '.join(predicted_tags)

    return tag_string, scores

def apply_constraints(tag_string):
    # Token length constraint
    if token_length is not None:
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(tag_string)
        if len(tokens) > token_length:
            truncated_tokens = tokens[:token_length]
            truncated_string = encoding.decode(truncated_tokens)
            tag_string = truncated_string.rsplit(' ', 1)[0]

    # String length constraint
    if string_length is not None:
        if len(tag_string) > string_length:
            tag_string = tag_string[:string_length].rsplit(' ', 1)[0]

    # Word count constraint
    if word_count is not None:
        words = tag_string.split(' ')
        if len(words) > word_count:
            tag_string = ' '.join(words[:word_count])

    return tag_string

def process_images(input_dir: str, save_in_same_directory=True, extension='txt', overwrite_existing=True):
    input_path = Path(input_dir)
    output_dir = Path('output')
    if not save_in_same_directory:
        output_dir.mkdir(exist_ok=True)

    supported_extensions = ('png', 'jpg', 'jpeg', 'webp')
    files_to_process = [os.path.join(root, file)
                        for root, _, files in os.walk(input_path)
                        for file in files if file.lower().endswith(supported_extensions)]

    for image_path in tqdm(files_to_process, desc="Processing images"):
        image_path = Path(image_path)
        if save_in_same_directory:
            output_file = image_path.with_suffix(f'.{extension}')
        else:
            output_file = output_dir / f"{image_path.stem}.{extension}"

        if not overwrite_existing and output_file.exists():
            #print(f"Skipping {image_path} as output file already exists.")
            continue

        image = Image.open(image_path)
        tag_string, scores = predict(image)

        save_tags = [tag for tag, score in scores.items() if score > save_threshold]
        save_tag_string = ', '.join(save_tags)

        # Apply constraints
        save_tag_string = apply_constraints(save_tag_string)

        with open(output_file, 'w') as f:
            f.write(save_tag_string)

        #print(f"Processed {image_path}: {save_tag_string}")

# Process images with the specified options
process_images(input_dir, save_in_same_directory, extension, overwrite_existing)
