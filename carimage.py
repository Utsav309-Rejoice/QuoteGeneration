import base64
import openai

import os
import json
import requests
from json_repair import repair_json
from huggingface_hub import login
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import gradio
from gradio_client import Client
import cv2
openai.api_key = st.secrets["OPENAI_API_KEY"]
# Hugging Face API Token and model details
hf_token = st.secrets["HF_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {hf_token}"}

login(hf_token)

def generating_pipeline():
    quote = generate_quotes()
    get_image_plain(quote)
    outdoor_image = indoor_outdoor("generated_image.png")
    font_file_path = "dejavu-sans-bold.ttf"
    print(outdoor_image)
    if outdoor_image == True:
        setting_type = "outdoor"
    else:
        setting_type = "indoor"
    result_path = create_image_overlay("generated_image.png", quote, setting_type, font_file_path)
    read_image = cv2.imread(result_path)
    st.image(read_image, caption="Generated Image", use_column_width=True)
    with open(result_path, "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="generated_image.png",
            mime="image/png"
        )


def generate_quotes(num_quotes=5, max_length=2048):
    """Generate new quotes using the trained model"""
    generated_quotes = []
    client = Client("UtsavSD/Quotemaker")
    for i in range(num_quotes):
        result = client.predict(api_name="/get_suggestion")
        print(result)
        generated_quotes.append(result)
    thought = np.random.choice(generated_quotes)
    return thought


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def get_image_plain(quote):
   image_generation_prompt = f"""Consider the given quote: '{quote}', generate a detailed Stable Diffusion prompt that will create a visually compelling wallpaper background featuring a realistic luxury/sports car/sedan/SUV, without including the quote text itself.
The image should naturally complement the meaning of the provided quote, but the quote text should not be present in the generated image.
The prompt should include:
Specific artistic style and visual elements featuring a luxury/sports car
High-quality technical parameters for Stable Diffusion
Mention that the quote will be overlaid later, so the image should have appropriate spacing/composition
A car prominently featured in the center of the frame, with atmospheric lighting and cinematic composition. High detail, 8K resolution, Hasselblad medium format quality, dramatic depth of field. Negative prompt: text, quote, lettering.
"""
   custom_response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert in generating descriptive and emotionally resonant image descriptions."},
        {"role": "user", "content": image_generation_prompt}
    ],
   
    temperature=0.8
    )
   stable_diffusion_prompt = str(custom_response.choices[0].message.content)
   image_bytes = query({"inputs": stable_diffusion_prompt})

        # Display the generated image
   image = Image.open(io.BytesIO(image_bytes))
        # Option to download the image
   image_path = "generated_image.png"
   image.save(image_path)
   

def indoor_outdoor(image_path):
    base64_image = encode_image(image_path)
    response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": """1. ENVIRONMENT DETECTION PROMPT:
Analyze the image and classify the environment based on these criteria:
Indoor indicators:
- Ceiling with fluorescent/artificial lighting
- Visible internal structure (beams, pipes, ducts)
- Showroom/garage flooring
- Controlled lighting conditions
- Multiple cars in display formation
Outdoor indicators:
- Natural sky/clouds
- Natural lighting
- Open roads/highways
- Natural landscapes (mountains, desert, trees)
- Street/outdoor parking
Parking Classification:
- Indoor: Covered garage, showroom, indoor facility
- Outdoor: Street parking, open parking lot, residential driveway
Return a JSON response in this format:
{
    "isOutdoor": false,
    "environment": {
        "type": "indoor",
        "specific": "showroom",
        "lighting": "artificial",
        "identifiers": [
            "fluorescent ceiling lights",
            "indoor flooring",
            "showroom display setup"
        ]
    }
}""",
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}"
          },
        },
      ],
    }
  ],
)
    
    print(response.choices[0].message.content)
    return json.loads(repair_json(response.choices[0].message.content)).get("isOutdoor")

def split_text(quote, max_words_per_line=5):
    words = quote.split()
    lines = []
    while words:
        lines.append(" ".join(words[:max_words_per_line]))
        words = words[max_words_per_line:]
    return lines

def create_image_overlay(image_path, quote, setting_type, font_path, output_path="output_image.jpg"):
    """
    Create an overlay on a car image with a motivational quote, adapting text placement based on the setting type.
    :param image_path: Path to the input image
    :param quote: Text to overlay on the image
    :param setting_type: Type of scene ('desert', 'road', 'house', 'parking garage')
    :param font_path: Path to the .ttf font file
    :param output_path: Path to save the output image
    """
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # Define font size dynamically based on image size
    font_size = int(width * 0.05)  # Adjust font size based on image dimensions
    font = ImageFont.truetype(font_path, font_size)
    # Split the quote into lines
    lines = split_text(quote, max_words_per_line=5)
    # Calculate the total height of the text block
    total_text_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines) + (len(lines) - 1) * 10
    # Determine text placement based on setting type
    if setting_type.lower() in ["outdoor","out"]:
        # For open settings, place text at the top
        y_start = int(height * 0.05)  # 5% from the top
        text_color = "white"  # Use white text for contrast with open skies
    elif setting_type.lower() in ["indoor","in"]:
        # For enclosed settings, center text on a wall or central area
        y_start = (height - total_text_height) // 2
        text_color = "black"  # Use black text for contrast with walls
    else:
        raise ValueError("Invalid setting type. Choose from 'desert', 'road', 'house', or 'parking garage'.")
    # Draw each line of text
    for line in lines:
        text_width, text_height = font.getbbox(line)[2] - font.getbbox(line)[0], font.getbbox(line)[3] - font.getbbox(line)[1]
        x = (width - text_width) // 2
        draw.text((x, y_start), line, fill=text_color, font=font)
        y_start += text_height + 10  # Add spacing between lines
    # Save the output image
    image.save(output_path)
    return output_path

st.title("Image Generation with Diffusion Model")
st.write("Generate an image based on a custom prompt or use a default prompt.")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        generating_pipeline()
