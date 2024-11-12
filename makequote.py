import torch
import streamlit as st
import requests
import io
from PIL import Image
from huggingface_hub import login
import numpy as np
import openai
import os
from dotenv import load_dotenv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# Hugging Face API Token and model details
hf_token = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": f"Bearer {hf_token}"}

# Log into Hugging Face (only needs to be done once)
login(hf_token)


def generate_quotes(model_path, num_quotes=5, max_length=2048):
    """Generate new quotes using the trained model"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    generated_quotes = []
    for _ in range(num_quotes):
        # Create input prompt
        input_text = f"<|startoftext|>"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        # Generate quote
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|endoftext|>")[0]
        )

        # Decode and clean up the generated quote
        quote = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_quotes.append(quote)

    return generated_quotes

# Define the function to query the Hugging Face model
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Streamlit UI
st.title("Image Generation with Diffusion Model")
st.write("Generate an image based on a custom prompt or use a default prompt.")

# Checkbox to choose between user prompt and default prompt
use_custom_prompt = st.checkbox("Use custom prompt", value=True)

# Default prompt
# default_prompts = [
#     "A serene, high-definition scene of a person surrounded by friends and family in a natural setting, each person showing gestures of kindness, warmth, and support. The central figure has a caring, thoughtful expression, and above their head, a gentle, glowing thought bubble appears with the text 'Take care of your people around you, and they will take care of you in return.' Soft golden sunlight filters through, casting a warm, peaceful glow over everyone, symbolizing trust and mutual care. The background includes lush greenery, subtle flowers, and a soft focus, giving the scene a calm, supportive atmosphere.",
#     "A peaceful meadow with a person sitting under a tree, surrounded by nature and animals. A thought bubble appears above their head with the words 'Harmony with nature brings harmony within.' The scene is illuminated by warm sunlight, giving a calming and introspective feel.",
#     "A group of friends gathered around a campfire under the night sky, sharing stories and laughter. The stars above are vibrant, and a gentle glow illuminates each face. A thought bubble above one person reads, 'Cherish these moments, they become our fondest memories.'",
#     "A lone figure walking through a dense forest path, sunlight filtering through the trees, casting gentle shadows. Above them is a soft thought bubble with the words 'The journey is as meaningful as the destination.' The atmosphere is quiet, reflective, and serene."
# ]

if use_custom_prompt:
    suggestions = st.text_area(
        "Enter a prompt describing the image you want to generate:",
    )
else:
    suggestions = ""


thoughts = generate_quotes("quotegeneratormodel")
thought = np.random.choice(thoughts)
print(thought)

custom_prompt = f"""
Create a detailed Stable Diffusion prompt that will generate an image featuring this quote: "{thought}"

User has provided these specific suggestions for the image: '{suggestions}'

Requirements:
1. The image must incorporate the user's suggestions while maintaining visual harmony
2. The text of the quote must be clearly visible and readable. Now ords within the text should change.
3. The composition should put emphasis on the quote while integrating suggested elements
4. Include specific style and quality-boosting terms for Stable Diffusion
5. All user suggestions must be naturally incorporated into the final image

Think about:
- How to best integrate user suggestions with the quote's meaning
- What artistic style would unite the suggestions and quote
- What text style and placement would ensure readability among the suggested elements
- How to maintain balance between quote visibility and suggested elements

Format your response as a single, detailed prompt suitable for Stable Diffusion.
Include technical parameters like aspect ratio (16:9 for wallpapers), quality boosters, and negative prompts.
DO NOT include explanations or multiple options - just the final prompt."""   

custom_response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an expert in generating descriptive and emotionally resonant image descriptions."},
        {"role": "user", "content": custom_prompt}
    ],
   
    temperature=0.8
)

stable_diffusion_prompt = str(custom_response.choices[0].message.content)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Generate the image bytes from the API
        image_bytes = query({"inputs": stable_diffusion_prompt})

        # Display the generated image
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Generated Image", use_column_width=True)

        # Option to download the image
        image_path = "generated_image.png"
        image.save(image_path)
        with open(image_path, "rb") as file:
            btn = st.download_button(
                label="Download Image",
                data=file,
                file_name="generated_image.png",
                mime="image/png"
            )

st.write("Note: Each generation may take a few moments. Try different prompts to explore!")