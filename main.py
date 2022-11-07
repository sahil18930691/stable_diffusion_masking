# !pip install diffusers==0.3.0

from diffusers import StableDiffusionImg2ImgPipeline
import requests
from PIL import Image
from io import BytesIO

url = "https://scitechdaily.com/images/Dog-Park.jpg"

response = requests.get(url)

init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((768, 512))

# using the 115,000 steps checkpoint
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("naclbit/trinart_stable_diffusion_v2", revision="diffusers-115k")
#pipe.to("cuda")

#images = pipe(prompt="Manga drawing of Brad Pitt", init_image=init_image, strength=0.75, guidance_scale=7.5).images
