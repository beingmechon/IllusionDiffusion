from inference import inference
from PIL import Image

control_image_path = ""
prompt = "Beautiful city"
negative_prompt = "monochrome, lowres, worst quality, low quality"

control_image = Image.open(control_image_path)
guidance_scale = 10.0
controlnet_conditioning_scale = 10
control_guidance_start = 0.5
control_guidance_end = 0.8
upscaler_strength = 1
seed = 123
sampler = "DPM++ Karras SDE"
sampler = "Euler"

result = inference(control_image,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
            upscaler_strength,
            seed,
            sampler)


result
