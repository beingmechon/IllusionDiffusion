from inference import inference
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,  # <-- Added import
    EulerDiscreteScheduler  # <-- Added import
)
import time

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
controlnet = ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float32)#, torch_dtype=torch.float32)
main_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float32,
).to("cuda")

image_pipe = StableDiffusionControlNetImg2ImgPipeline(**main_pipe.components, requires_safety_checker=False)

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
