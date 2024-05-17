from diffusers import AutoPipelineForText2Image
import torch
import json, os, requests
import runpod

discord_token = os.getenv('com_camenduru_discord_token')

pipe = AutoPipelineForText2Image.from_pretrained(
    "misri/cyberrealisticXL_v11VAE",
    torch_dtype=torch.float16,
    variant="fp16",
    requires_safety_checker=False).to("cuda:0")

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2

def generate(input):
    command = input["input"]
    command = json.dumps(command)
    values = json.loads(command)
    width = closestNumber(values['width'], 8)
    height = closestNumber(values['height'], 8)
    images = pipe(values['prompt'], negative_prompt=values['negative_prompt'], num_inference_steps=25, guidance_scale=7.5, width=width, height=height)
    result = f"/content/{input['id']}.png"
    images.images[0].save(result)
    
    response = None
    try:
        source_id = values['source_id']
        source_channel = values['source_channel']
        files = {f"image.png": open(result, "rb").read()}
        payload = {"content": f"{command} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})