import os, json, requests, runpod

import random
import torch
import numpy as np
from PIL import Image
from comfy.sd import load_checkpoint_guess_config
import nodes

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

with torch.inference_mode():
    model_patcher, clip, vae, clipvision = load_checkpoint_guess_config("/content/ComfyUI/models/checkpoints/model.safetensors", output_vae=True, output_clip=True, embedding_directory=None)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    positive_prompt = values['positive_prompt']
    negative_prompt = values['negative_prompt']
    width = values['width']
    height = values['height']
    seed = values['seed']
    steps = values['steps']
    cfg = values['cfg']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']

    latent = {"samples":torch.zeros([1, 4, height // 8, width // 8])}
    cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    n_cond, n_pooled = clip.encode_from_tokens(clip.tokenize(negative_prompt), return_pooled=True)
    n_cond = [[n_cond, {"pooled_output": n_pooled}]]
    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(seed)
    sample = nodes.common_ksampler(model=model_patcher, 
                            seed=seed, 
                            steps=steps, 
                            cfg=cfg, 
                            sampler_name=sampler_name, 
                            scheduler=scheduler, 
                            positive=cond, 
                            negative=n_cond,
                            latent=latent, 
                            denoise=1)
    sample = sample[0]["samples"].to(torch.float16)
    vae.first_stage_model.cuda()
    decoded = vae.decode_tiled(sample).detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/output_image.png")

    result = "/content/output_image.png"
    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
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
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})