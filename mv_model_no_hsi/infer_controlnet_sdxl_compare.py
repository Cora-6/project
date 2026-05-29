#!/usr/bin/env python3
import argparse
import os
from contextlib import nullcontext

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from diffusers import StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler

from cnet_dataset_sdxl import HSIControlNetDataset
from warp_and_controlnet_model import WarpAndControlNetModel


def parse_args():
    parser = argparse.ArgumentParser("SDXL + WarpControlNet inference for first 10 samples")
    parser.add_argument("--root_dir", type=str, required=True, help="测试数据根目录")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="训练得到的 controlnet 权重目录")
    parser.add_argument("--base_model", type=str, default="hotshotco/SDXL-512", help="SDXL base model")
    parser.add_argument("--output_dir", type=str, default="./test_result", help="输出根目录")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_to_pil(tensor_chw):
    arr = tensor_chw.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def vis48(ch48):
    ch48 = ch48.astype(np.float32)
    ch48 = (ch48 - ch48.min()) / (ch48.max() - ch48.min() + 1e-6)
    rgb = ch48[:3].transpose(1, 2, 0)
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def add_title(img: Image.Image, title: str, title_h: int = 32):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + title_h), color=(0, 0, 0))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255))
    return canvas


def split_2x2(img: Image.Image):
    w, h = img.size
    assert w % 2 == 0 and h % 2 == 0, f"期望偶数尺寸，得到: {(w, h)}"
    hw, hh = w // 2, h // 2
    tl = img.crop((0, 0, hw, hh))
    tr = img.crop((hw, 0, w, hh))
    bl = img.crop((0, hh, hw, h))
    br = img.crop((hw, hh, w, h))
    return {"tl": tl, "tr": tr, "bl": bl, "br": br}


def main():
    args = parse_args()

    compare_dir = os.path.join(args.output_dir, "compare")
    result_dir = os.path.join(args.output_dir, "result")
    os.makedirs(compare_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = WarpAndControlNetModel.from_pretrained(
        args.ckpt_dir,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
        device_map=None,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        use_safetensors=True,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    dataset = HSIControlNetDataset(root_dir=args.root_dir, image_size=args.image_size)
    n = min(args.num_samples, len(dataset))

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for idx in tqdm(range(n), desc="Infer"):
        sample = dataset[idx]

        gt_img = tensor_to_pil(sample["pixel_values"])
        cond_tensor = sample["conditioning_pixel_values"]
        cond_img = vis48(cond_tensor.detach().cpu().numpy())

        cond_batch = cond_tensor.unsqueeze(0).to(device=device, dtype=weight_dtype)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
        )
        with torch.inference_mode(), autocast_ctx:
            pred = pipe(
                prompt=sample["text"],
                image=cond_batch,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

        # compare: 带标注，不裁切
        gt_t = add_title(gt_img, "GT")
        cond_t = add_title(cond_img, "COND")
        pred_t = add_title(pred, "INFERENCE")

        w, h = gt_t.size
        compare_canvas = Image.new("RGB", (w * 3, h), color=(255, 255, 255))
        compare_canvas.paste(gt_t, (0, 0))
        compare_canvas.paste(cond_t, (w, 0))
        compare_canvas.paste(pred_t, (w * 2, 0))
        compare_canvas.save(os.path.join(compare_dir, f"{idx:04d}_compare.png"))

        # result: 2x2 裁切后保存
        crops = split_2x2(pred)
        crops["tl"].save(os.path.join(result_dir, f"{idx:04d}_tl.png"))
        crops["tr"].save(os.path.join(result_dir, f"{idx:04d}_tr.png"))
        crops["bl"].save(os.path.join(result_dir, f"{idx:04d}_bl.png"))
        crops["br"].save(os.path.join(result_dir, f"{idx:04d}_br.png"))

    print(f"✅ compare 已保存到: {compare_dir}")
    print(f"✅ result(裁切后) 已保存到: {result_dir}")


if __name__ == "__main__":
    main()
