#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import time
from typing import List, Dict, Set, Union, Tuple, Any

import cv2
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from cnet_dataset import HSIControlNetDataset
from get_camera import get_camera
from warp_and_controlnet_model import WarpAndControlNetModel
from peft import LoraConfig, inject_adapter_in_model

class DebugTimer:
    def __init__(self):
        self.curr_time_ns = time.time_ns()
    def reset(self):
        self.curr_time_ns = time.time_ns()
    def update_and_log(self, msg:str=""):
        curr = time.time_ns()
        print(f"[Timer] {msg} | elapsed time: {(curr - self.curr_time_ns) * 1e-9}")
        self.curr_time_ns = time.time_ns()

class NullTimer():
    def reset():
        pass
    def update_and_log(self, msg:str=""):
        pass

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from diffusers import DDIMScheduler
from model_with_camera import NewUNet2DConditionModel

from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# def log_validation(
#     vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
# ):
#     logger.info("Running validation... ")

#     if not is_final_validation:
#         controlnet = accelerator.unwrap_model(controlnet)
#     else:
#         controlnet = ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

#     pipeline = StableDiffusionControlNetPipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         vae=vae,
#         text_encoder=text_encoder,
#         tokenizer=tokenizer,
#         unet=unet,
#         controlnet=controlnet,
#         safety_checker=None,
#         revision=args.revision,
#         variant=args.variant,
#         torch_dtype=weight_dtype,
#     )
#     pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
#     pipeline = pipeline.to(accelerator.device)
#     pipeline.set_progress_bar_config(disable=True)

#     if args.enable_xformers_memory_efficient_attention:
#         pipeline.enable_xformers_memory_efficient_attention()

#     if args.seed is None:
#         generator = None
#     else:
#         generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

#     if len(args.validation_image) == len(args.validation_prompt):
#         validation_images = args.validation_image
#         validation_prompts = args.validation_prompt
#     elif len(args.validation_image) == 1:
#         validation_images = args.validation_image * len(args.validation_prompt)
#         validation_prompts = args.validation_prompt
#     elif len(args.validation_prompt) == 1:
#         validation_images = args.validation_image
#         validation_prompts = args.validation_prompt * len(args.validation_image)
#     else:
#         raise ValueError(
#             "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
#         )

#     image_logs = []
#     inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

#     for validation_prompt, validation_image in zip(validation_prompts, validation_images):
#         validation_image = Image.open(validation_image).convert("RGB")

#         images = []

#         for _ in range(args.num_validation_images):
#             with inference_ctx:
#                 image = pipeline(
#                     validation_prompt, validation_image, num_inference_steps=20, generator=generator
#                 ).images[0]

#             images.append(image)

#         image_logs.append(
#             {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
#         )

#     tracker_key = "test" if is_final_validation else "validation"
#     for tracker in accelerator.trackers:
#         if tracker.name == "tensorboard":
#             for log in image_logs:
#                 images = log["images"]
#                 validation_prompt = log["validation_prompt"]
#                 validation_image = log["validation_image"]

#                 formatted_images = []

#                 formatted_images.append(np.asarray(validation_image))

#                 for image in images:
#                     formatted_images.append(np.asarray(image))

#                 formatted_images = np.stack(formatted_images)

#                 tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
#         elif tracker.name == "wandb":
#             formatted_images = []

#             for log in image_logs:
#                 images = log["images"]
#                 validation_prompt = log["validation_prompt"]
#                 validation_image = log["validation_image"]

#                 formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

#                 for image in images:
#                     image = wandb.Image(image, caption=validation_prompt)
#                     formatted_images.append(image)

#             tracker.log({tracker_key: formatted_images})
#         else:
#             logger.warning(f"image logging not implemented for {tracker.name}")

#         del pipeline
#         gc.collect()
#         torch.cuda.empty_cache()

#         return image_logs

def log_validation(
    text_encoder, 
    tokenizer, 
    unet, 
    controlnet, 
    args, 
    accelerator, 
    weight_dtype, 
    step, 
    is_final_validation=False
):
    logger.info("Running validation (Custom No-VAE Pipeline)... ")

    # 1. 准备模型
    if not is_final_validation:
        # 训练中 validation，只 unwrap controlnet
        controlnet = accelerator.unwrap_model(controlnet)
        # UNet 已经在主函数里 unwrap 或者直接可用，这里保持原样
    else:
        # 最终 validation，加载训练好的权重
        controlnet = WarpAndControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)
        controlnet.to(accelerator.device)

    # 确保所有模型处于 eval 模式
    unet.eval()
    controlnet.eval()
    text_encoder.eval()

    # 2. 准备 Scheduler (手动创建，不依赖 pipeline)
    # 使用 DDIM 进行确定性采样，步数少且快
    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    scheduler.set_timesteps(20) # 固定 20 步

    # 3. 准备随机种子
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # 4. 准备验证数据
    # 这里我们简化逻辑：不从 args.validation_image 读取文件，
    # 而是生成一组随机的 HSI 条件 (Condition) 和相机参数，
    # 或者你可以写死加载某个特定的 npy 文件来测试效果。
    
    validation_prompts = args.validation_prompt
    if validation_prompts is None:
        validation_prompts = ["a 3d city view"]
    
    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    
    # 定义尺寸参数 (必须与 Dataset 保持一致)
    NUM_FRAMES = 4
    LATENT_SIZE = 32      # 256 // 8
    COND_SIZE = 256       # HSI 输入尺寸
    HSI_CHANNELS = 48
    
    for prompt in validation_prompts:
        images = []

        # --- A. 构造输入数据 (模拟 Dataset 行为) ---
        # 1. Text Embeddings
        with torch.no_grad():
            text_input = tokenizer(
                prompt, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                truncation=True, 
                return_tensors="pt"
            )
            text_embeddings = text_encoder(text_input.input_ids.to(accelerator.device))[0]
            # Repeat: (1, 77, 768) -> (NUM_FRAMES, 77, 768)
            # 注意：Batch Size = 1
            text_embeddings = text_embeddings.repeat_interleave(NUM_FRAMES, dim=0)

        # 2. Conditioning Image (HSI)
        # 真实情况：这里应该加载 validation_image 路径对应的 .npy 并处理
        # 模拟情况：生成随机噪声模拟 HSI，或者全零
        cond_image_tensor = torch.randn(
            (NUM_FRAMES, HSI_CHANNELS, COND_SIZE, COND_SIZE), 
            device=accelerator.device, 
            dtype=weight_dtype
        )
        
        # 3. Camera Pose
        # 使用默认相机参数
        camera_matrix = get_camera(
            NUM_FRAMES, 
            elevation=15, 
            azimuth_start=0, 
            azimuth_span=360, 
            camera_distance=2.0
        ).to(accelerator.device, dtype=weight_dtype)

        # 4. Latents (初始化噪声)
        # Shape: (Frames, Channels=4, 32, 32)
        # 这里的 Channels=4 对应 Depth(1)+Normal(3)
        latents = torch.randn(
            (NUM_FRAMES, 4, LATENT_SIZE, LATENT_SIZE), 
            generator=generator, 
            device=accelerator.device, 
            dtype=weight_dtype
        )
        latents = latents * scheduler.init_noise_sigma

        # --- B. 推理循环 (Sampling Loop) ---
        for _ in range(args.num_validation_images): # 通常是 1
            with inference_ctx:
                for t in tqdm(scheduler.timesteps, disable=True): # 禁用进度条以免刷屏
                    # 1. ControlNet Forward
                    # 输入: Latents (32x32), Cond (256x256)
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=cond_image_tensor,
                        return_dict=False,
                    )

                    # 2. UNet Forward
                    noise_pred = unet(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=text_embeddings,
                        camera=camera_matrix, # 传入 Camera
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    # 3. Scheduler Step
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

            # --- C. 后处理 (Upscale & Decode) ---
            # 直接插值放大回 256
            decoded_images = torch.nn.functional.interpolate(
                latents, size=(256, 256), mode='bilinear'
            )
            
            # [-1, 1] -> [0, 1]
            decoded_images = (decoded_images / 2 + 0.5).clamp(0, 1)
            # 转 Numpy: (4, 4, 256, 256) -> (4, 256, 256, 4)
            decoded_images = decoded_images.cpu().permute(0, 2, 3, 1).float().numpy()

            # 可视化处理：只取 Normal 部分 (后3通道) 并拼接
            # 如果想看 Depth，可以取 decoded_images[..., 0]
            normal_vis = decoded_images[..., 1:] # 取出 Normal 通道 (3通道)
            
            # 拼接 4 个视角: [Frame1, Frame2, Frame3, Frame4] 横向拼接
            combined_view = np.concatenate([normal_vis[i] for i in range(NUM_FRAMES)], axis=1)
            
            # 转 uint8 图片
            final_img = Image.fromarray((combined_view * 255).astype(np.uint8))
            images.append(final_img)

        # 记录 log
        image_logs.append(
            {"validation_image": images[0], "images": images, "validation_prompt": prompt}
        )

    # --- 5. 发送到 Tracker (Tensorboard/WandB) ---
    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                
                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))
                
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)
            tracker.log({tracker_key: formatted_images})

    # 清理显存
    del latents, cond_image_tensor, text_embeddings, controlnet
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=True,
        help="Allow execution of custom code from the dataset repository.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def collate_fn(examples: List[Dict[str, Any]]):
    """
    input dict (from HSIControlNetDataset):
        image: np.ndarray (Frames, 4, 32, 32)
        conditioning_image: np.ndarray (Frames, 48, 256, 256)
        camera_matrix: np.ndarray (Frames, 16)
        input_ids: torch.Tensor
    """
    pixel_values = torch.tensor(np.stack([ex["image"] for ex in examples], 0))  # (B, F, 4, H, W)
    conditioning_pixel_values = torch.tensor(np.stack([ex["conditioning_image"] for ex in examples], 0))  # (B, F, 48, 256, 256)
    camera_matrices = torch.tensor(np.stack([ex["camera_matrix"] for ex in examples], 0))  # (B, F, 16)
    input_ids = torch.stack([ex["input_ids"] for ex in examples])  # (B, seq_len)

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "camera_matrix": camera_matrices,
        "input_ids": input_ids,
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    # vae = AutoencoderKL.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    # )

    temp_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    unet = NewUNet2DConditionModel(**temp_unet.config, camera_dim=16)
    state_dict = torch.load("/home/viplab/liukejia/normal-depth-diffusion/diffusers_nd/py/nd_mv_unet.pth", map_location="cpu")
    unet.load_state_dict(state_dict, strict=True)

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = WarpAndControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = WarpAndControlNetModel.from_unet(unet,conditioning_channels=48)

    unet_lora_config = LoraConfig(r=8, lora_alpha=8, init_lora_weights="gaussian", target_modules=["to_k_proj", "to_q_proj", "to_v_proj", "to_out.0"])
    unet = inject_adapter_in_model(unet_lora_config, unet, adapter_name="unet_lora")
    print("Injected UNet LoRA adapters")


    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = WarpAndControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(controlnet.parameters())
    for name, param in unet.named_parameters():
        if "lora" in name or "conv_out" in name:
            params_to_optimize.append(param)
        
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # train_dataset = make_train_dataset(args, tokenizer, accelerator)
    train_dataset = HSIControlNetDataset(
        root_dir=os.path.join(args.dataset_name),
        image_size=args.resolution,
        tokenizer_name=args.pretrained_model_name_or_path,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    assert isinstance(controlnet, WarpAndControlNetModel), type(controlnet)

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    tim = NullTimer()

    NUM_FRAMES = 4 
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            tim.update_and_log('enter loop body')
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                # latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                # latents = latents * vae.config.scaling_factor

                # tim.update_and_log('vae encoding')

                # 1. 准备 Latents 和 Conditioning Image (需要展平)
                # 假设 Dataset 返回形状: [Batch, Frames, Channels, H, W]
                # 目标形状: [Batch * Frames, Channels, H, W]
                bsz = batch["pixel_values"].shape[0] # 获取 Batch Size (例如 4)
                latents = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                H, W = latents.shape[3], latents.shape[4]
                assert H == args.resolution and W == args.resolution, f"Expected image size to be {args.resolution}x{args.resolution}, but got {H}x{W}"
                latents = latents.view(bsz * NUM_FRAMES, 4, H, W) # Flatten

                tim.update_and_log('dont do vae encoding, just use images directly as latents')

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                # bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                timesteps = timesteps.repeat_interleave(NUM_FRAMES, dim=0)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                    dtype=weight_dtype
                )

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]# 输出形状: [Batch, 77, 768]

                 # 我们需要把每个 Prompt 复制 4 次，变成 [Prompt1, Prompt1, Prompt1, Prompt1, Prompt2...]
                # 输出形状: [Batch * 4, 77, 768]
                encoder_hidden_states = encoder_hidden_states.repeat_interleave(NUM_FRAMES, dim=0)

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)# torch.Size([8, 4, 48, 512, 512])
                cond_H, cond_W = controlnet_image.shape[3], controlnet_image.shape[4]
                controlnet_image = controlnet_image.view(bsz * NUM_FRAMES, 48, cond_H, cond_W)# torch.Size([32, 48, 512, 512])
                assert controlnet_image.shape[2:] == (H*8, W*8), f"Expected controlnet image to be at 8x scale of latents, but got {controlnet_image.shape[2:]} vs {(H*8, W*8)}"
                # controlnet_image = F.interpolate(
                #     controlnet_image,
                #     size=(cond_scale_H, cond_scale_W),
                #     mode="bilinear",
                #     align_corners=False,
                # )

                tim.update_and_log('data prep')

                # --- 5. 准备 Camera (新增) ---
                # 你的 Dataset 似乎没有返回 camera，我们在这里生成 (假设视角配置固定)
                # 如果你的 Dataset 里有 camera，请从 batch 读取并处理
                # camera_matrix = get_camera(NUM_FRAMES, elevation=15, azimuth_start=0, azimuth_span=360, camera_distance=2.0)
                # camera_matrix = camera_matrix.to(device=accelerator.device, dtype=weight_dtype) # [4, 16]
                
                # # 扩展到 Batch 维度: [Batch * 4, 16]
                # camera_matrix = camera_matrix.repeat(bsz, 1)

                # Camera matrices from dataset: (B, F, 16) -> (B*F, 16)
                camera_matrix = batch["camera_matrix"].to(device=accelerator.device, dtype=weight_dtype)
                camera_matrix = camera_matrix.view(bsz * NUM_FRAMES, 16)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                tim.update_and_log('ctrl net')

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                    camera=camera_matrix, # <--- 必须传！
                )[0]

                tim.update_and_log('unet')

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                tim.update_and_log('loss')

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            # vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            tim.update_and_log('end loop')

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(f"{args.output_dir}/controlnet")
        unet = unwrap_model(unet)
        unet.save_pretrained(f"{args.output_dir}/unet")

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                # vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
