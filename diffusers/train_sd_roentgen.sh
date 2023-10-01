#!/bin/sh
# code from: https://huggingface.co/docs/diffusers/training/text2image to fine tune the stable diffusion pipeline
# we stick to the default parameters because there were issues for example with different mixed_precision

export CUDA_VISIBLE_DEVICES=3

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_dir="local"

accelerate launch --mixed_precision="fp16" mod_train_text_to_image.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$dataset_dir" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=60000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="roentGen_sd_lessnF" 

