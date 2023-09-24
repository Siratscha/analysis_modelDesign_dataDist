$env:MODEL_NAME = "CompVis/stable-diffusion-v1-4"
$env:dataset_dir = "local"

accelerate launch --mixed_precision="fp16" mod_train_text_to_image.py `
  --pretrained_model_name_or_path=$env:MODEL_NAME `
  --train_data_dir=#$env:dataset_dir `
  --use_ema `
  --resolution=512 --center_crop --random_flip `
  --train_batch_size=1 `
  --gradient_accumulation_steps=4 `
  --gradient_checkpointing `
  --max_train_steps=15000 `
  --learning_rate=1e-05 `
  --max_grad_norm=1 `
  --lr_scheduler="constant" --lr_warmup_steps=0 `
  --output_dir="roentGen-sd" `


#Invoke-Expression -Command $command
Read-Host "Press Enter to exit"
