#!/bin/bash
sourcedir="/home/lex/Software/diffusers/examples/unconditional_image_generation"

accelerate launch "${sourcedir}"/train_unconditional.py \
  --dataset_name=/home/lex/Research/RiffusionGen/create_ghuzeng_ds/spectrograms \
  --resolution=512 \
  --output_dir=./holloway_guzheng_512 \
  --train_batch_size=1\
  --num_epochs=2 \
  --gradient_accumulation_steps=1 \
  --enable_xformers_memory_efficient_attention \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=bf16\


  #--gradient_checkpointing \
  #--use_8bit_adam \
  #--set_grads_to_none \
#holloway_guzheng_512/