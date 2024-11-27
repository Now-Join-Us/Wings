LOG_DIR=your_log_dir
TIME_STR=$(date +"%m_%d_%H_%M_%S")
SAVE_DIR=$LOG_DIR/$TIME_STR
echo $SAVE_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 2306 main.py \
    --deepspeed ./run/zero2.json \
    --model_name wings_qwen2 \
    --model_path your_model_path/Qwen1.5-7B-Chat \
    --data_name 'demo_finetune' \
    --wings_router_type linear \
    --tuned_keys .attn_pool. .attn_t_pool. .reweight_module. \
    --model_safetensors_load_path your_pretrained_model_path/00_00_00_00_00 \
    --lr_projector_follow_tuned_keys 'mm_projector' \
    --is_multimodal True \
    --weight_decay 1e-8 \
    --tune_llm True \
    --image_aspect_ratio pad \
    --vision_tower your_model_path/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_only_mm_mlp_adapter False \
    --mm_projector_lr 1e-5 \
    --learning_rate 2e-6 \
    --vision_tower_lr_follow_mm_projector True \
    --attn_layers_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --lora_dim 64 \
    --output_dir $SAVE_DIR \
    --group_by_modality_length False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --tf32 True \
    --image_token_length 729 \
    --system_prompt_length 14 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none
