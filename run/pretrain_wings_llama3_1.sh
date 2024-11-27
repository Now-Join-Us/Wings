LOG_DIR=your_log_dir
TIME_STR=$(date +"%m_%d_%H_%M_%S")
SAVE_DIR=$LOG_DIR/$TIME_STR
echo $SAVE_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 2306 main.py \
    --deepspeed ./run/zero2.json \
    --model_name wings_llama3_1 \
    --model_path your_model_path/Meta-Llama-3.1-8B-Instruct \
    --data_name 'demo_pretrain' \
    --wings_router_type pretrain \
    --tuned_keys .attn_pool. \
    --is_multimodal True \
    --weight_decay 1e-9 \
    --tune_llm False \
    --image_aspect_ratio pad \
    --vision_tower your_model_path/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_only_mm_mlp_adapter False \
    --learning_rate 2e-4 \
    --vision_tower_lr_follow_mm_projector False \
    --attn_layers_idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
    --mm_vision_select_layer -2 \
    --bf16 True \
    --lora_dim 64 \
    --output_dir $SAVE_DIR \
    --group_by_modality_length False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
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
    --system_prompt_length 17 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none \
    --system_slot '<|start_header_id|>user<|end_header_id|>\n\n' \
    --user_slot '<|start_header_id|>system<|end_header_id|>\n\n' \
    --gpt_slot '<|start_header_id|>assistant<|end_header_id|>\n\n' \
    --eot '<|eot_id|>'
