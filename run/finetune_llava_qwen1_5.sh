LOG_DIR=your_log_dir
TIME_STR=$(date +"%m_%d_%H_%M_%S")
SAVE_DIR=$LOG_DIR/$TIME_STR
echo $SAVE_DIR

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 2306 main.py \
    --deepspeed ./run/zero2.json \
    --model_name llava_qwen2 \
    --model_path your_model_path/Qwen1.5-7B-Chat \
    --data_name 'demo_finetune' \
    --is_multimodal True \
    --weight_decay 0. \
    --tune_llm True \
    --image_aspect_ratio pad \
    --vision_tower your_model_path/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_only_mm_mlp_adapter False \
    --mm_projector_lr 1e-5 \
    --learning_rate 2e-5 \
    --vision_tower_lr_follow_mm_projector True \
    --pretrain_mm_mlp_adapter your_pretrained_projector_path/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --bf16 True \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none
