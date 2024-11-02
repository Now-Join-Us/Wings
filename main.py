import pathlib
import torch

import transformers

from wings.utils import log_args, logger, load_from_safetensors, safe_save_model_for_hf_trainer, set_gradient, set_seed
from wings.configs import BEGIN_LINE, END_LINE
from wings.model.base_architecture import WingsMetaForCausalLM
from wings.trainer import WingsTrainer
from wings.dataloader.base import make_supervised_data_module
from wings.arguments import ModelArguments, DataArguments, TrainingArguments

def train():
    set_seed(42)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with training_args.main_process_first(local=False):
        if torch.cuda.current_device() == 0:
            log_args(model_args, data_args, training_args)

    model, tokenizer, conversation_formatter = WingsMetaForCausalLM.build(
        model_name=model_args.model_name,
        model_path=model_args.model_path,
        conversation_formatter_kwargs={
            'system_slot': model_args.system_slot,
            'user_slot': model_args.user_slot,
            'gpt_slot': model_args.gpt_slot,
            'eot': model_args.eot
        },
        model_max_length=model_args.model_max_length
    )

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    if hasattr(model, 'initialize_modules'):
        model.initialize_modules(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_max_length = tokenizer.model_max_length

    if model_args.model_safetensors_load_path is not None:
        model.load_state_dict(load_from_safetensors(model, model_args.model_safetensors_load_path))

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    set_gradient(model, vision_tower, training_args)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    if torch.cuda.current_device() == 0:
        logger.info('Tune:')
        logger.info('  '.join([name for name, param in model.named_parameters() if param.requires_grad]))

    data_module = make_supervised_data_module(
        data_args=data_args,
        tokenizer=tokenizer,
        conversation_formatter=conversation_formatter
    )

    trainer = WingsTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    train()
