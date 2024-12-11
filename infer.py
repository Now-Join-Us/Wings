from wings.utils import load_from_safetensors, set_seed
from wings.model.base_architecture import WingsMetaForCausalLM
from wings.arguments import ModelArguments, DataArguments, TrainingArguments

from abc import ABC, abstractmethod

import torch
import json
from typing import Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.quantization_config import QuantizationMethod

from PIL import Image

def load_image(image_file_path):
    return Image.open(image_file_path).convert('RGB')

class ModelWrapper(object):
    def __init__(self):
        self.force_use_generate = False

    def to(self, device):
        if hasattr(self, 'model') and not getattr(self.model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            try:
                self.model.to(device)
            except RuntimeError as e:
                pass
        return self

    def eval(self):
        if hasattr(self, 'model'):
            self.model.eval()
        return self

    def tie_weights(self):
        if hasattr(self, 'model') and hasattr(self.model, 'tie_weights'):
            self.model.tie_weights()
        return self

    def get_llm(self):
        if hasattr(self, 'model'):
            return self.model
        return None

    @abstractmethod
    def generate_text_only_from_token_id(self, conversation, **kwargs):
        raise NotImplementedError

    def is_overridden_generate_text_only_from_token_id(self, obj):
        return ModelWrapper.__dict__['generate_text_only_from_token_id'] is not obj.generate_text_only_from_token_id.__func__

    def _wrap_method(self, method):
        def wrapper(*args, **kwargs):
            return method(*args, **kwargs)
        return wrapper

    @abstractmethod
    def generate_text_only(self, conversation, **kwargs):
        raise NotImplementedError

    def is_overridden_generate_text_only(self, obj):
        return ModelWrapper.__dict__['generate_text_only'] is not obj.generate_text_only.__func__

    def generate_with_chat(self, tokenizer, conversation, history=[], **kwargs):
        response, _ = self.model.chat(tokenizer, conversation, history=history, **kwargs)
        return response

def retain_only_first_sub_str(s, sub_s):
    first_index = s.find(sub_s)

    if first_index != -1:
        s = s[:first_index + len(sub_s)] + s[first_index + len(sub_s):].replace(sub_s, '')
    return s

def remove_image_token(instruction, tokens):
    for src in tokens:
        if src in instruction:
            instruction = instruction.replace(src, '')
    return instruction

def replace_image_token(instruction, source_default_tokens, target_tokens, leaved_token_num=1):
    if isinstance(target_tokens, str):
        target_tokens = [target_tokens] * len(source_default_tokens)
    target_id = 0
    for src in source_default_tokens:
        if src in instruction:
            instruction = instruction.replace(src, target_tokens[target_id])
            instruction = retain_only_first_sub_str(instruction, target_tokens[target_id])
            target_id += 1
        if target_id >= leaved_token_num:
            break
    if target_id == 0 and len(target_tokens) > 0:
        instruction = target_tokens[0] + '\n' + instruction
    else:
        instruction = remove_image_token(instruction, source_default_tokens)
    return instruction


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

class Wings(ModelWrapper):
    def __init__(self, args_json_path, local_model_path):
        super().__init__()
        set_seed(42)

        with open(args_json_path) as json_file:
            config = json.load(json_file)
        if local_model_path is not None:
            config['model_args']['model_safetensors_load_path'] = local_model_path

        local_model_args = ModelArguments(**config['model_args'])
        data_args = DataArguments(**config['data_args'])
        training_args = TrainingArguments(**config['training_args'])

        self.model, self.tokenizer, self.conversation_formatter = WingsMetaForCausalLM.build(
            model_name=local_model_args.model_name,
            model_path=local_model_args.model_path,
            conversation_formatter_kwargs={
                'system_slot': local_model_args.system_slot,
                'user_slot': local_model_args.user_slot,
                'gpt_slot': local_model_args.gpt_slot,
                'eot': local_model_args.eot
            },
            model_max_length=local_model_args.model_max_length
        )

        self.model.get_model().initialize_vision_modules(
            model_args=local_model_args,
            fsdp=training_args.fsdp
        )

        if hasattr(self.model, 'initialize_modules'):
            self.model.initialize_modules(
                model_args=local_model_args,
                data_args=data_args,
                training_args=training_args,
            )
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_max_length = self.tokenizer.model_max_length

        if local_model_args.model_safetensors_load_path is not None:
            self.model.load_state_dict(load_from_safetensors(self.model, local_model_args.model_safetensors_load_path))

        vision_tower = self.model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        self.model.to(torch.bfloat16)

    def preprocess_vqa(self, data):
        instruction = replace_image_token(
            instruction=data['prompt_instruction'],
            source_default_tokens=['<image 1>', '<image 2>', '<image 3>', '<image 4>'],
            target_tokens='<image>',
            leaved_token_num=1
        )

        conversation = (instruction, data['image_preloaded'][0])

        return conversation

    def generate_vqa(self, conversation, **kwargs):
        instruction, image = conversation
        image_processor = getattr(self.model.get_vision_tower(), 'image_processor', None)
        if image is not None:
            image_tensor = process_images([image], image_processor, self.model.config).cuda()
        else:
            image_tensor = None

        prompt, input_ids = self.conversation_formatter.format_query(instruction)
        do_sample = False
        input_ids = input_ids.unsqueeze(0).cuda()
        with torch.inference_mode():
            kwargs = dict(
                images=image_tensor,
                do_sample=False,
                num_beams=1,
                max_new_tokens=32,
                repetition_penalty=None,
                use_cache=True
            )
            output_ids = self.model.generate(
                input_ids,
                **kwargs
                )
        input_token_len = input_ids.shape[1]
        response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        return response

    def generate_text_only_from_token_id(
        self,
        input_ids: torch.LongTensor = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        llm = self.get_llm()
        outputs = llm(input_ids=input_ids)

        return CausalLMOutputWithPast(
            logits=outputs[1]
        )


import argparse

def main():
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_model_path', '-p', type=str, default=None, help='Path to the safetensors'
    )
    args = parser.parse_args()

    data = {
        "prompt_instruction": "<image 1>\nWhat color is this image?",
        "image_path": [
            './data/images/red.jpg'
        ]
    }
    data['image_preloaded'] = [load_image(data['image_path'][0])]

    model_wrapper = Wings("./run/infer_args.json", args.local_model_path)
    model_wrapper.to(device).eval().tie_weights()

    print(f'Question: {data["prompt_instruction"]} (image(s): {data["image_path"]})')
    print(f'Response: {model_wrapper.generate_vqa(model_wrapper.preprocess_vqa(data))}')

if __name__ == '__main__':
    main()
