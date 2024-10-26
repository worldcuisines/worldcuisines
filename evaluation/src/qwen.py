from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    MIN_PX = 200 * 200
    MAX_PX = 1600 * 1200

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=MIN_PX, max_pixels=MAX_PX
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if not fp32 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto" if multi_gpu else "cuda:0",
    )
    return model, processor


def eval_instance(model, processor, image_file, query, tokenizer=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file},
                {"type": "text", "text": query},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

    out_with_template = processor.batch_decode(output, skip_special_tokens=True)[0]
    out = out_with_template[
        out_with_template.index("\nassistant\n") + len("\nassistant\n") :
    ]

    return out
