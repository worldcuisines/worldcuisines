from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)
from PIL import Image
import torch
import argparse
import os

import base_model

MODEL_HANDLE = {
    "llava-hf/llava-v1.6-vicuna-7b-hf": "llava-1.6-7b",
    "llava-hf/llava-v1.6-vicuna-13b-hf": "llava-1.6-13b",
}


def get_prompt(vqa_data, url_jpg_map, type, qa_id):
    row = vqa_data[vqa_data["qa_id"] == qa_id]
    image = url_jpg_map[row["image_url"][0]]
    image_file = Image.open(image)

    if type == "mc":
        prompt = row["multi_choice_prompt"][0]
    elif type == "oe":
        prompt = row["open_ended_prompt"][0]
    else:
        raise ValueError("Invalid type of question. Please choose from 'mc' or 'oe'")

    return image_file, prompt


def load_model_processor(model_path, fp16=True, multi_gpu=False):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    if not multi_gpu:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=(torch.float16 if fp16 else torch.float32),
            low_cpu_mem_usage=True
        )
        model.to("cuda:0")
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=(torch.float16 if fp16 else torch.float32),
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    return model, processor


def eval_instance(
    model, processor, image_file, query
):  # TODO: Make this versatile to other model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    out_with_template = processor.decode(output[0], skip_special_tokens=True)
    out = out_with_template[
        out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
    ]

    return out


if __name__ == "__main__":
    # operator overloading
    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    parser = argparse.ArgumentParser(description="LLaVA Evaluation")
    parser.add_argument(
        "--task", type=int, default=1, help="Task number to evaluate (1 or 2)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="mc",
        help="Type of question to evaluate (open ended [oe] or multiple choice [mc])",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="llava-hf/llava-v1.6-vicuna-7b-hf",
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        default=False,
        help="Use float32 precision rather than float16.",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Use multiple gpu if needed",
    )
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = base_model.main(args.task, args.type, args.model_path, not(args.fp32), args.multi_gpu)
    base_model.export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred.jsonl",
    )
