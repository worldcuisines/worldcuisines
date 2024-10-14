from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)
from PIL import Image
import torch
import argparse
import os

import base_model

MODEL_HANDLE = {
    "microsoft/Phi-3.5-vision-instruct": "Phi-3.5-vision-instruct",
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


def load_model_processor(model_path, bp16=True, multi_gpu=False):
    processor = processor = AutoProcessor.from_pretrained(model_path, 
      trust_remote_code=True, 
      num_crops=16
    ) 
    if not multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(
          model_path, 
          device_map="cuda", 
          trust_remote_code=True, 
          torch_dtype=(torch.bfloat16 if bp16 else "auto"),
          _attn_implementation='eager'    
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=(torch.bfloat16 if bp16 else "auto"),
            device_map="auto",
        )

    return model, processor


def eval_instance(
    model, processor, image_file, query
):  # TODO: Make this versatile to other model
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    
    images = [image_file]
    
    messages = [
        {"role": "user", "content": f"<|image_{1}|>\n"+query},
    ]
    
    prompt = processor.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )
    
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 
    
    generation_args = { 
        "max_new_tokens": 50, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 
    
    generate_ids = model.generate(**inputs, 
      eos_token_id=processor.tokenizer.eos_token_id, 
      **generation_args
    )
    
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=False)[0] 
    
    return response


if __name__ == "__main__":
    # operator overloading
    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    parser = argparse.ArgumentParser(description="Phi Evaluation")
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
        help="Use float32 precision rather than bfloat16.",
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
