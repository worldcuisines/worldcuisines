from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor
)
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
from io import BytesIO
import os
import json
from PIL import Image
import argparse
from tqdm import tqdm

MODEL_HANDLE = {
    # TODO: Change model
    "Qwen/Qwen2-VL-2B-Instruct": "qwen-vl-2b"
}

def set_all_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_kb_from_hf():
    ds = load_dataset("world-cuisines/kb")
    data = ds["main"].to_pandas()
    return data

def get_url_jpg_map(data):
    url_jpg_map = {}
    for _, row in data.iterrows():
        for i in range(1, 9):
            if row[f"image{i}_url"] is not None:
                url_jpg_map[row[f"image{i}_url"]] = Image.open(
                    BytesIO(row[f"image{i}"]["bytes"])
                )
    return url_jpg_map

def get_vqa_from_hf(task):
    if task == 1:
        vqa = load_dataset("world-cuisines/vqa", name="task1")
    elif task == 2:
        vqa = load_dataset("world-cuisines/vqa", name="task2")
    else:
        raise ValueError(
            "Invalid task number. Please choose from 1 (dish name) or 2 (region)"
        )

    vqa_data = vqa["test_large"].to_pandas()
    return vqa_data

def load_model_processor(model_path, fp16=True, multi_gpu=False):
    min_pixels = 200 * 200
    max_pixels = 1600 * 1200
    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if fp16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto" if multi_gpu else None
    )
    model.to("cuda:0")
    return model, processor

def eval_instance(model, processor, image_file, query):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_file},
                {"type": "text", "text": query},
            ]
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

def log_error(error_message, log_file="error.txt"):
    with open(log_file, "a") as f:
        f.write(error_message + "\n")

def main(task, qa_type, model_path, fp16, multi_gpu, limit=np.inf, st_idx=None, ed_idx=None):
    set_all_seed()
    kb_data = get_kb_from_hf()
    url_jpg_map = get_url_jpg_map(kb_data)
    vqa_data = get_vqa_from_hf(task)
    model, processor = load_model_processor(model_path, fp16, multi_gpu)

    if st_idx is not None or ed_idx is not None:
        vqa_data = vqa_data.iloc[st_idx:ed_idx]

    list_res = []
    count = 0
    error_counter = 0

    try:
        for _, row in tqdm(vqa_data.iterrows(), total=len(vqa_data)):
            res = {}
            try:
                image_file = url_jpg_map[row["image_url"]]
                query = row["multi_choice_prompt"] if qa_type == "mc" else row["open_ended_prompt"]
                response = eval_instance(model, processor, image_file, query)

                res["qa_id"] = row["qa_id"]
                res["prediction"] = response
                res["lang"] = row["lang"]
                res["answer"] = row["answer" if qa_type == "oe" else "multi_choice_answer"]

                list_res.append(res)
            except Exception as e:
                error_message = f"Error at row {row['qa_id']}: {str(e)}"
                log_error(error_message, f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_error.txt")
                export_result(list_res, f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_pred_cp{error_counter}.jsonl")
                error_counter += 1

            count += 1
            if count == limit:
                break

    except KeyboardInterrupt:
        error_message = f"KeyboardInterrupt at row {row['qa_id']}"
        log_error(error_message, f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_error.txt")
        export_result(list_res, f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_pred_interrupt.jsonl")

    return list_res

def export_result(result, path):
    path = get_unique_filename(path)
    with open(path, "w") as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write("\n")

def get_unique_filename(path: str):
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base} ({counter}){ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base} ({counter}){ext}"

    return new_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL Evaluation")
    parser.add_argument("--task", type=int, default=1, help="Task number to evaluate (1 or 2)")
    parser.add_argument("--type", type=str, default="mc", help="Type of question to evaluate (oe or mc)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model path")
    parser.add_argument("--fp32", action="store_true", default=False, help="Use float32 instead of float16")
    parser.add_argument("--multi_gpu", action="store_true", default=False, help="Use multiple GPUs")
    
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = main(args.task, args.type, args.model_path, not args.fp32, args.multi_gpu)
    export_result(result, f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred.jsonl")
    
    # EXAMPLE COMMAND

    # python qwen.py --task 1 --type mc
    # python qwen.py --task 1 --type oe
    # python qwen.py --task 2 --type oe

    # etc.
