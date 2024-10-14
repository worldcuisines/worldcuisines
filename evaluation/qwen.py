from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor
)
from qwen_vl_utils import process_vision_info
import torch
import os
import argparse

import base_model

MODEL_HANDLE = {
    "Qwen/Qwen2-VL-2B-Instruct": "qwen-vl-2b",
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2-vl-7b-instruct",
    "Qwen/Qwen2-VL-72B-Instruct": "qwen2-vl-72b",
}


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
        device_map="auto" if multi_gpu else "cuda:0"
    )
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


if __name__ == "__main__":
    # operator overloading
    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    parser = argparse.ArgumentParser(description="Qwen2-VL Evaluation")
    parser.add_argument("--task", type=int, default=1, help="Task number to evaluate (1 or 2)")
    parser.add_argument("--type", type=str, default="mc", help="Type of question to evaluate (oe or mc)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-72B-Instruct", help="Model path")
    parser.add_argument("--fp32", action="store_true", default=False, help="Use float32 instead of float16")
    parser.add_argument("--multi_gpu", action="store_true", default=False, help="Use multiple GPUs")
    parser.add_argument("-n", "--chunk_num", type=int, default=1, help="Chunks count")
    parser.add_argument("-k", "--chunk_id", type=int, default=0, help="Chunk ID (0-based)")
    parser.add_argument("-s", "--st_idx", default=None, type=int, help="Slice data, start index (inclusive).")
    parser.add_argument("-e", "--ed_idx", default=None, type=int, help="Slice data, end index (exclusive).")

    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = base_model.main(args.task, args.type, args.model_path, not args.fp32, args.multi_gpu,
                             st_idx=args.st_idx, ed_idx=args.ed_idx, chunk_num=args.chunk_num, chunk_id=args.chunk_id)
    base_model.export_result(result, f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred" +
                                     ("" if args.st_idx is None else f"_s{args.st_idx}") + ("" if args.ed_idx is None else f"_e{args.ed_idx}") +
                                     (f".chunk{args.chunk_id}_of_{args.chunk_num}" if not ((args.chunk_num == 1) and (args.chunk_id == 0)) else "") +
                                     (f".{os.getenv('SLURM_JOB_ID')}" if 'SLURM_JOB_ID' in os.environ else "") +
                                     f".jsonl")
    
    # EXAMPLE COMMAND

    # python qwen.py --task 1 --type mc
    # python qwen.py --task 1 --type oe
    # python qwen.py --task 2 --type oe

    # etc.
