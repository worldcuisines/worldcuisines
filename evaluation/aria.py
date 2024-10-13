from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import argparse
import os

import base_model

MODEL_HANDLE = {
    "rhymes-ai/Aria": "aria-25B-moe-4B",
}


def load_model_processor(model_path, fp16=True, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    # if fp16:
    #     model.half()

    return model, processor


def eval_instance(model, processor, image_file, query):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "text", "text": query},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image_file, text=prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=True,
            top_k=1,
        )
        out = processor.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    return out


if __name__ == "__main__":
    # operator overloading
    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    parser = argparse.ArgumentParser(description="Aria Evaluation")
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
        default="rhymes-ai/Aria",
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "-s", "--st_idx",
        default=None,
        type=int,
        help="Slicing index to evaluate, start index (inclusive).",
    )
    parser.add_argument(
        "-e", "--ed_idx",
        default=None,
        type=int,
        help="Slicing index to evaluate, end index (exclusive).",
    )
    parser.add_argument(
        "-n", "--chunk_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-k", "--chunk_id",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = base_model.main(args.task, args.type, args.model_path, True, False,
                             # chunk_size=args.chunk_size, chunk_num=args.chunk_id,
                             st_idx=args.st_idx, ed_idx=args.ed_idx,
                             )
    base_model.export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_chunk{args.chunk_id}_of_{args.chunk_size}_pred.jsonl",
    )