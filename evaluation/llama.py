from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import argparse
import os

import base_model

MODEL_HANDLE = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "llama-3.2-90b",
}


def load_model_processor(model_path, fp16=True, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(model_path)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if fp16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.tie_weights()
    # if fp16:
    #     model.half()

    return model, processor


def eval_instance(
    model, processor, image_file, query
):  # TODO: Make this versatile to other model
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        top_k=1,
    )
    out = processor.decode(output[0][inputs['input_ids'].size(1):], skip_special_tokens=True)

    return out


if __name__ == "__main__":
    # operator overloading
    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    parser = argparse.ArgumentParser(description="LLaMA Evaluation")
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
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use float16 precision if supported",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Use multiple gpu if needed",
    )
    parser.add_argument(
        "--chunk_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = base_model.main(args.task, args.type, args.model_path, args.fp16, args.multi_gpu, chunk_size=args.chunk_num, chunk_num=args.chunk_id-1)
    base_model.export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_chunk{args.chunk_id}_of_{args.chunk_num}_pred.jsonl",
    )