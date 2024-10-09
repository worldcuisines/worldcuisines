import os
import torch
import argparse
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

import eval

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_HANDLE = {
    'allenai/Molmo-7B-D-0924': "molmo-7b-d",
    'allenai/Molmo-7B-O-0924': "molmo-7b-o",
}


def load_model_processor(model_path, fp16=True, multi_gpu=False):
    assert model_path in MODEL_HANDLE.keys(), \
        f"Model path is not valid.\n\tAllowed keys: {list(MODEL_HANDLE.keys())}\n\t(but Found: '{model_path}')"
    dtype = torch.float16 if fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map='auto',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=device if not multi_gpu else 'auto',
    )
    return model, processor


def eval_instance(model, processor, image_file, query):
    inputs = processor.process(images=[image_file], text=query, )
    batch = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
    batch["images"] = batch["images"].to(model.dtype)

    output = model.generate_from_batch(
        batch,
        GenerationConfig(max_new_tokens=50, stop_strings="<|endoftext|>", top_k=1, do_sample=True),
        tokenizer=processor.tokenizer,
    )
    generated_tokens = output[0, batch['input_ids'].size(1):]
    out = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return out


if __name__ == "__main__":
    # operator overloading
    eval.load_model_processor = load_model_processor
    eval.eval_instance = eval_instance
    eval.MODEL_HANDLE = MODEL_HANDLE

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
        default=None,
        help="Path to the pretrained model",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 precision if supported",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        default=False,
        help="Use multiple gpu if needed",
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
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = eval.main(args.task, args.type, args.model_path, args.fp16, args.multi_gpu,
                       st_idx=args.st_idx, ed_idx=args.ed_idx,)
    eval.export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred.{os.getenv('SLURM_JOB_ID', 'nan')}.jsonl",
    )
