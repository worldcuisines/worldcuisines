import argparse
import os
import src.base_model as base_model

MODEL_HANDLE = {
    "rhymes-ai/Aria": "aria-25B-moe-4B",
    
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "llama-3.2-90b",
    
    "llava-hf/llava-v1.6-vicuna-7b-hf": "llava-1.6-7b",
    "llava-hf/llava-v1.6-vicuna-13b-hf": "llava-1.6-13b",
    
    'allenai/MolmoE-1B-0924': "molmoe-1b",
    'allenai/Molmo-7B-D-0924': "molmo-7b-d",
    'allenai/Molmo-7B-O-0924': "molmo-7b-o",

    "microsoft/Phi-3.5-vision-instruct": "Phi-3.5-vision-instruct",

    "Qwen/Qwen2-VL-2B-Instruct": "qwen-vl-2b",
    "Qwen/Qwen2-VL-7B-Instruct": "qwen2-vl-7b-instruct",
    "Qwen/Qwen2-VL-72B-Instruct": "qwen2-vl-72b",

    "mistralai/Pixtral-12B-2409": "pixtral-12b",

    # TODO: add Proprietary models (if possible)
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2-VL Evaluation")
    parser.add_argument("--task", type=int, default=1, help="Task number to evaluate (1 or 2)")
    parser.add_argument("--type", type=str, default="mc", help="Type of question to evaluate (oe or mc)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-72B-Instruct", help="Model path")
    parser.add_argument("--fp32", action="store_true", default=False, help="Use float32 instead of float16/bfloat16")
    parser.add_argument("--multi_gpu", action="store_true", default=False, help="Use multiple GPUs")
    parser.add_argument("-n", "--chunk_num", type=int, default=1, help="Chunks count")
    parser.add_argument("-k", "--chunk_id", type=int, default=0, help="Chunk ID (0-based)")
    parser.add_argument("-s", "--st_idx", default=None, type=int, help="Slice data, start index (inclusive).")
    parser.add_argument("-e", "--ed_idx", default=None, type=int, help="Slice data, end index (exclusive).")
    parser.add_argument("--test", action="store_true", help="Set to true if in test mode.")

    args = parser.parse_args()

    assert args.model_path in MODEL_HANDLE, f"Model path {args.model_path} not found in MODEL_HANDLE"

    if args.model_path == "rhymes-ai/Aria":
        from src.aria import load_model_processor
        from src.aria import eval_instance
    elif args.model_path.startswith("meta-llama"):
        from src.llama import load_model_processor
        from src.llama import eval_instance
    elif args.model_path.startswith("llava-hf"):
        from src.llava import load_model_processor
        from src.llava import eval_instance
    elif args.model_path.startswith("allenai/Molmo"):
        from src.molmo import load_model_processor
        from src.molmo import eval_instance
    elif args.model_path.startswith("microsoft/Phi"):
        from src.phi import load_model_processor
        from src.phi import eval_instance
    elif args.model_path.startswith("Qwen"):
        from src.qwen import load_model_processor
        from src.qwen import eval_instance
    elif args.model_path.startswith("mistralai/Pixtral"):
        raise NotImplementedError("Pixtral is not implemented yet")
    else:
        raise NotImplementedError(f"{args.model_path} not implemented yet")

    base_model.load_model_processor = load_model_processor
    base_model.eval_instance = eval_instance
    base_model.MODEL_HANDLE = MODEL_HANDLE

    RESULT_PATH = './result' if not args.test else './test'
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    if not os.path.exists(f"{RESULT_PATH}/error"):
        os.makedirs(f"{RESULT_PATH}/error")

    result = base_model.main(
        task=args.task,
        qa_type=args.type,
        model_path=args.model_path,
        fp32=args.fp32,
        multi_gpu=args.multi_gpu,
        st_idx=args.st_idx,
        ed_idx=args.ed_idx,
        chunk_num=args.chunk_num,
        chunk_id=args.chunk_id,
    )
    
    base_model.export_result(
        result,
        f"{RESULT_PATH}/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred"  # ./result/task1_mc_qwen2-vl-72b_pred
        + ("" if args.st_idx is None else f"_s{args.st_idx}")  # ./result/task1_mc_qwen2-vl-72b_pred_s0
        + ("" if args.ed_idx is None else f"_e{args.ed_idx}")  # ./result/task1_mc_qwen2-vl-72b_pred_s0_e100
        + (f".chunk{args.chunk_id}_of_{args.chunk_num}" if not ((args.chunk_num == 1) and (args.chunk_id == 0)) else "")  # ./result/task1_mc_qwen2-vl-72b_pred_s0_e100.chunk0_of_1
        + (f".{os.getenv('SLURM_JOB_ID')}" if 'SLURM_JOB_ID' in os.environ else "")  # ./result/task1_mc_qwen2-vl-72b_pred_s0_e100.chunk0_of_1.1234
        + ".jsonl" # ./result/task1_mc_qwen2-vl-72b_pred_s0_e100.chunk0_of_1.1234.jsonl
    )