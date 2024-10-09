from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
)  # TODO: Other model requires different import
from PIL import Image
from datasets import load_dataset
import torch
import numpy as np
import json
import argparse
from io import BytesIO
from tqdm import tqdm
import os

MODEL_HANDLE = {
    "llava-hf/llava-v1.6-vicuna-7b-hf": "llava-1.6-7b",
    "llava-hf/llava-v1.6-vicuna-13b-hf": "llava-1.6-13b",
}


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
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        model.to("cuda:0")
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    if fp16:
        model.half()

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

    input = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **input,
        max_new_tokens=50,
        top_k=1,
    )

    out_with_template = processor.decode(output[0], skip_special_tokens=True)
    out = out_with_template[
        out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
    ]

    return out


def log_error(error_message, log_file="error.txt"):
    """Log error message to a text file."""
    with open(log_file, "a") as f:  # Append mode
        f.write(error_message + "\n")  # Write the error message with a newline


def main(task, qa_type, model_path, fp16, multi_gpu, limit=np.inf, st_idx=None, ed_idx=None):
    kb_data = get_kb_from_hf()
    url_jpg_map = get_url_jpg_map(kb_data)
    vqa_data = get_vqa_from_hf(task)
    model, processor = load_model_processor(model_path, fp16, multi_gpu)

    if st_idx is not None or ed_idx is not None:
        _ = len(vqa_data)
        vqa_data = vqa_data.iloc[st_idx:ed_idx]
        print(f"  > Total Data to Process: {len(vqa_data):8,}.  (of {_:,})")
        print(f"          Start-End Index:  {st_idx}  to  {ed_idx}")

    list_res = []
    count = 0
    error_counter = 0
    try:
        for _, row in tqdm(vqa_data.iterrows(), total=len(vqa_data)):
            res = {}
            try:
                image_file = url_jpg_map[row["image_url"]]
                query = (
                    row["multi_choice_prompt"]
                    if qa_type == "mc"
                    else row["open_ended_prompt"]
                )
                response = eval_instance(model, processor, image_file, query)

                res["qa_id"] = row["qa_id"]
                res["prediction"] = response
                res["lang"] = row["lang"]
                res["answer" if qa_type == "oe" else "multi_choice_answer"] = row[
                    "answer" if qa_type == "oe" else "multi_choice_answer"
                ]

                list_res.append(res)
            except Exception as e:
                error_message = f"Error at row {row['qa_id']}: {str(e)}"
                log_error(
                    error_message,
                    f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_error.txt",
                )
                export_result(
                    list_res,
                    f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_pred_cp{error_counter}.jsonl",
                )
                error_counter += 1

            count += 1
            if count == limit:
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Exporting latest results...")
        error_message = f"KeyboardInterrupt at row {row['qa_id']}"
        log_error(
            error_message,
            f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_error.txt",
        )
        export_result(
            list_res,
            f"./result/error/task{task}_{qa_type}_{MODEL_HANDLE[model_path]}_pred_interrupt.jsonl",
        )

    return list_res


def get_unique_filename(path: str):
    """Generate a unique file name by adding (1), (2), etc. if the file already exists."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{base} ({counter}){ext}"

    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base} ({counter}){ext}"

    return new_path


def export_result(result: list, path: str):
    """Export results to a file, ensuring unique file names."""
    path = get_unique_filename(path)
    with open(path, "w") as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write("\n")


if __name__ == "__main__":
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
    args = parser.parse_args()

    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = main(args.task, args.type, args.model_path, args.fp16, args.multi_gpu)
    export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred.jsonl",
    )
