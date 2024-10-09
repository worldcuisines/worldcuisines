## PLAN
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from datasets import load_dataset
import torch
import numpy as np
import json
import argparse
from io import BytesIO

from tqdm import tqdm
import os


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


def load_model_processor(
    model_path, fp16=True, multi_gpu=False
):  # TODO: Make this versatile for other model
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

    return model, processor


def eval_instance(model, processor, image_file, query):
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
        max_length=1000,
        top_k=1,
    )

    out_with_template = processor.decode(output[0], skip_special_tokens=True)
    out = out_with_template[
        out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
    ]

    return out

def main(task, type, model_path, fp16, multi_gpu, limit=np.inf):
    kb_data = get_kb_from_hf()
    url_jpg_map = get_url_jpg_map(kb_data)
    vqa_data = get_vqa_from_hf(task)
    model, processor = load_model_processor(model_path, fp16, multi_gpu)

    list_res = []
    count = 0
    error_counter = 0
    for _, row in tqdm(vqa_data.iterrows(), total=len(vqa_data)):
        res = {}
        try:
            image_file = url_jpg_map[row["image_url"]]
            query = (
                row["multi_choice_prompt"] if type == "mc" else row["open_ended_prompt"]
            )
            response = eval_instance(model, processor, image_file, query)

            res["qa_id"] = row["qa_id"]
            res["prediction"] = response
            res["lang"] = row["lang"]
            res["answer" if type == "oe" else "multi_choice_answer"] = row[
                "answer" if type == "oe" else "multi_choice_answer"
            ]

            list_res.append(res)
        except Exception as e:
            export_result(
                res,
                f"./result/error/task{task}_{type}_{model_path}_pred_error{error_counter}.jsonl",
            )
            error_counter += 1

        count += 1
        if count == limit:
            break

    return list_res


def export_result(result: list, path: str):
    with open(path, "w") as outfile:
        for entry in result:
            json.dump(entry, outfile)
            outfile.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Evaluation")
    parser.add_argument("--task", type=int, default=1, help="Task number to evaluate")
    parser.add_argument(
        "--type", type=str, default="mc", help="Type of question to evaluate"
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

    # os mkdir if no result folder
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/error"):
        os.makedirs("./result/error")

    result = main(args.task, args.type, args.model_path, args.fp16, args.multi_gpu)
    export_result(
        result, f"./result/task{args.task}_{args.type}_{args.model_path}_pred.jsonl"
    )
