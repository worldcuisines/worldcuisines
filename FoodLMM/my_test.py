from tqdm import tqdm
from transformers import BertTokenizer, BertModel
# from utils.mapping import get_bert_embedding, mapping_ingre, find_similar_ingredient, ingredients_embedding
import argparse
import os
from io import BytesIO
import sys
import time
import cv2
from PIL import Image
import numpy as np
import torch
import json
import requests
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from utils import conversation as conversation_lib
from model.LISA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.config import Config
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

MODEL_HANDLE = {
    "llava-hf/llava-v1.6-vicuna-7b-hf": "llava-1.6-7b",
    "llava-hf/llava-v1.6-vicuna-13b-hf": "llava-1.6-13b",
    "Yueha0/FoodLMM-Chat": "FoodLMM",
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



def eval_instance(
    prompt,
    image_file,
    clip_image_processor,
    transform,
    tokenizer,
    model,
    args,

):

    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = np.array(image_file)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    output_ids, pred_masks, nurtition_output_dict = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split('ASSISTANT: ')[1].replace('.', '').replace('</s>', '').replace('It is', '')
    return text_output


def log_error(error_message, log_file="error.txt"):
    """Log error message to a text file."""
    with open(log_file, "a") as f:  # Append mode
        f.write(error_message + "\n")  # Write the error message with a newline


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x



def load_model_processor(args):

    configs = Config(args)
    # configs.pretty_print_system()
    args = configs.args
    args = argparse.Namespace(**args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.mass_token_idx = tokenizer("[MASS_TOTAL]", add_special_tokens=False).input_ids[0]
    args.calorie_token_idx = tokenizer("[CAL_TOTAL]", add_special_tokens=False).input_ids[0]
    args.fat_token_idx = tokenizer("[FAT_TOTAL]", add_special_tokens=False).input_ids[0]
    args.carbohydrate_token_idx = tokenizer("[CARB_TOTAL]", add_special_tokens=False).input_ids[0]
    args.protein_token_idx = tokenizer("[PRO_TOTAL]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {
        "torch_dtype": torch_dtype,
        "seg_token_idx": args.seg_token_idx,
        "mass_token_idx": args.mass_token_idx,
        "calorie_token_idx": args.calorie_token_idx,
        "fat_token_idx": args.fat_token_idx,
        "carbohydrate_token_idx": args.carbohydrate_token_idx,
        "protein_token_idx": args.protein_token_idx,
    }

    args_dict = vars(args)
    for i in range(1, args.max_seg_num + 1):
        args_dict['seg_token_idx_%s' % i] = tokenizer("[SEG{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['mass_token_idx_%s' % i] = tokenizer("[MASS{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['calorie_token_idx_%s' % i] = tokenizer("[CAL{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['fat_token_idx_%s' % i] = tokenizer("[FAT{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['carbohydrate_token_idx_%s' % i] = tokenizer("[CARB{}]".format(i), add_special_tokens=False).input_ids[0]
        args_dict['protein_token_idx_%s' % i] = tokenizer("[PRO{}]".format(i), add_special_tokens=False).input_ids[0]

        kwargs.update({
            'seg_token_idx_%s' % i: args_dict['seg_token_idx_%s' % i],
            'mass_token_idx_%s' % i: args_dict['mass_token_idx_%s' % i],
            'calorie_token_idx_%s' % i: args_dict['calorie_token_idx_%s' % i],
            'fat_token_idx_%s' % i: args_dict['fat_token_idx_%s' % i],
            'carbohydrate_token_idx_%s' % i: args_dict['carbohydrate_token_idx_%s' % i],
            'protein_token_idx_%s' % i: args_dict['protein_token_idx_%s' % i],
        })
    args = argparse.Namespace(**args_dict)

    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.model_path, low_cpu_mem_usage=True, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    return model, tokenizer, clip_image_processor, transform




# def eval_instance(prompt, image_file, clip_image_processor, transform, tokenizer, model, args,):
    
#     text_output = get_text_output(prompt,image_file,clip_image_processor,transform,tokenizer,model,args)

#     out = text_output.split('ASSISTANT: ')[1].replace('.', '').replace('</s>', '').replace('It is', '').split(", ")
#     print(out[0])


def log_error(error_message, log_file="error.txt"):
    """Log error message to a text file."""
    with open(log_file, "a") as f:  # Append mode
        f.write(error_message + "\n")  # Write the error message with a newline


def main(task, type, model_path, args, limit=np.inf):
    kb_data = get_kb_from_hf()
    url_jpg_map = get_url_jpg_map(kb_data)
    vqa_data = get_vqa_from_hf(task)
    # vqa_data = vqa_data[:10]
    model, tokenizer, clip_image_processor, transform = load_model_processor(args)   
    model.eval()

    list_res = []
    count = 0
    error_counter = 0
    try:
        for _, row in tqdm(vqa_data.iterrows(), total=len(vqa_data)):
            # print("_, row", _, row)
            res = {}
            try:
                image_file = url_jpg_map[row["image_url"]]
                query = (
                    row["multi_choice_prompt"]
                    if type == "mc"
                    else row["open_ended_prompt"]
                )
                # print("query", query)
                response = eval_instance(query, image_file, clip_image_processor, transform, tokenizer, model, args)

                res["qa_id"] = row["qa_id"]
                res["prediction"] = response
                res["lang"] = row["lang"]
                res["answer" if type == "oe" else "multi_choice_answer"] = row[
                    "answer" if type == "oe" else "multi_choice_answer"
                ]

                list_res.append(res)
            except Exception as e:
                error_message = f"Error at row {row['qa_id']}: {str(e)}"
                log_error(
                    error_message,
                    f"./result/error/task{task}_{type}_{MODEL_HANDLE[model_path]}_error.txt",
                )
                export_result(
                    list_res,
                    f"./result/error/task{task}_{type}_{MODEL_HANDLE[model_path]}_pred_cp{error_counter}.jsonl",
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
            f"./result/error/task{task}_{type}_{MODEL_HANDLE[model_path]}_error.txt",
        )
        export_result(
            list_res,
            f"./result/error/task{task}_{type}_{MODEL_HANDLE[model_path]}_pred_interrupt.jsonl",
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
    parser = argparse.ArgumentParser(description="FoodLISA chat")
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--model_path", default="Yueha0/FoodLMM-Chat", required=False)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=3000, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--load_error_types", action="store_true", default=False)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format.",
    )
    parser.add_argument(
        "--task", type=int, default=1, help="Task number to evaluate (1 or 2)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="oe",
        help="Type of question to evaluate (open ended [oe] or multiple choice [mc])",
    )

    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     default=True,
    #     help="Use float16 precision if supported",
    # )
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

    result = main(args.task, args.type, args.model_path, args)
    export_result(
        result,
        f"./result/task{args.task}_{args.type}_{MODEL_HANDLE[args.model_path]}_pred.jsonl",
    )
