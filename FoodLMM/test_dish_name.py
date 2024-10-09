import tqdm
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
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from utils import conversation as conversation_lib
from model.LISA import LISAForCausalLM
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.config import Config
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


#把图片地址字符串用逗号分成图片地址列表
def image_parser(image_file):
    return image_file.split(',')

#把网络图片下载并打开，本地图片直接打开
def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

#打开多张图片
def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_log_prob(model, tokenizer, prefix, targets):
    log_sums = []
    for target in targets:
        input_tokens = tokenizer.encode(prefix, add_special_tokens=False, return_tensors='pt')
        output_tokens = tokenizer.encode(target, add_special_tokens=False, return_tensors='pt')

        tokens = torch.cat([input_tokens, output_tokens], dim=1)
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits

        log_sum = 0
        range_index = range(input_tokens.shape[1] - 1, tokens.shape[1] - 1)
        len_range = tokens.shape[1] - 1 - (input_tokens.shape[1] - 1) 
        for i in range_index:
            past_tok, current_tok = i, i + 1
            token_logit = logits[0, past_tok, :]
            token_log_probs = torch.nn.functional.log_softmax(token_logit, dim=-1)
            log_token_prob = token_log_probs[tokens[0, current_tok]].item()
            log_sum += log_token_prob

        log_sums.append(log_sum / len_range)

    normalized_scores = normalize(log_sums)
    pred = targets[np.argmax(normalized_scores)]
    return pred, normalized_scores


def parse_args(args):
    parser = argparse.ArgumentParser(description="FoodLISA chat")
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--version", default="/home/guiyinxuan/FoodLMM-ckpt", required=False)
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
    return parser.parse_args(args)


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



def get_text_output(
    prompt,
    image,
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

    image_np = np.array(image)
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
    return text_output.replace("\n", "").replace("  ", " ")


def load_model(args):

    configs = Config(args)
    # configs.pretty_print_system()
    args = configs.args
    args = argparse.Namespace(**args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
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
        args.version, low_cpu_mem_usage=True, **kwargs
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

def eval_model(prompt, image_files, clip_image_processor, transform, tokenizer, model, args,):
    image_files = image_parser(image_files)
    images = load_images(image_files)
    
    for image in images:
        text_output = get_text_output(prompt,image,clip_image_processor,transform,tokenizer,model,args)

        # print(text_output)
        out = text_output.split('ASSISTANT: ')[1].replace('.', '').replace('</s>', '').replace('It is', '').split(", ")
        print(out[0])


def main():
    parser = argparse.ArgumentParser(description="FoodLISA chat")
    parser.add_argument("--cfg_file", required=False, help="path to configuration file.")
    parser.add_argument("--version", default="/home/guiyinxuan/FoodLMM-ckpt", required=False)
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

    use_hardcoded = True
    if use_hardcoded:
        image_files = "pics/873px-Cotoletta_alla_milanese_in_milano.jpg"
        # image_files = "https://eatwellabi.com/wp-content/uploads/2019/01/IMG_5172-500x375.jpg"
        prompt = "What is this dish name? Print only the answer."
#         prompt = """
# Gimana cara orang di Italia ngenalin makanan ini?
# 1. Karađorđeva šnicla
# 2. Schnitzel
# 3. Ayam Kyiv
# 4. Goetta
# 5. Cotoletta alla milanese
# Print only the answer with a single answer id (1,2,3,4,5).
#         """
    else:
        parser.add_argument('--version', default="/home/guiyinxuan/FoodLMM-ckpt", required=False)
        parser.add_argument('--image_files', type=str, required=True,
                            help='Comma-separated list of image URLs or file paths')
        parser.add_argument('--prompt', type=str, required=True,
                            help='prompt to be evaluated with the images')        



    args = parse_args(sys.argv[1:])
    model, tokenizer, clip_image_processor, transform = load_model(args)   
    model.eval()

    eval_model(prompt, image_files, clip_image_processor, transform, tokenizer, model, args)




if __name__ == "__main__":
    main()
