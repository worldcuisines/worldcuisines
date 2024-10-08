import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, InstructBlipConfig, AutoModelForVision2Seq
import argparse
from accelerate import infer_auto_device_map, init_empty_weights
import ast

def image_parser(image_file):
    return image_file.split(',')

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
                }
        response = requests.get(image_file, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

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

def load_model_processor(model_path, fp_16=True, multi_gpu=False):
    if not multi_gpu:
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
        processor = InstructBlipProcessor.from_pretrained(model_path)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if fp_16:
            model.half()
        model.to(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model with the given configuration.
        config = InstructBlipConfig.from_pretrained(model_path)
        with init_empty_weights():
            model = AutoModelForVision2Seq.from_config(config)
            model.tie_weights()
        
        # Infer device map based on the available resources.
        device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB"},
                                   no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer',
                                                            'LlamaDecoderLayer'])
        device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model'
                                                                                               '.embed_tokens')]
        
        # Load the processor and model for image processing.
        offload = "offload"
        processor = InstructBlipProcessor.from_pretrained(model_path, device_map="auto")
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path,
                                                                     torch_dtype=torch.float16,
                                                                     device_map=device_map,
                                                                     offload_folder=offload, 
                                                                     offload_state_dict=True)

    return model, processor, device

def eval_model(model, processor, image_files, query, options, device='cuda'):
    image_files = image_parser(image_files)
    images = load_images(image_files)

    all_preds = []

    for image in images:
        log_lik_scores = []
        images_tensor = processor(images=image, return_tensors="pt").pixel_values.to(device).half()  # cast to fp16

        for option in options:
            target_prompt = query + ' ' + option

            inputs = processor(text=target_prompt, return_tensors="pt").to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Mandatory q-former for InstructBLIP
            qformer_inputs = processor(images=image, text=target_prompt, return_tensors="pt").to(device)
            qformer_input_ids = qformer_inputs.input_ids

            with torch.inference_mode(), torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=images_tensor,  # float-16
                    qformer_input_ids=qformer_input_ids,  # required
                )

            log_lik_scores.append((option, -outputs.loss.item()))

        pred_id = np.argmax(np.asarray([x[1] for x in log_lik_scores]))
        print(log_lik_scores)
        print('Prediction: {}'.format(log_lik_scores[pred_id]))
        all_preds.append(log_lik_scores[pred_id])

    return all_preds

def main():
    use_hardcoded = False

    if use_hardcoded:
        model_path = "Salesforce/instructblip-vicuna-13b"
        image_files = "https://eatwellabi.com/wp-content/uploads/2019/01/IMG_5172-500x375.jpg"
        query = "What is this dish name?"
        options = ['nasi goreng', 'nasi uduk', 'laksa', 'nasi kuning']
        fp16 = True
        multi_gpu = False
        model, processor, device = load_model_processor(model_path, fp_16=fp16, multi_gpu=multi_gpu)
    
        shared_prompt = 'This is an image of a '
        options = [shared_prompt + option for option in options]
    
        eval_model(
            model=model,
            processor=processor,
            image_files=image_files,
            query=query,
            options=options,
            device=device,
        )
    else:
        parser = argparse.ArgumentParser(description='InstructBLIP Evaluation')
        parser.add_argument('--model_path', type=str, default="Salesforce/instructblip-vicuna-13b",
                            help='Path to the pretrained model')
        parser.add_argument('--file_path', type=str, default='small_eval_task1.csv', help='Path to the vqa dataset')
        parser.add_argument('--start_index', type=int, default=0, help='Start index of vqa dataset prediction')
        parser.add_argument('--fp16', action='store_true', default=True, help='Use float16 precision if supported')
        parser.add_argument('--multi_gpu', action='store_true', default=False, help='Use multiple gpu if needed')
        args = parser.parse_args()

        model_path = args.model_path
        file_path = args.file_path
        start_index = args.start_index
        fp16 = args.fp16
        multi_gpu = args.multi_gpu

        vqa_df = pd.read_csv(file_path)
        prompts = [col for col in vqa_df.columns if col.startswith('prompt_')]
        prompts.remove('prompt_id')
        for prompt in prompts:
            if 'prediction_'+ prompt.split('prompt_')[1] not in vqa_df.columns:
                vqa_df['prediction_'+ prompt.split('prompt_')[1]] = None

        model, processor, device = load_model_processor(model_path, fp_16=fp16, multi_gpu=multi_gpu)

        for i in range (start_index, vqa_df.shape[0]):
            image_files = vqa_df['image_url'][i]
            for prompt in prompts:
                query = vqa_df[prompt][i]
                answer = 'answer_' + prompt.split('prompt_')[1]
                options = ast.literal_eval(vqa_df[answer][i])  
                # shared_prompt = 'This is an image of a '
                # options = [shared_prompt + option for option in options]
            
                predictions = eval_model(
                            model=model,
                            processor=processor,
                            image_files=image_files,
                            query=query,
                            options=options,
                            device=device,
                        )
                prediction = np.argmax(predictions)
                vqa_df.loc[i, 'prediction_'+ prompt.split('prompt_')[1]] = prediction
    
            vqa_df.to_csv('prediction_' + file_path, index=False)

if __name__ == '__main__':
    main()