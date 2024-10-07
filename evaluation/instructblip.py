import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
from sklearn.preprocessing import normalize
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import argparse

def image_parser(image_file):
    return image_file.split(',')

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
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

def load_model_processor(model_path, fp_16=True):
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path)
    processor = InstructBlipProcessor.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp_16:
        model.half()
    model.to(device)

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
    #------ ARGPARSE VERSION------#
    parser = argparse.ArgumentParser(description='InstructBLIP Evaluation')
    parser.add_argument('--model_path', type=str, default="Salesforce/instructblip-vicuna-13b",
                        help='Path to the pretrained model')
    parser.add_argument('--image_files', type=str, required=True,
                        help='Comma-separated list of image URLs or file paths')
    parser.add_argument('--query', type=str, required=True,
                        help='Query to be evaluated with the images')
    parser.add_argument('--options', type=str, required=True,
                        help='Comma-separated list of options for evaluation')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use float16 precision if supported')
    
    args = parser.parse_args()


    #------ HARD CODED VERSION------#
    # model_path = "Salesforce/instructblip-vicuna-13b"
    # model,processor, device = load_model_processor(model_path, fp_16=True)

    # prompt = "What is this dish name?"
    # image_file = "https://eatwellabi.com/wp-content/uploads/2019/01/IMG_5172-500x375.jpg"

    # shared_prompt = 'This is an image of a '
    # options = [shared_prompt+x for x in ['nasi goreng', 'nasi uduk', 'laksa', 'nasi kuning']]
    # args.image_files = image_file
    # args.query = prompt
    # args.options = ','.join(options)
    # args.fp16 = True

    # Load model and processor
    model, processor, device = load_model_processor(args.model_path, fp_16=args.fp16)

    # Prepare options
    shared_prompt = 'This is an image of a '
    options = [shared_prompt + option for option in args.options.split(',')]

    # Evaluate the model
    eval_model(
        model=model,
        processor=processor,
        image_files=args.image_files,
        query=args.query,
        options=options,
        device=device,
    )

if __name__ == '__main__':
    main()
