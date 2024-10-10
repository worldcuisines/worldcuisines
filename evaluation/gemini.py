import os
import requests
import argparse
import json
import jsonlines
from tqdm import tqdm
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


def load_history(history_file):
    """Load the saved history from the JSONL file."""
    history = {}
    if os.path.exists(history_file):
        with jsonlines.open(history_file, mode='r') as reader:
            for obj in reader:
                history[f"{obj['qa_id']}-{obj['lang']}"] = obj
    return history


def save_history(history_file, results):
    """Append the results to the history file."""
    with jsonlines.open(history_file, mode='a') as writer:
        writer.write_all(results)


def load_uploaded_files(uploaded_files_file):
    """Load the dictionary of uploaded files from a JSON file."""
    uploaded_files = {}
    if os.path.exists(uploaded_files_file):
        with open(uploaded_files_file, 'r') as f:
            uploaded_files = json.load(f)
    return uploaded_files


def save_uploaded_files(uploaded_files_file, uploaded_files):
    """Save the dictionary of uploaded files to a JSON file."""
    with open(uploaded_files_file, 'w') as f:
        json.dump(uploaded_files, f, indent=4)


def get_image(url, image_parent='.images'):
    """Check if the image exists; if not, download it."""
    os.makedirs(image_parent, exist_ok=True)

    # Extract filename from URL
    file_name = url.split('/')[-1].split('?')[0]
    file_path = os.path.join(image_parent, file_name)
    
    if os.path.exists(file_path):
        return file_path

    headers = {
        'User-Agent': os.environ.get("WIKIMEDIA_USER_AGENT")
    }

    print(f"Downloading image: {url}")
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Image saved: {file_path}.")
    
    return file_path
    

def main(task, q_type, data_split, model_name, save_every):
    model = genai.GenerativeModel(model_name)
    dataset = load_dataset("world-cuisines/vqa", f"task{task}", split=data_split)

    history_path = f"result/task{task}_{q_type}_{model_name}_pred.jsonl"
    history = load_history(history_path)

    error_history_path = f"result/task{task}_{q_type}_{model_name}_pred_errors.jsonl"

    uploaded_files_file = f"result/gemini_uploaded_files.json"
    uploaded_files = load_uploaded_files(uploaded_files_file)
    
    results = []
    skipped_results = []
    for item in tqdm(dataset, desc="Running inference"):
        if f"{item['qa_id']}-{item['lang']}" in history:
            continue

        downloaded_img = get_image(item['image_url'])

        # Skipping unsupported gif file (for now)
        if downloaded_img.lower().endswith('gif'):
            error_reason = f"Skipping question {item['qa_id']} due to unsupported image type (gif)."
            print(error_reason)
            skipped_results.append({
                'qa_id': item['qa_id'],
                'lang': item['lang'],
                'food_id': item['food_id'],
                'prompt_id': item['prompt_id'],
                'image_url': item['image_url'],
                'reason': error_reason
            })
            continue

        if downloaded_img not in uploaded_files:
            myfile = genai.upload_file(downloaded_img)
            uploaded_files[downloaded_img] = myfile.name
        else:
            myfile = genai.get_file(uploaded_files[downloaded_img])

        prompt = item['multi_choice_prompt'] if q_type == "mc" else item['open_ended_prompt']
        
        response = model.generate_content(
            [myfile, "\n", prompt],
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=50,
                top_k=1,
            ),
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]
        )

        cleaned_res = response.text.strip()
        if q_type == "mc" and len(cleaned_res) == 2 and cleaned_res[-1] == '.':
            cleaned_res = cleaned_res[:-1]

        res = {}
        res["qa_id"] = item["qa_id"]
        res["prediction"] = cleaned_res
        res["lang"] = item["lang"]
        res["answer" if type == "oe" else "multi_choice_answer"] = item[
            "answer" if type == "oe" else "multi_choice_answer"
        ]
        results.append(res)
        history[item['qa_id']] = res

        # Save every n API calls
        if len(results) >= save_every:
            save_history(history_path, results)
            save_history(error_history_path, skipped_results)
            save_uploaded_files(uploaded_files_file, uploaded_files)
            results = []
            skipped_results = []
    

if __name__ == "__main__":
    # Usage example: python gemini.py --task 1 --q_type mc --split test_large --model_name gemini-1.5-flash --save_every 10
    parser = argparse.ArgumentParser(description="Gemini Evaluation")
    parser.add_argument(
        "--task", type=int, default=1, help="Task number to evaluate (1 or 2)"
    )
    parser.add_argument(
        "--q_type",
        type=str,
        default="mc",
        help="Type of question to evaluate (open ended [oe] or multiple choice [mc])",
    )
    parser.add_argument(
        "--split", type=str, default="test_large", help="Dataset split",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-1.5-flash",
        help="API endpoint name to the model",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5, 
        help="Number of API calls after which to save the results to the history file",
    )
    args = parser.parse_args()

    main(args.task, args.q_type, args.split, args.model_name, args.save_every)
