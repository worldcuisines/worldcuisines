# TODO: Refactor to match other models way of running
# TODO: add dependencies to req.txt (if any)

from datasets import load_dataset
import json

def get_vqa_from_hf(task):
    if task == 1:
        vqa = load_dataset("world-cuisines/vqa", name="task1")
    elif task == 2:
        vqa = load_dataset("world-cuisines/vqa", name="task2")
    else:
        raise ValueError(
            "Invalid task number. Please choose from 1 (dish name) or 2 (region)"
        )
    print(vqa)
    vqa_data = vqa["test_large"]
    return vqa_data

for task in range(1, 3):
    data = get_vqa_from_hf(task)
    count = 0
    with open(f"task_multi_choice_{task}.jsonl", "w+") as f_out:
        for example in data:
            qa_id = example["qa_id"]
            image_url = example["image_url"]
            multi_choice_prompt = example["multi_choice_prompt"]
            open_ended_prompt = example["open_ended_prompt"]
            obj = {
                "custom_id": f"request-{qa_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "max_completion_tokens": 50,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{multi_choice_prompt}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"{image_url}"
                                    }
                                }
                            ]
                        }
                    ],
                    "seed": 42,
                    "temperature": 0
                }
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    count = 0
    with open(f"task_open_ended_{task}.jsonl", "w+") as f_out:
        for example in data:
            obj = {
                "custom_id": f"request-{qa_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "max_completion_tokens": 50,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{open_ended_prompt}"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"{image_url}"
                                    }
                                }
                            ]
                        }
                    ],
                    "seed": 42,
                    "temperature": 0
                }
            }
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
