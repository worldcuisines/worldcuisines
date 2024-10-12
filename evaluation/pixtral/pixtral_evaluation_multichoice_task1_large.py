from huggingface_hub import login

# Log in with your API key
login("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")

# Load Dataset
from datasets import load_dataset

# Replace 'dataset_name' with the name of the dataset, and 'subset_name' and 'split_name' accordingly
dataset_1_large = load_dataset('world-cuisines/vqa', name='task1', split='test_large')

import pandas as pd
dataset_1_large_df = pd.DataFrame(dataset_1_large)

# Just test running, see the output
from vllm import LLM
from vllm.sampling_params import SamplingParams

from tqdm import tqdm
tqdm.pandas()

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

def infer_with_error_handling_multichoice(row):
    try:
        response = llm.chat(
            [{
                "role": "user",
                "content": [{"type": "text", "text": row["multi_choice_prompt"]}, {"type": "image_url", "image_url": {"url": row["image_url"]}}]
            }],
            SamplingParams(max_tokens=2048)
        )
        return response[0].outputs[0].text
    except Exception as e:
        print(f"Error processing row: {row['multi_choice_prompt']} | Error: {str(e)}")
        return "-1"  # Return default value on error

dataset_1_large_df['pixtral_inferrence_1_multichoice'] = dataset_1_large_df.progress_apply(infer_with_error_handling_multichoice, axis=1)

dataset_1_large_df.to_csv(f"./inferred_task1_large_multichoice.csv", index=False)