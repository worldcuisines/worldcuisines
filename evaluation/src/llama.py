from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os
load_dotenv()

# Log in with your API key
login(os.environ.get("HUGGINGFACE_API_KEY"))

def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path, 
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if not fp32 else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if multi_gpu else "cuda:0",
    )
    model.tie_weights()

    return model, processor


def eval_instance(model, processor, image_file, query):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        top_k=1,
    )
    out = processor.decode(
        output[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
    )

    return out
