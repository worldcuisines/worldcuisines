from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
import torch


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if not fp32 else torch.float32,
        trust_remote_code=True,
        device_map="auto" if multi_gpu else "cuda:0",
    )

    return model, processor


def eval_instance(model, processor, image_file, query):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "text", "text": query},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image_file, text=prompt, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.cuda.amp.autocast(
        dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else None
    ):
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            stop_strings=["<|im_end|>"],
            tokenizer=processor.tokenizer,
            do_sample=True,
            top_k=1,
        )
        out = processor.decode(
            output[0][inputs["input_ids"].size(1) :], skip_special_tokens=True
        )

    return out
