from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=(torch.float16 if not fp32 else torch.float32),
        low_cpu_mem_usage=True,
        device_map="auto" if multi_gpu else "cuda:0",
    )

    return model, processor


def eval_instance(model, processor, image_file, query, tokenizer=None):
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

    inputs = processor(images=image_file, text=prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **inputs,
        max_new_tokens=50, 
        do_sample=False
    )

    out_with_template = processor.decode(output[0], skip_special_tokens=True)
    out = out_with_template[
        out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
    ]

    return out
