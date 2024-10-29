from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = LlavaNextProcessor.from_pretrained(model_path)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=(torch.float16 if not fp32 else torch.float32),
        low_cpu_mem_usage=True,
        device_map="auto" if multi_gpu else "cuda:0",
    )
    processor.patch_size = 14
    processor.vision_feature_select_strategy = "default"
    processor.tokenizer.padding_side = "left"
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return model, processor


def eval_instance(model, processor, image_file, query):
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
        max_new_tokens=512, 
        do_sample=True,
        temperature=0.2,
    )

    out_with_template = processor.decode(output[0], skip_special_tokens=True)
    out = out_with_template[
        out_with_template.index("ASSISTANT: ") + len("ASSISTANT: ") :
    ]

    return out
