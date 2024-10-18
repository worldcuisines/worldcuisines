from transformers import AutoModelForCausalLM, AutoProcessor
import torch


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, num_crops=16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if multi_gpu else "cuda:0",
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if not fp32 else torch.float32),
    )

    return model, processor


def eval_instance(model, processor, image_file, query):
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    images = [image_file]

    messages = [
        {"role": "user", "content": f"<|image_{1}|>\n" + query},
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
    }

    generate_ids = model.generate(
        **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
    )

    # remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response
