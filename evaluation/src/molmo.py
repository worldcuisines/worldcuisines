import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_processor(model_path, fp32=False, multi_gpu=False):
    dtype = torch.float16 if not fp32 else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device if not multi_gpu else "auto",
    )
    return model, processor


def eval_instance(model, processor, image_file, query, tokenizer=None):
    inputs = processor.process(
        images=[image_file],
        text=query,
    )
    batch = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}
    batch["images"] = batch["images"].to(model.dtype)

    output = model.generate_from_batch(
        batch,
        GenerationConfig(
            max_new_tokens=50, stop_strings="<|endoftext|>", top_k=1, do_sample=True
        ),
        tokenizer=processor.tokenizer,
    )
    generated_tokens = output[0, batch["input_ids"].size(1) :]
    out = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return out
