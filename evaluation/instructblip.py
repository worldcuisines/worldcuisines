# Clone LAVIS Repository
# Change the requirement for open3d 0.13.0 => 0.18.0, then install the requirement
# Change llm_model: "lmsys/vicuna-13b-v1.1" in lavis/configs/models/blip2/blip2_instruct_vicuna_13b.yaml

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model
import sys

def get_instructblip_model(device='cuda', dtype=torch.bfloat16, use_multi_gpus=True):
    model, vis_processors, txt_processors = load_model_and_preprocess(
                    name='blip2_vicuna_instruct',
                    model_type='vicuna13b',
                    is_eval=True,
                )
    model.to(dtype)
    if use_multi_gpus:
        # Trying to use 2x24 GB GPU, but still not working
        device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=['LlamaDecoderLayer', 'VisionTransformer'])
        device_map['llm_model.lm_head'] = device_map['llm_proj'] = device_map['llm_model.model.embed_tokens']
        print(device_map)
        model = dispatch_model(model, device_map=device_map)
        torch.cuda.empty_cache()
    else:
        # Should be working, but don't have the GPU :)
        model.to('cuda:0')
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    return model, (txt_processors, vis_processors)

# Haven't tried, found this from github issue. Maybe focus on this since it's already available in hf.
# The above code which uses cloning LAVIS can be adapted for X-InstructBLIP
# def get_instructblip_model_hf():
#     # Load the model configuration.
#     config = InstructBlipConfig.from_pretrained("Salesforce/instructblip-vicuna-13b")

#     # Initialize the model with the given configuration.
#     with init_empty_weights():
#         model = AutoModelForVision2Seq.from_config(config)
#         model.tie_weights()

#     # Infer device map based on the available resources.
#     device_map = infer_auto_device_map(model, max_memory={0: "20GiB", 1: "20GiB"}, no_split_module_classes=['InstructBlipEncoderLayer', 'InstructBlipQFormerLayer','LlamaDecoderLayer'])
#     device_map['language_model.lm_head'] = device_map['language_projection'] = device_map[('language_model.model','.embed_tokens')]

#     offload = ""
#     # Load the processor and model for image processing.
#     processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto")
#     model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map=device_map, offload_folder=offload, offload_state_dict=True)
#
#     return model

# load sample image
raw_image = Image.open("./docs/_static/Confusing-Pictures.jpg").convert("RGB")
print('Image Loaded...')

# loads InstructBLIP model
model, vis_processors, _ = get_instructblip_model()

# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to('cuda')
output = model.generate({"image": image, "prompt": "What is unusual about this image?"})
print(output)