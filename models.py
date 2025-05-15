# models.py
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
# Paligemma Model Setup
from transformers import AutoProcessor as PaliProcessor, PaliGemmaForConditionalGeneration


paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224", torch_dtype=torch.bfloat16, device_map="cuda:0", revision="bfloat16"
).eval()
paligemma_processor = PaliProcessor.from_pretrained("google/paligemma-3b-pt-224")
paligemma_size = (224, 224)

#paligemma ft setup
paligemmaFT_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224", torch_dtype=torch.bfloat16, device_map="cuda:0", revision="bfloat16"
).eval()
paligemmaFT_processor = PaliProcessor.from_pretrained("google/paligemma-3b-pt-224")
MODEL_PATH = "paligemma-weights.pth"  # Tvoj súbor
paligemmaFT_model.load_state_dict(torch.load("paligemma-weights.pth"), strict=False)

#florence2 ft setup
florence2modelFT= AutoModelForCausalLM.from_pretrained("allmodel", trust_remote_code=True, torch_dtype='auto').eval().cuda()
florence2processorFT = AutoProcessor.from_pretrained("allmodel", trust_remote_code=True)

model= AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, torch_dtype='auto').eval().cuda()
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
