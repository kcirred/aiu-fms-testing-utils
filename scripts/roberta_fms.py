from fms.models import get_model
import torch
from transformers import AutoTokenizer
from fms_mo.aiu_addons.fp8 import fp8_attn
from fms_mo.aiu_addons.fp8 import fp8_spyre_op


model = get_model(
    "hf_pretrained",
    model_path="/tmp/models/roberta-base_FP8",
    device_type="cpu",
    fused_weights=False,
)

print(model)

model.eval()
model.compile(backend="sendnn")

tokenizer = AutoTokenizer.from_pretrained("/tmp/models/roberta-base_FP8")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")

with torch.no_grad():
    model(
        inputs["input_ids"],
        mask=inputs["attention_mask"],
        attn_name="sdpa_bidirectional",
    )
