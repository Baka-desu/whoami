"""One-off: download the segmentation model and dump its class list so we can build the
3-class remap by hand. Not used by the running server."""
from transformers import SegformerForSemanticSegmentation

MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"

model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
for idx, name in sorted(model.config.id2label.items()):
    print(f"{idx}\t{name}")
