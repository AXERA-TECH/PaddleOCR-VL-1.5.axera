from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- Settings ----
model_path = "./PaddleOCR-VL-1.5"
image_path = "../assets/paddleocr_vl_demo.png"
# image_path = "../assets/IMG_0462.JPG"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula' | 'spotting' | 'seal'
# ------------------

# ---- Image Preprocessing For Spotting ----
image = Image.open(image_path).convert("RGB")

# 导出 ONNX 需要固定图像尺寸
# resize_h, resize_w = 768, 1024 # 1368, 1524
resize_h, resize_w = 576, 768 # 1368, 1524, 最小尺寸为 336x336
image = image.resize((resize_w, resize_h))

orig_w, orig_h = image.size
spotting_upscale_threshold = 1500

if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
    process_w, process_h = orig_w * 2, orig_h * 2
    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS
    image = image.resize((process_w, process_h), resample_filter)

# Set max_pixels: use 1605632 for spotting, otherwise use default ~1M pixels
max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28

# ---------------------------

# -------- Inference --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}

model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels, "longest_edge": max_pixels}},
).to(model.device)

print("inputs.pixel_values.shape:", inputs.pixel_values.shape)
print("inputs.input_ids.shape:", inputs.input_ids.shape)
print("inputs.image_grid_thw:", inputs.image_grid_thw)
# import pdb; pdb.set_trace()

outputs = model.generate(**inputs, max_new_tokens=1024)
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)
# ---------------------------
