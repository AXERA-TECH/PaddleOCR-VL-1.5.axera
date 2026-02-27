from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import onnxruntime as ort
import numpy as np
import os
from ml_dtypes import bfloat16
from utils.infer_func import InferManager
from utils.vision_output import describe_output_shapes, select_vit_output
import argparse
from PIL import Image


def _prepare_image(image_path, task):

    image = Image.open(image_path).convert("RGB")

    resize_h, resize_w = 576, 768
    image = image.resize((resize_w, resize_h))

    # AX vision model is compiled with fixed 576x768 token layout.
    # Keep spotting path aligned to avoid variable token counts.
    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    return image, max_pixels


def _run_vit_onnx(session, pixel_values, target_hidden_size, expected_tokens=None):
    outputs = session.run(None, {"pixel_values": pixel_values})
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _run_vit_axmodel(session, pixel_values, target_hidden_size, expected_tokens=None):
    outputs = session.run(None, {"pixel_values": pixel_values})
    return (
        select_vit_output(outputs, target_hidden_size, expected_tokens=expected_tokens),
        describe_output_shapes(outputs),
    )


def _expected_image_tokens(image_grid_thw, merge_size):
    merge_area = int(merge_size) * int(merge_size)
    return int(sum(int(t) * int(h) * int(w) // merge_area for t, h, w in image_grid_thw))


def _replace_image_tokens(token_ids, token_embeds, image_embeds, image_token_id):
    image_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == image_token_id]
    if not image_positions:
        return token_embeds

    flat_image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if len(image_positions) != flat_image_embeds.shape[0]:
        raise ValueError(
            f"Image tokens and image features do not match: tokens={len(image_positions)}, features={flat_image_embeds.shape[0]}"
        )
    if token_embeds.shape[-1] != flat_image_embeds.shape[-1]:
        raise ValueError(
            f"Embedding dim mismatch: token_dim={token_embeds.shape[-1]}, image_dim={flat_image_embeds.shape[-1]}"
        )
    token_embeds[image_positions, :] = flat_image_embeds
    return token_embeds


if __name__ == "__main__":

    """
    python3 infer_axmodel.py \
        --hf_model ./PaddleOCR-VL-1.5 \
        --axmodel_path ./PaddleOCR-VL-1.5_axmodel \
        --vit_model_path ./vit_models/vit_576x768.axmodel \
        --image_path ../assets/IMG_0462.JPG \
        --task ocr
    """
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.5 axmodel inference")
    parser.add_argument("--hf_model", type=str, default="./PaddleOCR-VL-1.5",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./PaddleOCR-VL-1.5_axmodel",
                        help="Path to compiled axmodel folder")
    parser.add_argument("--vit_model_path", type=str,
                        default="./vit_models/vit_576x768.axmodel",
                        help="Path to PaddleOCR-VL vision ONNX model or .axmodel")
    parser.add_argument("--image_path", type=str, default="../assets/IMG_0462.JPG",
                        help="Input image path")
    parser.add_argument("--task", type=str, default="ocr",
                        choices=["ocr", "table", "chart", "formula", "spotting", "seal"],
                        help="Task type")
    args = parser.parse_args()

    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path

    embeds = np.load(os.path.join(axmodel_path, "model.embed_tokens.weight.npy"))

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    image, max_pixels = _prepare_image(args.image_path, args.task)
    prompts = {
        "ocr": "OCR:",
        "table": "Table Recognition:",
        "formula": "Formula Recognition:",
        "chart": "Chart Recognition:",
        "spotting": "Spotting:",
        "seal": "Seal Recognition:",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompts[args.task]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={
            "size": {
                "shortest_edge": processor.image_processor.min_pixels,
                "longest_edge": max_pixels,
            }
        },
    )

    token_ids = inputs.input_ids[0].cpu().numpy().tolist()
    image_grid_thw = inputs.image_grid_thw.cpu().numpy().tolist()
    merge_size = config.vision_config.spatial_merge_size
    expected_tokens = _expected_image_tokens(image_grid_thw, merge_size)

    pixel_values = inputs.pixel_values
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.cpu().numpy().astype(np.float32)

    if args.vit_model_path.endswith(".axmodel"):
        try:
            from axengine import InferenceSession
        except Exception as exc:
            raise ImportError("axengine is required for .axmodel inference") from exc
        vit_session = InferenceSession(args.vit_model_path)
        image_embeds, vit_output_shapes = _run_vit_axmodel(
            vit_session,
            pixel_values,
            target_hidden_size=config.hidden_size,
            expected_tokens=expected_tokens,
        )
    else:
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        vit_session = ort.InferenceSession(args.vit_model_path, providers=providers)
        image_embeds, vit_output_shapes = _run_vit_onnx(
            vit_session,
            pixel_values,
            target_hidden_size=config.hidden_size,
            expected_tokens=expected_tokens,
        )

    if image_embeds.ndim == 3:
        image_embeds = image_embeds[0]
    image_seq_len = image_embeds.shape[0]
    if image_seq_len != expected_tokens:
        expected_features = int(sum(int(t) * int(h) * int(w) for t, h, w in image_grid_thw))
        if image_seq_len == expected_features:
            raise ValueError(
                "Vision output is pre-projector features. "
                f"got={image_seq_len}, expected_projected_tokens={expected_tokens}. "
                "Please re-export VIT ONNX with projector included (model_convert/export_onnx.py), "
                f"then re-compile to .axmodel. vit_output_shapes={vit_output_shapes}"
            )
        raise ValueError(
            "Unexpected image feature length. "
            f"got={image_seq_len}, expected_projected_tokens={expected_tokens}, "
            f"expected_pre_projector_features={expected_features}, vit_output_shapes={vit_output_shapes}"
        )
    projected_embeds = image_embeds

    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = _replace_image_tokens(
        token_ids,
        prefill_data,
        projected_embeds,
        image_token_id=config.image_token_id,
    )
    prefill_data = prefill_data.astype(bfloat16)

    eos_token_id = None
    if isinstance(config.eos_token_id, list) and len(config.eos_token_id) > 1:
        eos_token_id = config.eos_token_id

    slice_len = 128
    max_seq_len = 2048 - 1

    imer = InferManager(config, axmodel_path, max_seq_len=max_seq_len)
    token_ids = imer.prefill(tokenizer, token_ids, prefill_data, slice_len=slice_len)
    imer.decode(tokenizer, token_ids, embeds, slice_len=slice_len, eos_token_id=eos_token_id)
    print("\n")
