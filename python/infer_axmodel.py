from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import onnxruntime as ort
import numpy as np
import os
from ml_dtypes import bfloat16
from utils.infer_func import InferManager
import argparse
from PIL import Image
import torch


def _get_resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _prepare_image(image_path, task):

    image = Image.open(image_path).convert("RGB")

    resize_h, resize_w = 576, 768
    image = image.resize((resize_w, resize_h))

    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500
    if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
        image = image.resize((orig_w * 2, orig_h * 2), _get_resample_filter())

    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    return image, max_pixels


def _select_vit_output(outputs, target_hidden_size):
    image_embeds = None
    for output in outputs:
        if output.ndim >= 2 and output.shape[-1] == target_hidden_size:
            image_embeds = output
            break
    if image_embeds is None:
        image_embeds = outputs[0]
    if image_embeds.ndim == 2:
        image_embeds = image_embeds[None, ...]
    return image_embeds


def _run_vit_onnx(session, pixel_values, target_hidden_size):
    outputs = session.run(None, {"pixel_values": pixel_values})
    return _select_vit_output(outputs, target_hidden_size)


def _run_vit_axmodel(session, pixel_values, target_hidden_size):
    outputs = session.run(None, {"pixel_values": pixel_values})
    return _select_vit_output(outputs, target_hidden_size)


def _expected_image_features(image_grid_thw):
    return int(sum(int(t) * int(h) * int(w) for t, h, w in image_grid_thw))


def _expected_image_tokens(image_grid_thw, merge_size):
    merge_area = int(merge_size) * int(merge_size)
    return int(sum(int(t) * int(h) * int(w) // merge_area for t, h, w in image_grid_thw))


class _Projector(torch.nn.Module):
    def __init__(self, vision_hidden, text_hidden, merge_size):
        super().__init__()
        self.merge_size = int(merge_size)
        self.pre_norm = torch.nn.LayerNorm(vision_hidden, eps=1e-5)
        hidden_size = int(vision_hidden) * self.merge_size * self.merge_size
        self.linear_1 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(hidden_size, int(text_hidden), bias=True)

    def forward(self, image_features, image_grid_thw):
        chunk_sizes = [int(t) * int(h) * int(w) for t, h, w in image_grid_thw]
        image_features_chunks = torch.split(image_features, chunk_sizes, dim=0)

        processed = []
        for image_feature, grid in zip(image_features_chunks, image_grid_thw):
            t, h, w = [int(v) for v in grid]
            image_feature = self.pre_norm(image_feature)
            d = image_feature.shape[-1]
            h_block = h // self.merge_size
            w_block = w // self.merge_size

            image_feature = image_feature.reshape(t, h_block, self.merge_size, w_block, self.merge_size, d)
            image_feature = image_feature.transpose(2, 3)
            image_feature = image_feature.reshape(t * h_block * w_block, self.merge_size * self.merge_size * d)

            hidden_states = self.linear_1(image_feature)
            hidden_states = self.act(hidden_states)
            hidden_states = self.linear_2(hidden_states)
            processed.append(hidden_states)

        return torch.cat(processed, dim=0)


def _load_projector(hf_model_path, vision_hidden, text_hidden, merge_size):
    try:
        from safetensors.torch import load_file
    except Exception:
        return None

    weight_path = os.path.join(hf_model_path, "model.safetensors")
    if not os.path.exists(weight_path):
        return None

    state = load_file(weight_path, device="cpu")
    prefixes = ["model.projector.", "projector.", "model.mlp_AR.", "mlp_AR."]
    proj_state = {}
    for prefix in prefixes:
        proj_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if proj_state:
            break
    if not proj_state:
        return None

    projector = _Projector(vision_hidden, text_hidden, merge_size)
    missing, unexpected = projector.load_state_dict(proj_state, strict=False)
    if missing or unexpected:
        return None
    projector.eval()
    return projector


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
    expected_features = _expected_image_features(image_grid_thw)

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
        image_embeds = _run_vit_axmodel(vit_session, pixel_values, target_hidden_size=config.hidden_size)
    else:
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        vit_session = ort.InferenceSession(args.vit_model_path, providers=providers)
        image_embeds = _run_vit_onnx(vit_session, pixel_values, target_hidden_size=config.hidden_size)

    if image_embeds.ndim == 3:
        image_embeds = image_embeds[0]
    image_seq_len = image_embeds.shape[0]
    if image_seq_len == expected_tokens:
        projected_embeds = image_embeds
    elif image_seq_len == expected_features:
        projector = _load_projector(
            hf_model_path,
            vision_hidden=config.vision_config.hidden_size,
            text_hidden=config.hidden_size,
            merge_size=merge_size,
        )
        if projector is None:
            raise ValueError(
                "Projector weights are unavailable. Use a vit axmodel/onnx that already outputs merged features "
                f"(expected length {expected_tokens}), or provide projector weights in {hf_model_path}."
            )
        with torch.no_grad():
            projected = projector(
                torch.from_numpy(image_embeds),
                image_grid_thw,
            )
        projected_embeds = projected.cpu().numpy()
    else:
        raise ValueError(
            "Unexpected image feature length. "
            f"got={image_seq_len}, expected_tokens={expected_tokens}, expected_features={expected_features}"
        )

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