import torch
from transformers import AutoModelForImageTextToText

import argparse
from loguru import logger
import os
import onnx
import onnxruntime
from onnx.shape_inference import infer_shapes
from onnxsim import simplify
import numpy as np
import torch.nn as nn


def onnx_sim(onnx_path):
    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    logger.info(f"onnx simpilfy successed, and model saved in {onnx_path}")


def _seq_len_from_output(output: np.ndarray):
    if output.ndim < 2:
        return None
    if output.ndim == 2:
        return int(output.shape[0])
    return int(output.shape[-2])


def verify_vit_onnx_output(onnx_path: str, sample_pixel_values: np.ndarray, expected_tokens: int):
    providers = ["CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    if "CPUExecutionProvider" not in available:
        providers = available
    sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
    outputs = sess.run(None, {"pixel_values": sample_pixel_values})
    output_shapes = [tuple(int(v) for v in out.shape) for out in outputs]
    has_expected = any(_seq_len_from_output(out) == int(expected_tokens) for out in outputs if out.ndim >= 2)
    if not has_expected:
        raise RuntimeError(
            f"Exported ONNX does not contain projected tokens output: "
            f"expected_tokens={expected_tokens}, output_shapes={output_shapes}"
        )
    logger.info(f"onnx output verify success, output_shapes={output_shapes}")


class VisionModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.vision_model = model.model.visual
        self.projector = model.model.projector

    def forward(self, pixel_values):
        # Prepare the input tensors for the vision model
        # 768x1024 输入图像的尺寸
        # cu_seqlens = torch.tensor([0, 3996], device='cuda:0', dtype=torch.int32)
        # image_grid_thw = torch.tensor([[1, 54, 74]], device='cuda:0')

        # 576x768 输入图像的尺寸
        device = pixel_values.device
        cu_seqlens = torch.tensor([0, 2268], device=device, dtype=torch.int32)
        image_grid_thw = torch.tensor([[1, 42, 54]], device=device, dtype=torch.int32)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            cu_seqlens=cu_seqlens,
            return_dict=True,
        )
        image_embeds = vision_outputs.last_hidden_state
        image_embeds = self.projector(image_embeds, image_grid_thw)
        return image_embeds


if __name__ == '__main__':

    """
    Usage:
        python export_onnx.py -m ../python/PaddleOCR-VL-1.5 -o ./vit-models
    """
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-m", "--model", type=str, help="hugging fance model path")
    parser.add_argument("--name", type=str, default=None, help="onnx name")
    parser.add_argument(
        "--imgsize",
        type=int,
        default=384,
        help="onnx input image size (reserved, current export path is fixed to 576x768)",
    )
    parser.add_argument("-o", "--onnx_save_dir", type=str, default='./vit-models', help="vit onnx model save path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = args.model
    onnx_save_dir = args.onnx_save_dir

    if not os.path.exists(onnx_save_dir):
        os.makedirs(onnx_save_dir)

    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()

    # Force eager attention to avoid SDPA/GQA export path not supported by ONNX symbolic
    if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "_attn_implementation"):
        model.config.vision_config._attn_implementation = "eager"
    if hasattr(model, "model") and hasattr(model.model, "config") and hasattr(model.model.config, "_attn_implementation"):
        model.model.config._attn_implementation = "eager"

    patch_size = 14
    # 对应 768x1024 输入图像的尺寸
    # B, N, C, H, W = 1, 3996, 3, patch_size, patch_size
    # 对应 576x768 输入图像的尺寸, ceil(H / patch_size) = 42, floor(W / patch_size) = 54
    B, N, C, H, W = 1, 2268, 3, patch_size, patch_size
    pixel_values = torch.randn(B, N, C, H, W).to(device=device, dtype=torch.float32)
    paddle_ocr_vl_vit_onnx_save_dir = os.path.join(
        onnx_save_dir,
        f'paddle_ocr_vl_vit_model_{B}x{N}x{C}x{H}x{W}.onnx' if args.name is None else args.name
    )
    vision_model_wrapper = VisionModelWrapper(model).to(device=device, dtype=torch.float32)

    torch.onnx.export(
        vision_model_wrapper,
        pixel_values,
        paddle_ocr_vl_vit_onnx_save_dir,
        opset_version=17, # 14
        do_constant_folding=True,
        verbose=False,
        input_names=["pixel_values"],
        output_names=["feature"],
    )
    verify_vit_onnx_output(
        paddle_ocr_vl_vit_onnx_save_dir,
        pixel_values.cpu().numpy().astype(np.float32),
        expected_tokens=(42 * 54) // 4,
    )

    logger.info("export paddle_ocr_vl_vit_model onnx succee!")
    onnx_sim(paddle_ocr_vl_vit_onnx_save_dir)
    logger.debug("Use onnxslim to fine-tune the ONNX model. Please ensure onnxslim is installed first.")
    import subprocess
    result = subprocess.run(
        ["onnxslim", paddle_ocr_vl_vit_onnx_save_dir, paddle_ocr_vl_vit_onnx_save_dir],
        capture_output=True,
        text=True
    )
    logger.debug(result.stdout)
