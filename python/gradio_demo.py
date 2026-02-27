import argparse
import os
import socket
import time
from typing import Generator, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from ml_dtypes import bfloat16
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from axengine import InferenceSession
from utils.infer_func import InferManager

try:
    import onnxruntime as ort
except Exception:
    ort = None


TASK_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def _list_host_ips() -> List[str]:
    ips = set()
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, family=socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    if not ips:
        ips.add("127.0.0.1")
    return sorted(ips)


def _get_resample_filter():
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _prepare_image(image: Image.Image, task: str) -> Tuple[Image.Image, int]:
    image = image.convert("RGB")
    resize_h, resize_w = 576, 768
    image = image.resize((resize_w, resize_h))

    orig_w, orig_h = image.size
    spotting_upscale_threshold = 1500
    if task == "spotting" and orig_w < spotting_upscale_threshold and orig_h < spotting_upscale_threshold:
        image = image.resize((orig_w * 2, orig_h * 2), _get_resample_filter())

    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    return image, max_pixels


def _select_vit_output(outputs, target_hidden_size: int) -> np.ndarray:
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


def _run_vit_onnx(session, pixel_values: np.ndarray, target_hidden_size: int) -> np.ndarray:
    outputs = session.run(None, {"pixel_values": pixel_values})
    return _select_vit_output(outputs, target_hidden_size)


def _run_vit_axmodel(session, pixel_values: np.ndarray, target_hidden_size: int) -> np.ndarray:
    outputs = session.run(None, {"pixel_values": pixel_values})
    return _select_vit_output(outputs, target_hidden_size)


def _expected_image_features(image_grid_thw) -> int:
    return int(sum(int(t) * int(h) * int(w) for t, h, w in image_grid_thw))


def _expected_image_tokens(image_grid_thw, merge_size: int) -> int:
    merge_area = int(merge_size) * int(merge_size)
    return int(sum(int(t) * int(h) * int(w) // merge_area for t, h, w in image_grid_thw))


class _Projector(torch.nn.Module):
    def __init__(self, vision_hidden: int, text_hidden: int, merge_size: int):
        super().__init__()
        self.merge_size = int(merge_size)
        self.pre_norm = torch.nn.LayerNorm(vision_hidden, eps=1e-5)
        hidden_size = int(vision_hidden) * self.merge_size * self.merge_size
        self.linear_1 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.act = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(hidden_size, int(text_hidden), bias=True)

    def forward(self, image_features: torch.Tensor, image_grid_thw) -> torch.Tensor:
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


def _load_projector(hf_model_path: str, vision_hidden: int, text_hidden: int, merge_size: int):
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


def _replace_image_tokens(
    token_ids: List[int], token_embeds: np.ndarray, image_embeds: np.ndarray, image_token_id: int
) -> np.ndarray:
    image_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == image_token_id]
    if not image_positions:
        return token_embeds

    flat_image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
    if len(image_positions) != flat_image_embeds.shape[0]:
        raise ValueError(
            f"Image tokens and image features do not match: tokens={len(image_positions)}, "
            f"features={flat_image_embeds.shape[0]}"
        )
    if token_embeds.shape[-1] != flat_image_embeds.shape[-1]:
        raise ValueError(
            f"Embedding dim mismatch: token_dim={token_embeds.shape[-1]}, image_dim={flat_image_embeds.shape[-1]}"
        )
    token_embeds[image_positions, :] = flat_image_embeds
    return token_embeds


class PaddleOCRVLGradioDemo:
    def __init__(self, hf_model: str, axmodel_dir: str, vit_model: str, max_seq_len: int = 2047):
        self.hf_model = hf_model
        self.axmodel_dir = axmodel_dir
        self.vit_model = vit_model

        self.embeds = np.load(os.path.join(axmodel_dir, "model.embed_tokens.weight.npy"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.hf_model, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(self.hf_model, trust_remote_code=True)

        self.merge_size = self.config.vision_config.spatial_merge_size
        self.projector = _load_projector(
            self.hf_model,
            vision_hidden=self.config.vision_config.hidden_size,
            text_hidden=self.config.hidden_size,
            merge_size=self.merge_size,
        )

        if self.vit_model.endswith(".axmodel"):
            self.vit_session = InferenceSession(self.vit_model)
            self.vit_mode = "axmodel"
        else:
            if ort is None:
                raise ImportError("onnxruntime is required when --vit_model is an onnx file")
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.vit_session = ort.InferenceSession(self.vit_model, providers=providers)
            self.vit_mode = "onnx"

        self.infer_manager = InferManager(self.config, self.axmodel_dir, max_seq_len=max_seq_len)

    def _build_prompt_inputs(self, image: Image.Image, task: str, user_text: str):
        image, max_pixels = _prepare_image(image, task)
        prompt_text = user_text.strip() if user_text.strip() else TASK_PROMPTS[task]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        )
        return prompt_text, inputs

    def _prepare_model_inputs(self, inputs):
        token_ids = inputs.input_ids[0].cpu().numpy().tolist()
        image_grid_thw = inputs.image_grid_thw.cpu().numpy().tolist()
        expected_tokens = _expected_image_tokens(image_grid_thw, self.merge_size)
        expected_features = _expected_image_features(image_grid_thw)

        pixel_values = inputs.pixel_values
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.cpu().numpy().astype(np.float32)

        if self.vit_mode == "axmodel":
            image_embeds = _run_vit_axmodel(self.vit_session, pixel_values, target_hidden_size=self.config.hidden_size)
        else:
            image_embeds = _run_vit_onnx(self.vit_session, pixel_values, target_hidden_size=self.config.hidden_size)

        if image_embeds.ndim == 3:
            image_embeds = image_embeds[0]
        image_seq_len = image_embeds.shape[0]

        if image_seq_len == expected_tokens:
            projected_embeds = image_embeds
        elif image_seq_len == expected_features:
            if self.projector is None:
                raise ValueError(
                    "Projector weights are unavailable. Use a VIT model that already outputs merged features, "
                    "or provide projector weights in model.safetensors."
                )
            with torch.no_grad():
                projected = self.projector(
                    torch.from_numpy(image_embeds),
                    image_grid_thw,
                )
            projected_embeds = projected.cpu().numpy()
        else:
            raise ValueError(
                "Unexpected image feature length. "
                f"got={image_seq_len}, expected_tokens={expected_tokens}, expected_features={expected_features}"
            )

        prefill_data = np.take(self.embeds, token_ids, axis=0)
        prefill_data = _replace_image_tokens(
            token_ids,
            prefill_data,
            projected_embeds,
            image_token_id=self.config.image_token_id,
        )
        prefill_data = prefill_data.astype(bfloat16)
        return token_ids, prefill_data

    def _stream_generate(self, token_ids: List[int], prefill_data: np.ndarray, max_new_tokens: int = 1024):
        for k_cache in self.infer_manager.k_caches:
            k_cache.fill(0)
        for v_cache in self.infer_manager.v_caches:
            v_cache.fill(0)

        eos_token_id = None
        if isinstance(self.config.eos_token_id, list) and len(self.config.eos_token_id) > 1:
            eos_token_id = self.config.eos_token_id

        slice_len = 128
        t_start = time.time()
        token_ids = self.infer_manager.prefill(self.tokenizer, token_ids, prefill_data, slice_len=slice_len)

        mask = np.zeros((1, 1, self.infer_manager.max_seq_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.infer_manager.max_seq_len] -= 65536
        seq_len = len(token_ids) - 1
        if slice_len > 0:
            mask[:, :, :seq_len] = 0

        ttft_ms: Optional[float] = (time.time() - t_start) * 1000
        decode_tokens = 0
        decode_elapsed_ms: float = 0.0
        generated_text = self.tokenizer.decode(token_ids[seq_len:], skip_special_tokens=True)
        yield generated_text, ttft_ms, None, 1, False

        remaining_decode_budget = max(0, int(max_new_tokens) - 1)
        for step_idx in range(self.infer_manager.max_seq_len):
            if remaining_decode_budget <= 0:
                break
            if slice_len > 0 and step_idx < seq_len:
                continue

            cur_token = token_ids[step_idx]
            indices = np.array([step_idx], np.uint32).reshape((1, 1))
            data = self.embeds[cur_token, :].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)

            for layer_idx in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.infer_manager.k_caches[layer_idx],
                    "V_cache": self.infer_manager.v_caches[layer_idx],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.infer_manager.decoder_sessions[layer_idx].run(None, input_feed, shape_group=0)
                self.infer_manager.k_caches[layer_idx][:, step_idx, :] = outputs[0][:, :, :]
                self.infer_manager.v_caches[layer_idx][:, step_idx, :] = outputs[1][:, :, :]
                data = outputs[2]

            mask[..., step_idx] = 0
            if step_idx < seq_len - 1:
                continue

            post_out = self.infer_manager.post_process_session.run(None, {"input": data})[0]
            next_token, _, _ = self.infer_manager.post_process(post_out, temperature=0.7)

            if eos_token_id is not None and next_token in eos_token_id:
                break
            if next_token == self.tokenizer.eos_token_id:
                break

            token_ids.append(next_token)
            remaining_decode_budget -= 1

            generated_text = self.tokenizer.decode(token_ids[seq_len:], skip_special_tokens=True)
            decode_tokens += 1
            decode_elapsed_ms = (time.time() - t_start) * 1000 - ttft_ms
            avg_decode = decode_elapsed_ms / decode_tokens if decode_tokens > 0 else None
            total_tokens = 1 + decode_tokens
            yield generated_text, ttft_ms, avg_decode, total_tokens, False

        avg_decode = decode_elapsed_ms / decode_tokens if decode_tokens > 0 else None
        total_tokens = 1 + decode_tokens
        yield generated_text, ttft_ms, avg_decode, total_tokens, True

    def chat(self, user_input: str, image: Optional[Image.Image], task: str) -> Generator:
        if image is None:
            err = "请先上传图片，再执行识别。"
            metrics = (
                "<div style='text-align: right; font-size: 13px; color: #b91c1c; font-family: monospace;'>"
                "需要输入图像"
                "</div>"
            )
            yield [("输入", err)], gr.update(value=""), gr.update(), gr.update(value=metrics), gr.update(interactive=True)
            return

        yield (
            [("处理中", "模型准备中…")],
            gr.update(value=""),
            gr.update(),
            gr.update(
                value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                "TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>"
            ),
            gr.update(interactive=False),
        )

        try:
            prompt_text, inputs = self._build_prompt_inputs(image, task, user_input or "")
            token_ids, prefill_data = self._prepare_model_inputs(inputs)
        except Exception as exc:
            err = f"输入处理失败: {exc}"
            metrics = (
                "<div style='text-align: right; font-size: 13px; color: #b91c1c; font-family: monospace;'>"
                "预处理失败"
                "</div>"
            )
            yield [(prompt_text if "prompt_text" in locals() else "输入", err)], gr.update(value=""), gr.update(), gr.update(value=metrics), gr.update(interactive=True)
            return

        chatbot_history = [(prompt_text, "")]
        for partial, ttft_ms, avg_decode_ms, total_tokens, finished in self._stream_generate(
            token_ids, prefill_data, max_new_tokens=1024
        ):
            chatbot_history[-1] = (prompt_text, partial)
            ttft_disp = f"{ttft_ms:.0f}" if ttft_ms is not None else "--"
            decode_disp = f"{avg_decode_ms:.1f}" if avg_decode_ms is not None else "--"
            tok_disp = f"{total_tokens}" if total_tokens is not None else "--"
            metrics_text = (
                "<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                f"TTFT {ttft_disp} ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode {decode_disp} ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens {tok_disp}"
                "</div>"
            )
            if finished:
                yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_text), gr.update(interactive=True)
            else:
                yield chatbot_history, gr.update(value=""), gr.update(), gr.update(value=metrics_text), gr.update(interactive=False)

    @staticmethod
    def build_ui(demo: "PaddleOCRVLGradioDemo", server_name: str = "0.0.0.0", server_port: int = 7860, share: bool = False):
        custom_js = """
        function() {
            setTimeout(() => {
                const textareas = document.querySelectorAll('#user-input textarea');
                textareas.forEach(textarea => {
                    textarea.removeEventListener('keydown', textarea._customKeyHandler);
                    textarea._customKeyHandler = function(e) {
                        if (e.key === 'Enter') {
                            if (e.shiftKey) {
                                e.preventDefault();
                                const start = this.selectionStart;
                                const end = this.selectionEnd;
                                const value = this.value;
                                this.value = value.substring(0, start) + '\\n' + value.substring(end);
                                this.selectionStart = this.selectionEnd = start + 1;
                                this.dispatchEvent(new Event('input', { bubbles: true }));
                            } else {
                                e.preventDefault();
                                const sendBtn = document.querySelector('#send-btn');
                                if (sendBtn) {
                                    sendBtn.click();
                                }
                            }
                        }
                    };
                    textarea.addEventListener('keydown', textarea._customKeyHandler);
                });
            }, 500);
        }
        """

        with gr.Blocks(title="PaddleOCR-VL-1.5 AX Gradio Demo", theme=gr.themes.Soft(), js=custom_js) as iface:
            gr.HTML(
                """<style>
                #image-pane img {object-fit: contain; max-height: 380px;}
                #chat-wrap {position: relative;}
                #metrics-display {position: absolute; right: 12px; bottom: 12px; z-index: 5; pointer-events: none; text-align: right;}
                #metrics-display > div {display: inline-block;}
                </style>"""
            )
            gr.Markdown("### PaddleOCR-VL-1.5 图文识别演示\n上传图片，选择任务后执行识别。")

            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group(elem_id="chat-wrap"):
                        chatbot = gr.Chatbot(height=500, label="结果")
                        metrics_md = gr.Markdown(
                            "<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                            "TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>",
                            elem_id="metrics-display",
                        )

                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="可选：输入自定义提示词；留空将使用任务默认提示",
                            lines=2,
                            scale=7,
                            max_lines=5,
                            show_label=False,
                            elem_id="user-input",
                        )
                        with gr.Column(scale=1, min_width=100):
                            send_btn = gr.Button("发送", variant="primary", size="sm", elem_id="send-btn")
                            clear_btn = gr.Button("清空对话", variant="secondary", size="sm")

                with gr.Column(scale=3):
                    image_input = gr.Image(
                        type="pil",
                        label="上传图片",
                        height=380,
                        image_mode="RGB",
                        show_download_button=False,
                        elem_id="image-pane",
                    )
                    task_input = gr.Dropdown(
                        choices=["ocr", "table", "chart", "formula", "spotting", "seal"],
                        value="ocr",
                        label="任务类型",
                    )
                    gr.Markdown(
                        "- 支持单张图像推理\n"
                        "- 默认提示会根据任务自动设置\n"
                        "- 识别耗时受硬件和图像分辨率影响"
                    )

            def _clear():
                return (
                    [],
                    gr.update(value=""),
                    gr.update(),
                    gr.update(
                        value="<div style='text-align: right; font-size: 13px; color: #6b7280; font-family: monospace;'>"
                        "TTFT -- ms&nbsp;&nbsp;|&nbsp;&nbsp;Decode -- ms/token&nbsp;&nbsp;|&nbsp;&nbsp;Tokens --</div>"
                    ),
                    gr.update(interactive=True),
                )

            send_btn.click(
                fn=demo.chat,
                inputs=[user_input, image_input, task_input],
                outputs=[chatbot, user_input, image_input, metrics_md, send_btn],
                show_progress=False,
                queue=True,
            )
            clear_btn.click(fn=_clear, inputs=None, outputs=[chatbot, user_input, image_input, metrics_md, send_btn])

        target_port = server_port or 7860
        host_candidates: List[str] = []
        if server_name:
            host_candidates.append(server_name)
        host_candidates.extend(_list_host_ips())
        printed = set()
        print("可访问地址 (请任选其一):")
        for ip in host_candidates:
            if ip and ip not in printed:
                printed.add(ip)
                print(f"  http://{ip}:{target_port}")

        iface.queue().launch(server_name=server_name, server_port=server_port, share=share)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.5 AX gradio demo")
    parser.add_argument("--hf_model", type=str, default="./PaddleOCR-VL-1.5", help="HuggingFace 模型路径")
    parser.add_argument("--axmodel_path", type=str, default="./PaddleOCR-VL-1.5_axmodel", help="LLM axmodel 目录")
    parser.add_argument(
        "--vit_model",
        type=str,
        default="./vit_models/vit_576x768.axmodel",
        help="VIT 模型路径（支持 .axmodel 或 .onnx）",
    )
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio 监听地址")
    parser.add_argument("--share", action="store_true", help="启用 gradio share")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = PaddleOCRVLGradioDemo(args.hf_model, args.axmodel_path, args.vit_model)
    PaddleOCRVLGradioDemo.build_ui(demo, server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    """
    python3 gradio_demo.py \
        --hf_model ./PaddleOCR-VL-1.5 \
        --axmodel_path ./PaddleOCR-VL-1.5_axmodel \
        --vit_model ./vit_models/vit_576x768.axmodel
    """
    main()
