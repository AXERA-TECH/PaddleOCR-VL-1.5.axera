# PaddleOCR-VL-1.5 模型转换

本文档描述 `PaddleOCR-VL-1.5` 相关模型的导出与转换流程.

## 目录说明

```bash
model_convert/
├── export_onnx.py
├── prepare_calibration.py
├── pulsar2_configs/config.json
├── vit-models/
└── dataset/
```

## 环境准备

创建 conda 虚拟环境:

```sh
conda create -n paddle python=3.13 -y
conda activate paddle
```

然后安装下面一些依赖包:

```bash
# ensure the transformers v5 is installed
python -m pip install "transformers>=5.0.0"
pip install torch onnx onnxruntime onnxsim onnxslim loguru pillow numpy
```

## 1. 导出 Vision ONNX

在 `model_convert/` 目录执行:

```bash
python export_onnx.py -m ../python/PaddleOCR-VL-1.5 -o ./vit-models
```

默认输出类似:

- `vit-models/paddle_ocr_vl_vit_model_1x2268x3x14x14.onnx`

说明:

- 该导出脚本目前固定按 `576x768` 预处理路径对应的 token 排布导出.
- 导出的 Vision ONNX 包含 `visual + projector`，输出应为 merge 后的视觉 token（`42x54/4=567`），而不是 `2268` 个 pre-projector 特征.
- 导出脚本会在导出后自动做一次 ONNX 输出校验，若不包含 `567` token 输出会直接报错.
- 若修改了输入分辨率，请同步核对 `prepare_calibration.py` 和推理脚本中的预处理逻辑.

## 2. 生成校准数据

在 `model_convert/` 目录执行:

```bash
python prepare_calibration.py
```

脚本会遍历 `dataset/MSRA-TD500/` 图像，并生成 `*_pixel_values.npy`.

## 3. 打包校准集

将生成的 `calibration` 数据 (`*.npy` 文件) 打包为 `.tar` 文件, 放在 `dataset` 目录下.

当前 `pulsar2_configs/config.json` 默认读取:

- `dataset/paddle_ocr_vl_576x768_calibration.tar`

请注意命名.

## 4. 使用 pulsar2 编译

示例（AX650N）:

```sh
pulsar2 build \
  --output_dir ./compiled_output \
  --config pulsar2_configs/config.json \
  --npu_mode NPU3 \
  --input vit-models/paddle_ocr_vl_vit_model_1x2268x3x14x14.onnx \
  --compiler.check 0 \
  --target_hardware AX650
```

编译后的产物示例如下:

- `compiled_output/compiled.axmodel`

可拷贝到推理目录:

```bash
cp compiled_output/compiled.axmodel ../python/vit_models/vit_576x768.axmodel
```

## 5. 在推理脚本中使用 Vision axmodel

在 `python/` 目录执行:

```bash
python infer_axmodel.py \
  --hf_model ./PaddleOCR-VL-1.5 \
  --axmodel_path ./PaddleOCR-VL-1.5_axmodel \
  --vit_model_path ./vit_models/vit_576x768.axmodel \
  --image_path ../assets/IMG_0462.JPG \
  --task ocr
```

## 6. 大模型编译

> 要求 `pulsar2` 工具链的版本大于 `5.1-patch1`.

执行下面的命令:

```sh
pulsar2 llm_build --input_path PaddleOCR-VL-1.5 \
  --output_path PaddleOCR-VL-1.5_axmodel \
  --hidden_state_type bf16 \
  --prefill_len 128 \
  --kv_cache_len 2047 \
  --last_kv_cache_len 128 \
  --last_kv_cache_len 256 \
  --last_kv_cache_len 384 \
  --last_kv_cache_len 512 \
  --last_kv_cache_len 640 \
  --chip AX650 -c 0 \
  --parallel 18 # 根据实际的硬件环境调整并行度
```

当编译目标平台为 `AX650N` 时, 设置 `FLOAT_MATMUL_USE_CONV_EU=1` 环境变量可以大幅度提高模型 TTFT 时间 (AX620E 无效).
