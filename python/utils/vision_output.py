from typing import List, Optional, Sequence, Tuple

import numpy as np


def seq_len_from_output(output: np.ndarray) -> Optional[int]:
    if output.ndim < 2:
        return None
    if output.ndim == 2:
        return int(output.shape[0])
    return int(output.shape[-2])


def normalize_vit_output(
    output: np.ndarray,
    target_hidden_size: int,
    expected_tokens: Optional[int] = None,
) -> np.ndarray:
    normalized = output
    if expected_tokens is not None:
        if normalized.ndim == 3 and normalized.shape[1] == target_hidden_size and normalized.shape[2] == expected_tokens:
            normalized = np.transpose(normalized, (0, 2, 1))
        elif normalized.ndim == 2 and normalized.shape[0] == target_hidden_size and normalized.shape[1] == expected_tokens:
            normalized = np.transpose(normalized, (1, 0))
    return normalized


def describe_output_shapes(outputs: Sequence[np.ndarray]) -> List[Tuple[int, ...]]:
    return [tuple(int(v) for v in output.shape) for output in outputs]


def select_vit_output(
    outputs: Sequence[np.ndarray],
    target_hidden_size: int,
    expected_tokens: Optional[int] = None,
) -> np.ndarray:
    normalized_outputs = [
        normalize_vit_output(output, target_hidden_size, expected_tokens=expected_tokens) for output in outputs
    ]

    image_embeds = None
    if expected_tokens is not None:
        for output in normalized_outputs:
            if output.ndim >= 2 and seq_len_from_output(output) == expected_tokens and output.shape[-1] == target_hidden_size:
                image_embeds = output
                break
        if image_embeds is None:
            for output in normalized_outputs:
                if output.ndim >= 2 and seq_len_from_output(output) == expected_tokens:
                    image_embeds = output
                    break

    if image_embeds is None:
        for output in normalized_outputs:
            if output.ndim >= 2 and output.shape[-1] == target_hidden_size:
                image_embeds = output
                break

    if image_embeds is None:
        image_embeds = normalized_outputs[0]
    if image_embeds.ndim == 2:
        image_embeds = image_embeds[None, ...]
    return image_embeds
