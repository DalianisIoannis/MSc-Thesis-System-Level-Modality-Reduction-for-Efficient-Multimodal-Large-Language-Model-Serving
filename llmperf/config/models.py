import os

from dataclasses import dataclass
from typing import LiteralString, Optional, Union

from llmperf.constants import MODELS_DIR

@dataclass
class Model:
    name: str
    path: Union[str,LiteralString]
    max_model_len: int
    alias: str
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    audio_token_index: Optional[int] = None

    def __hash__(self):
        return hash((self.name, self.alias))

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.name == other.name and self.alias == other.alias
        return False

MODELS = {
    Model(
        name="Mistral-7b",
        path=os.path.join(MODELS_DIR, "Mistral-7B-Instruct-v0.2"),
        max_model_len=32768,
        alias="text-mistral"
    ),
    Model(
        name="LLaVA 1.6 (Mistral-7b)",
        path="llava-hf/llava-v1.6-mistral-7b-hf",
        max_model_len=32768,
        alias="image-mistral",
        image_token_index=32000
    ),
    Model(
        name="LLaVA-Next-Video (Mistral-7b)",
        path="llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
        max_model_len=32768,
        alias="video-mistral",
        image_token_index=32001,
        video_token_index=32000
    ),
    Model(
        name="Qwen2-Audio-7b",
        path="Qwen/Qwen2-Audio-7B-Instruct",
        max_model_len=8192,
        alias="audio-qwen",
        audio_token_index=151646
    ),
    Model(
        name="Qwen2-VL-2B-Instruct",
        # path="Qwen/Qwen2-VL-2B-Instruct",
        path="/srv/muse-lab/datasets/VLMEvalKitdata/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c",
        max_model_len=32768,
        alias="qwen-2b-instruct",
        # the index of modal tokens which I count in order to get number of modal outputs from the model
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        name="Qwen2-VL-7B-Instruct",
        path="/srv/muse-lab/datasets/VLMEvalKitdata/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac",
        max_model_len=32768,
        alias="qwen-7b-instruct",
        image_token_index=151655,
        video_token_index=151656
    ),
    Model(
        # different path and version by the one used in VLMEvalKit
        name="llava_onevision_qwen2_7b_ov",
        path="/srv/muse-lab/models/llava-onevision-qwen2-7b-ov-chat-hf",
        max_model_len=32768,
        alias="llava-ov-qwen2-7b",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        # different path and version by the one used in VLMEvalKit
        name="llava_onevision_qwen2_0.5b_ov",
        path="/srv/muse-lab/models/llava-onevision-qwen2-0.5b-ov-hf",
        max_model_len=32768,
        alias="llava-ov-qwen2-0.5b",
        image_token_index=151646,
        video_token_index=151647
    ),
    Model(
        name="pixtral_12b",
        path="/srv/muse-lab/models/pixtral-12b",
        # max_model_len=38432,
        max_model_len=45056,
        # max-num-batched-tokens=45056, 
        # num-gpu-blocks-override": 2816,
        # multi_image=True
        alias="pixtral_12b",
        image_token_index=10,
        video_token_index=10
    )
}

def get_model_by_name(name: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "name", None) == name:
            return model
    return None

def get_model_by_alias(alias: str) -> Union[None, Model]:
    for model in MODELS:
        if getattr(model, "alias", None) == alias:
            return model
    return None