import json
import os

from dataclasses import dataclass, field
from typing import Callable, Dict, List, LiteralString, Optional, Tuple, Union

from llmperf.constants import DATASETS_DIR
from llmperf.preprocessing.input import get_clotho_input, get_llava_image_reasoning_input, get_llava_video_description_input, get_sharegpt_input, get_aokvqa_input
from llmperf.preprocessing.output import get_clotho_output, get_llava_image_reasoning_output, get_llava_video_description_output, get_sharegpt_output, get_aokvqa_output, get_llava_video_multi_choice_output
from llmperf.preprocessing.modality_path import get_clotho_audio_path, get_llava_image_path, get_llava_video_path, get_aokvqa_image_path
from llmperf.preprocessing.modality_size import get_audio_size, get_image_size, get_video_size, get_image_size_aokvqa

@dataclass
class Dataset:
    name: str
    path: Union[str,LiteralString]
    file: str
    alias: str
    color: Optional[str] = None

    get_input: Callable[[Optional[Dict]], Optional[str]] = field(default=None)
    get_output: Callable[[Dict], Optional[str]] = field(default=None)
    _get_modality_path: Callable[[Union[str,LiteralString],Dict], Union[str,LiteralString]] = field(default=None)
    _get_modality_size: Callable[[Union[str,LiteralString],Dict], Union[int,Tuple[int,int],List[Tuple[int,int]]]] = field(default=None)

    def get_modality_path(self, record: Dict) -> Union[str,LiteralString]:
        return self._get_modality_path(self.path, record)

    def get_modality_size(self, record: Dict) -> Union[int,Tuple[int,int],List[Tuple[int,int]]]:
        return self._get_modality_size(self.path, record)

    def __hash__(self):
        return hash((self.name, self.alias))
    
    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.name == other.name and self.alias == other.alias
        return False

    def load(self) -> List[dict]:
        with open(os.path.join(self.path, self.file)) as f:
            data = []
            for line in f:
                data.append(json.loads(line))
            return data

DATASETS = {
    Dataset(
        name="Text Conversations",
        path=os.path.join(DATASETS_DIR, "ShareGPT"),
        file="sharegpt.jsonl",
        alias="text-conv",
        color="#577FBC",
        get_input=get_sharegpt_input,
        get_output=get_sharegpt_output
    ),
    Dataset(
        name="Image Reasoning",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="complex_reasoning.jsonl",
        alias="img-reason",
        color="#57B593",
        get_input=get_llava_image_reasoning_input,
        get_output=get_llava_image_reasoning_output,
        _get_modality_path=get_llava_image_path,
        _get_modality_size=get_image_size
    ),
    ###########################################################
    ###########################################################
    Dataset(
        name="Image Conversation",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="conversation.jsonl",
        alias="img-conv",
        color="#c76955",
        get_input=get_llava_image_reasoning_input,
        get_output=get_llava_image_reasoning_output,
        _get_modality_path=get_llava_image_path,
        _get_modality_size=get_image_size
    ),
    Dataset(
        name="Image Detail",
        path=os.path.join(DATASETS_DIR, "LLaVA-Instruct-150K"),
        file="detail.jsonl",
        alias="img-det",
        color="#46bbaf",
        get_input=get_llava_image_reasoning_input,
        get_output=get_llava_image_reasoning_output,
        _get_modality_path=get_llava_image_path,
        _get_modality_size=get_image_size
    ),
    ###########################################################
    ###########################################################
    Dataset(
        name="Video Description",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="captions_all_duration.jsonl",
        alias="vid-desc",
        color="#F8DE4B",
        get_input=get_llava_video_description_input,
        get_output=get_llava_video_description_output,
        _get_modality_path=get_llava_video_path,
        _get_modality_size=get_video_size
    ),
    ###########################################################
    ###########################################################
    Dataset(
        name="Video Description 0-30",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="cap_0_30_3397.jsonl",
        alias="vid-desc-0-30",
        color="#1848a9",
        get_input=get_llava_video_description_input,
        get_output=get_llava_video_description_output,
        _get_modality_path=get_llava_video_path,
        _get_modality_size=get_video_size
    ),
    Dataset(
        name="Video Multi-Choice 0-30",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="mc_0_30_1663.jsonl",
        alias="vid-mc-0-30",
        color="#a938d3",
        get_input=get_llava_video_description_input,
        # get_output=get_llava_video_description_output,
        get_output=get_llava_video_multi_choice_output,
        _get_modality_path=get_llava_video_path,
        _get_modality_size=get_video_size
    ),
    Dataset(
        name="Video Open-Ended 0-30",
        path=os.path.join(DATASETS_DIR, "LLaVA-Video"),
        file="oe_0_30_3270.jsonl",
        alias="vid-oe-0-30",
        color="#24d6a0",
        get_input=get_llava_video_description_input,
        get_output=get_llava_video_description_output,
        _get_modality_path=get_llava_video_path,
        _get_modality_size=get_video_size
    ),
    ###########################################################
    ###########################################################
    Dataset(
        name="Audio Captioning",
        path=os.path.join(DATASETS_DIR, "Clotho"),
        file="understanding.jsonl",
        alias="audio-cap",
        color="#E16F65",
        get_input=get_clotho_input,
        get_output=get_clotho_output,
        _get_modality_path=get_clotho_audio_path,
        _get_modality_size=get_audio_size
    ),
    ###########################################################
    ###########################################################
    Dataset(
        # image classification-like question answering
        name="A-OKVQA",
        path=os.path.join(DATASETS_DIR, "A-OKVQA"),
        file="aokvqa.jsonl",
        alias="aokvqa",
        color="#b0a435",
        get_input=get_aokvqa_input,
        get_output=get_aokvqa_output,
        _get_modality_path=get_aokvqa_image_path,
        _get_modality_size=get_image_size_aokvqa
    ),
    # Dataset(
    #     # audio classification-like question answering
    #     name="Clotho-AQA",
    #     path=os.path.join("/home/ioannis.dalianis/code/download_MM_datasets/Clotho-AQA", "ClothoAQA"),
    #     file="conversations_full.jsonl",
    #     alias="clotho-aqa",
    #     color="#d880de",
    #     # get_input=get_aokvqa_input,
    #     # get_output=get_aokvqa_output,
    #     # _get_modality_path=get_aokvqa_image_path,
    #     # _get_modality_size=get_image_size_aokvqa
    # )
}

def get_dataset_by_name(name: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "name", None) == name:
            return dataset
    return None

def get_dataset_by_alias(alias: str) -> Union[None, Dataset]:
    for dataset in DATASETS:
        if getattr(dataset, "alias", None) == alias:
            return dataset
    return None