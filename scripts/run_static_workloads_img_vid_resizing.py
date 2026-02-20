import sys
sys.path.append("/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration")
import plot_utils

import time
import numpy as np

from datetime import datetime
from transformers import AutoProcessor, AutoTokenizer, LlavaNextProcessor, LlavaNextVideoProcessor
from transformers import Qwen2TokenizerFast, Qwen2VLProcessor, PreTrainedTokenizerFast
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset, VideoAsset_CleverSample, VideoAsset_for_Qwen_plus_CleverSample

from llmperf.config.approaches import get_approach_by_name
from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import Workload, WORKLOADS
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import load_image

import os
import json
import sys

def start_llm(model, approach, gpu_util, swap_space):
    """
    Starts an LLM (Large Language Model) based on the given model, approach, gpu memory utilization, and swap space.

    Parameters
    ----------
    model : Model
        The model to use for the LLM.
    approach : Approach
        The approach to use for the LLM.
    gpu_util : float
        The GPU memory utilization to use for the LLM.
    swap_space : bool
        Whether or not to use swap space for the LLM.

    Returns
    -------
    LLM
        The started LLM.
    """
    if model.name in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"]:
        llm = LLM(
            model=model.path,
            gpu_memory_utilization=gpu_util,
            swap_space=swap_space,
            scheduling_policy=approach.scheduling_policy,
            disable_log_stats=False,
            max_num_seqs=5,
            limit_mm_per_prompt={"image": 64, "video": 1}
        )
    elif "pixtral" in model.name:
        llm = LLM(
            model=model.path,
            max_model_len=model.max_model_len,
            # max_model_len=45056,
            gpu_memory_utilization=gpu_util,
            swap_space=swap_space,
            disable_log_stats=False,
            scheduling_policy=approach.scheduling_policy,
            # max_num_batched_tokens=45056,
            num_gpu_blocks_override=2816,
            limit_mm_per_prompt={"image": 64},
            disable_mm_preprocessor_cache=True,
            max_num_seqs=1
        )
    else:
        # llava_onevision_qwen2_7b_ov so far goes here
        llm = LLM(
            model=model.path,
            gpu_memory_utilization=gpu_util,
            swap_space=swap_space,
            scheduling_policy=approach.scheduling_policy,
            disable_log_stats=False,
        )
    
    return llm

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    FRAMES = 64

    total_video_asset = None # time that the VideoAsset takes for clever sampling. Will be None in the other two cases
    there_are_frames = None # frames in the video, we only care about this in the video_clever_sampling case and in the video case

    GPU_UTIL = 0.95
    SWAP_SPACE = 0
    approach = get_approach_by_name("Isolation")

    type = sys.argv[2] # whether or not we have imgs or videos
    
    if type == "image":
        type_workload = "image"
        
        # model_name = "LLaVA 1.6 (Mistral-7b)"
        # model_name = "Qwen2-VL-2B-Instruct"
        # model_name = "Qwen2-VL-7B-Instruct"
        # model_name = "llava_onevision_qwen2_0.5b_ov"
        # model_name = "llava_onevision_qwen2_7b_ov"
        model_name = "pixtral_12b"
        
        out_save = plot_utils.OUTPUTS_350_FOLDER_AOKVQA # from paths_n_filters
        path_use=plot_utils.AOKVQA_RGB_STATICS # from paths_n_filters
    else:   # video
        type_workload = "video"
        # SOS #
        # model_name = "LLaVA-Next-Video (Mistral-7b)"
        # model_name = "Qwen2-VL-2B-Instruct"
        # model_name = "Qwen2-VL-7B-Instruct"
        model_name = "llava_onevision_qwen2_0.5b_ov"
        out_save = plot_utils.OUTPUTS_350_FOLDER_LLAVA_VID
        path_use=plot_utils.LLAVA_VID_RGB_STATICS
        if type == "video_clever_sampling":
    
            strategy = sys.argv[3]
            param_dict = json.loads(sys.argv[4])

            actual_param_dict = {}
            for key, value in param_dict.items():
                if key == "max_frames":
                    actual_param_dict[key] = plot_utils.MAX_FRAMES[value]
                elif key == "target_fps":
                    actual_param_dict[key] = plot_utils.FPS[value]
                elif key == "content_threshold":
                    actual_param_dict[key] = plot_utils.CONTENT_THRESHOLD[value]
                elif key == "motion_threshold":
                    actual_param_dict[key] = plot_utils.MOTION_THRESHOLD[value]
                elif key == "sharpness_threshold":
                    actual_param_dict[key] = plot_utils.SHARPNESS_THRESHOLD[value]
            print(f"Initial value in main_script: {actual_param_dict}") # Access via module object

    workload_name = sys.argv[1]

    workload =  Workload(
            name=workload_name,
            path=path_use,
            alias=workload_name,
            modalities=type_workload,
            modality_pct=1.0
    )

    print(110*"*", "\n", workload_name, " --- ", model_name, "\n", 110*"*", "\n", sep="")
    
    start_time_loop = time.time()
    try:
        model = get_model_by_name(model_name)

        llm = start_llm(model, approach, GPU_UTIL, SWAP_SPACE)

        workload.load()
        requests = workload.requests

        outputs = []
        llm_responses = []
        modality_token_index = -1
        start_time = time.time()
        ######################################
        # # I will save from every workload a number of frames as returned from the clever sampling
        # MAX_SAMPLED_FRAMES_TO_SAVE = 25
        ######################################
        for request in tqdm(requests):    
            tokenizer = AutoTokenizer.from_pretrained(model.path)
            if isinstance(tokenizer, Qwen2TokenizerFast) or isinstance(tokenizer, PreTrainedTokenizerFast):
                output_length = len(tokenizer.encode(request.output))
            else:
                output_length = len(tokenizer.encode(request.output)[1:])

            sampling_params = SamplingParams(
                # if ignore_eos False and max_tokens None or commented, might
                # return only letters for video multiple choice. Have to check
                ignore_eos=True,
                max_tokens=int(output_length)
            )


            if type == "image":
                modality_token_index = model.image_token_index
                if isinstance(tokenizer, Qwen2TokenizerFast):
                    processor = Qwen2VLProcessor.from_pretrained(model.path)
                elif "pixtral" in model_name:
                    processor = AutoProcessor.from_pretrained(model.path)
                else:
                    processor = LlavaNextProcessor.from_pretrained(model.path)
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": request.input}
                        ]
                    }
                ]
                # here I can play with the image
                image = load_image(request.modality_path)
                modal = image
            elif type == "video":
                modality_token_index = model.video_token_index
                if isinstance(tokenizer, Qwen2TokenizerFast):
                    processor = AutoProcessor.from_pretrained(model.path)
                else:
                    processor = LlavaNextVideoProcessor.from_pretrained(model.path)
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            # {"type": "text", "text": request.input}
                            {"type": "text", "text": "<video>\n" + request.input}
                        ]
                    }
                ]
                video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays
                ################################
                # video.shape[0] is the number of frames
                there_are_frames = video.shape[0]
                ################################
                modal = video
            elif type == "video_clever_sampling":
                modality_token_index = model.video_token_index
                if isinstance(tokenizer, Qwen2TokenizerFast):
                    processor = AutoProcessor.from_pretrained(model.path)
                else:
                    processor = LlavaNextVideoProcessor.from_pretrained(model.path)
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": "<video>\n" + request.input}
                        ]
                    }
                ]
                # video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays
                t0_video_asset = time.time()
                if model_name in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"]:
                    video = VideoAsset_for_Qwen_plus_CleverSample(name=request.modality_path, strategy=strategy, num_frames=FRAMES, param_dict=actual_param_dict).np_ndarrays
                else:
                    video = VideoAsset_CleverSample(name=request.modality_path, strategy=strategy, param_dict=actual_param_dict).np_ndarrays
                ################################
                # video.shape[0] is the number of frames
                # those are used just for checknig and visualizing
                there_are_frames = video.shape[0]
                # sampled_frames = plot_utils.MAX_SAMPLED_FRAMES_TO_SAVE
                # if sampled_frames > 0:
                #     if not os.path.exists(os.path.join(plot_utils.SAVED_FRAMES_FOLDER, workload_name)):
                #         os.makedirs(os.path.join(plot_utils.SAVED_FRAMES_FOLDER, workload_name))
                #     np.save(os.path.join(plot_utils.SAVED_FRAMES_FOLDER, workload_name, f"{request.id}.npy"), video)
                #     sampled_frames -= 1
                ################################
                t1_video_asset = time.time()
                total_video_asset = t1_video_asset-t0_video_asset
                modal = video
            
            formatted_prompt = processor.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False
            )
            
            final_prompt = {
                "prompt": formatted_prompt,
                # "multi_modal_data": {type: modal}
                "multi_modal_data": {type_workload: modal}
            }

            req_output = llm.generate(
                final_prompt,
                sampling_params,
                use_tqdm=False
            )[0]

            # this is the model's response to the prompt
            llm_responses.append({
                "id": request.id,
                "correct_answer": request.output,
                "llm_response": req_output.outputs[0].text
                })

            now = time.time()
            outputs.append(
                RequestOutput(
                    id=request.id,
                    prompt_tokens_cnt=len(req_output.prompt_token_ids),
                    modality_tokens_cnt=req_output.prompt_token_ids.count(modality_token_index),
                    decode_tokens_cnt=len(req_output.outputs[0].token_ids),
                    processor_time=req_output.metrics.processor_time,
                    encoder_time=req_output.metrics.encoder_time if req_output.metrics.encoder_time else 0,
                    ttft=req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time,
                    tbt=0 if len(req_output.outputs[0].token_ids) <= 1 else (req_output.metrics.finished_time - req_output.metrics.first_token_time) / (len(req_output.outputs[0].token_ids)-1),
                    e2e=req_output.metrics.finished_time - req_output.metrics.first_scheduled_time,
                    arrival_time=req_output.metrics.arrival_time,
                    last_token_time=req_output.metrics.last_token_time,
                    first_scheduled_time=req_output.metrics.first_scheduled_time,
                    first_token_time=req_output.metrics.first_token_time,
                    time_in_queue=req_output.metrics.time_in_queue,
                    finished_time=req_output.metrics.finished_time,
                    scheduler_time=req_output.metrics.scheduler_time,
                    model_forward_time=req_output.metrics.model_forward_time,
                    model_execute_time=req_output.metrics.model_execute_time
                    ###########################################################
                    , video_asset_time=total_video_asset
                    , video_frames=there_are_frames
                    ###########################################################
                )
            )

    finally:
        elapsed_time = now - start_time

        experiment_output = ExperimentOutput(
            id=f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}",
            elapsed_time=elapsed_time,
            request_outputs=outputs
        )
        
        if out_save:
            experiment_output.save(out_save)

        llm = None

        save_path_resp = os.path.join(plot_utils.SAVE_LLM_RESPONSES_FOLDER, f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}-responses.jsonl")
        with open(save_path_resp, "w", encoding="utf-8") as file:
            for request_output in llm_responses:
                file.write(json.dumps(request_output) + "\n")
            file.close()
    
    end_time_loop = time.time()
    elapsed_time_loop = end_time_loop - start_time_loop
    # Convert seconds to hours, minutes, and seconds
    hours, rem = divmod(elapsed_time_loop, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"{'*' * 110}\nIteration took {int(hours)}h {int(minutes)}m {seconds:.2f}s\n{'*' * 110}\n")