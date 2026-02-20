import io
import numpy as np
import time

from PIL import Image
# from transformers import PreTrainedTokenizerFast, Qwen2TokenizerFast, Qwen2VLProcessor
# from tqdm import tqdm
# from vllm.assets.audio import AudioAsset

from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import get_workload_by_name, Workload
# , WORKLOADS
from llmperf.config.approaches import get_approach_by_name
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import load_image

import os
import json

import sys
sys.path.append("/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration")
from plot_utils import OUTPUTS_350_FOLDER_AOKVQA, AOKVQA_RGB_STATICS,\
OUTPUTS_350_FOLDER_LLAVA_VID, LLAVA_VID_RGB_STATICS, MAX_FRAMES, FPS,CONTENT_THRESHOLD, MOTION_THRESHOLD,\
SHARPNESS_THRESHOLD, PTH_TO_PROJECT_FOLDER

from datetime import datetime
from transformers import AutoProcessor, AutoTokenizer, LlavaNextProcessor, LlavaNextVideoProcessor
from transformers import Qwen2TokenizerFast, Qwen2VLProcessor, PreTrainedTokenizerFast
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset, VideoAsset_for_Qwen_plus_CleverSample, VideoAsset_CleverSample



# SAVE_LLM_RESPONSES_FOLDER = None
SAVE_LLM_RESPONSES_FOLDER = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses"

# FRAMES = 64
GPU_UTIL = 0.95
SWAP_SPACE = 0

FRAMES = 64

WORKLOAD_NAME = "Video Multi-Choice 0-30"
TYPE_WORKLOAD = "video" # whether we are working with imgs or vids

# WORKLOAD_NAME = "A-OKVQA Image Classification"
# TYPE_WORKLOAD = "image" # whether we are working with imgs or vids


TEXT_ALIASES = ["text-static_350"]
IMAGE_ALIASES = ["image-static_350", "img-conv_350", "img-det_350", "aokvqa_350", "aokvqa-grayscale_350", "aokvqa_30", "aokvqa-common-shapes_225"]
VIDEO_ALIASES = ["video-static_350", "vid-desc-0-30_350", "vid-mc-0-30_350", "vid-oe-0-30_350", "vid-mc-0-30_799"]
AUDIO_ALIASES = ["audio-static_350"]


# SOS #
# MODEL_NAME = "pixtral_12b"
# MODEL_NAME = "LLaVA-Next-Video (Mistral-7b)"
# MODEL_NAME = "Qwen2-VL-2B-Instruct"
# MODEL_NAME = "Qwen2-VL-7B-Instruct"
MODEL_NAME = "llava_onevision_qwen2_0.5b_ov"
# MODEL_NAME = "llava_onevision_qwen2_7b_ov"

def load_image_from_array(array: np.ndarray) -> Image.Image:
    """
    Loads an image from a given numpy array.

    Parameters
    ----------
    array : np.ndarray
        A numpy array representing the image.

    Returns
    -------
    Image.Image
        The loaded image.
    """
    img = Image.fromarray(array.astype("uint8"))
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    image_file = Image.open(buf)
    return image_file

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

def build_prompt(request, model, workload, num_frames, param_dict=None):

    total_video_asset = None
    there_are_frames = None
    print(f"In build_prompt, workload.alias: {workload.alias}")

    if workload.alias in TEXT_ALIASES:
        pass
        # modality_token_index = -1
        # tokenizer = AutoTokenizer.from_pretrained(model.path)
        # prompt = [
        #     {"role": "user", "content": request.input}
        # ]
        # formatted_prompt = tokenizer.apply_chat_template(
        #     prompt,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        # final_prompt = {
        #     "prompt": formatted_prompt
        # }

    elif workload.alias in IMAGE_ALIASES or "_bdp_lan_" in workload.alias:
        modality_token_index = model.image_token_index
        
        tokenizer = AutoTokenizer.from_pretrained(model.path)
        if isinstance(tokenizer, Qwen2TokenizerFast):
            processor = Qwen2VLProcessor.from_pretrained(model.path)
        elif "pixtral" in model.name:
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
        formatted_prompt = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )
        image = load_image(request.modality_path)
        final_prompt = {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image}
        }

    elif workload.alias in VIDEO_ALIASES or "vid-mc-0-30_350_" in workload.alias:
        t0_video_asset = time.time()

        if model.name in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"]:
        # if isinstance(tokenizer, Qwen2TokenizerFast):
            # video = VideoAsset_for_Qwen_plus_CleverSample(name=request.modality_path, num_frames=FRAMES, param_dict=None).np_ndarrays
            video = VideoAsset_for_Qwen_plus_CleverSample(name=request.modality_path, num_frames=num_frames, param_dict=param_dict).np_ndarrays
            # video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays
        else:
            if param_dict is not None:
                video = VideoAsset_CleverSample(name=request.modality_path, strategy=strategy, param_dict=param_dict).np_ndarrays
            else:
                # if model.name in ["llava_onevision_qwen2_0.5b_ov", "llava_onevision_qwen2_7b_ov"]:
                #     video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays
                video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays
        
        there_are_frames = video.shape[0]
        t1_video_asset = time.time()
        total_video_asset = t1_video_asset - t0_video_asset
        
        if "pixtral" in model.name:
            pass
            # # Use multi image approach
            # modality_token_index = model.image_token_index
            # processor = AutoProcessor.from_pretrained(model.path, use_fast=True)
            
            # frames = []
            # for frame in video:
            #     image = load_image_from_array(frame)
            #     frames.append(image)

            # frame_placeholders = [{"type": "image"}] * num_frames
            # prompt = [
            #     {
            #         "role": "user",
            #         "content": [
            #             *frame_placeholders,
            #             {"type": "text", "text": request.input}
            #         ]
            #     }
            # ]

            # modal_name = "image"
            # modal = frames

        # # elif model.name in ["llava_onevision_qwen2_0.5b_ov", "llava_onevision_qwen2_7b_ov"]:
        else:
            modality_token_index = model.video_token_index
            tokenizer = AutoTokenizer.from_pretrained(model.path)
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
            modal_name = "video"
            modal = video

        formatted_prompt = processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        final_prompt = {
            "prompt": formatted_prompt,
            # "multi_modal_data": {"video": video}
            # "multi_modal_data": {"image": frames}
            "multi_modal_data": {modal_name: modal}
        }

    elif workload.alias in AUDIO_ALIASES :
        pass
        # modality_token_index = model.audio_token_index
        # processor = AutoProcessor.from_pretrained(model.path)
        # audio_in_prompt = "".join([
        #     f"Audio 1: "
        #     f"<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        # ])
        # prompt = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "audio"},
        #             {"type": "text", "text": audio_in_prompt + request.input}
        #         ]
        #     }
        # ]
        # formatted_prompt = processor.apply_chat_template(
        #     prompt,
        #     add_generation_prompt=True,
        #     tokenize=False
        # )
        # audio = AudioAsset(request.modality_path).audio_and_sample_rate
        # final_prompt = {
        #     "prompt": formatted_prompt,
        #     "multi_modal_data": {"audio": audio}
        # }
    else:
        raise ValueError("Could not build prompt. Unknown workload")

    return final_prompt, modality_token_index, total_video_asset, there_are_frames

def get_running_parameters():
    
    approach = get_approach_by_name("Isolation")
    out_save = None # means I am running default and it will save to its default place
    actual_param_dict = None
    save_llm_resp_folder = SAVE_LLM_RESPONSES_FOLDER

    if len(sys.argv) > 1:
        workload_name = sys.argv[1]
        type_run = sys.argv[2] # whether or not we have imgs or videos
        strategy = sys.argv[3]
        param_dict = json.loads(sys.argv[4])
        save_llm_resp_folder = os.path.join(PTH_TO_PROJECT_FOLDER, "artifacts/outputs/llm_responses/rgb_compr_responses")
        try:
            model_name = sys.argv[5]
        except IndexError:
            model_name = MODEL_NAME
    else:
        # SAVE_LLM_RESPONSES_FOLDER = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses"
        workload_name = WORKLOAD_NAME
        type_run = TYPE_WORKLOAD
        strategy = None
        param_dict = None
        model_name = MODEL_NAME
    
    if type_run == "image":
        type_run_workload = "image"
        path_use=AOKVQA_RGB_STATICS
        if len(sys.argv) > 1:
            print(f"Will create workload {workload_name} from {path_use}")
            out_save = OUTPUTS_350_FOLDER_AOKVQA
            workload =  Workload(
                name=workload_name,
                path=path_use,
                alias=workload_name,
                modalities=type_run_workload,
                modality_pct=1.0
            )
        else:
            print(f"Will use workload {workload_name}")
            workload = get_workload_by_name(workload_name)
    else:
        type_run_workload = "video"
        path_use=LLAVA_VID_RGB_STATICS
        if len(sys.argv) > 1:
            out_save = OUTPUTS_350_FOLDER_LLAVA_VID
        
        if type_run == "video_clever_sampling":
            workload =  Workload(
                name=workload_name,
                path=path_use,
                alias=workload_name,
                modalities=type_run_workload,
                modality_pct=1.0
            )
            actual_param_dict = {}
            for key, value in param_dict.items():
                if key == "max_frames":
                    actual_param_dict[key] = MAX_FRAMES[value]
                elif key == "target_fps":
                    actual_param_dict[key] = FPS[value]
                elif key == "content_threshold":
                    actual_param_dict[key] = CONTENT_THRESHOLD[value]
                elif key == "motion_threshold":
                    actual_param_dict[key] = MOTION_THRESHOLD[value]
                elif key == "sharpness_threshold":
                    actual_param_dict[key] = SHARPNESS_THRESHOLD[value]
            print(f"Initial value in main_script: {actual_param_dict}") # Access via module object
        else:
            workload = get_workload_by_name(WORKLOAD_NAME)

    return workload_name, workload, approach, strategy, actual_param_dict, model_name, out_save, save_llm_resp_folder

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

    total_video_asset = None # time that the VideoAsset takes for clever sampling. Will be None in the other two cases
    there_are_frames = None # frames in the video, we only care about this in the video_clever_sampling case and in the video case

    workload_name, workload, approach, strategy, param_dict, model_name, out_save,\
        save_llm_resp_folder = get_running_parameters()
    
    print(110*"*", "\n", workload_name, " --- ", model_name, "\n", 110*"*", "\n", sep="")
    
    start_time_loop = time.time()
    try:
        model = get_model_by_name(model_name)
        llm = start_llm(model, approach, GPU_UTIL, SWAP_SPACE)

        workload.load()
        requests = workload.requests
        print(requests[5])

        outputs = []
        llm_responses = []
        modality_token_index = -1
        start_time = time.time()

        # for request in tqdm(requests):    
        for request in tqdm(requests[:2]):    
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

            technique = len(sys.argv) > 1
            final_prompt, modality_token_index, total_video_asset,\
                there_are_frames = build_prompt(request, model, workload, num_frames=FRAMES, param_dict=param_dict)
                                                                                                #    , technique=technique)

            req_outputs = llm.generate(
                final_prompt,
                sampling_params,
                use_tqdm=False
            )

            # now = time.time()
            now = time.time()
            # TODO: Fix Request Output creation (metrics and fields)
            if req_outputs is None or len(req_outputs) < 1:
                # Aborted request because prompt was longer than max model length
                import logging
                logging.warning(f"Request {request.id} was aborted because the prompt was longer than the model's maximum length.")
                processed_inputs = llm.llm_engine.processor.input_preprocessor.preprocess(
                    final_prompt,
                    lora_request=None,
                    prompt_adapter_request=None,
                    return_mm_hashes=False,
                )
                prompt_token_ids = processed_inputs["prompt_token_ids"]
                outputs.append(
                    RequestOutput(
                        id=request.id,
                        prompt_token_cnt=len(prompt_token_ids),
                        modality_tokens_cnt=prompt_token_ids.count(modality_token_index),
                        decode_tokens_cnt=int(output_length),
                        aborted=True
                    )
                )
            else:
                req_output = req_outputs[0]

                llm_responses.append({
                    "id": request.id,
                    "correct_answer": request.output,
                    "llm_response": req_output.outputs[0].text
                })
                
                processor_time = req_output.metrics.input_processed_time - req_output.metrics.arrival_time
                encoder_time = req_output.metrics.model_encoder_time
                # ttft = req_output.metrics.processor_time + req_output.metrics.time_in_queue + (req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time)
                ttft = processor_time + req_output.metrics.time_in_queue + (req_output.metrics.first_token_time - req_output.metrics.first_scheduled_time)
                e2e = req_output.metrics.finished_time - req_output.metrics.arrival_time
                tbt = 0.0 if len(req_output.outputs[0].token_ids) <= 1 else (e2e - ttft) / (len(req_output.outputs[0].token_ids) - 1)

                outputs.append(
                    RequestOutput(
                        id=request.id,
                        prompt_tokens_cnt=len(req_output.prompt_token_ids),
                        modality_tokens_cnt=req_output.prompt_token_ids.count(modality_token_index),
                        decode_tokens_cnt=len(req_output.outputs[0].token_ids),
                        
                        processor_time=processor_time,
                        encoder_time=encoder_time,
                        ttft=ttft,
                        tbt=tbt,
                        e2e=e2e,
                        
                        arrival_time=req_output.metrics.arrival_time,
                        last_token_time=req_output.metrics.last_token_time,
                        first_scheduled_time=req_output.metrics.first_scheduled_time,
                        first_token_time=req_output.metrics.first_token_time,
                        time_in_queue=req_output.metrics.time_in_queue,
                        finished_time=req_output.metrics.finished_time,
                        scheduler_time=req_output.metrics.scheduler_time,
                        model_forward_time=req_output.metrics.model_forward_time,
                        model_execute_time=req_output.metrics.model_execute_time,
                        aborted=False
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
        else:
            experiment_output.save()
        file = f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}-responses.jsonl"
        save_path_resp = os.path.join(save_llm_resp_folder, file)

        llm = None

        print(f"Saved experiment responses to {save_path_resp}")

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

# python3 scripts/universal_static_workloads.py 