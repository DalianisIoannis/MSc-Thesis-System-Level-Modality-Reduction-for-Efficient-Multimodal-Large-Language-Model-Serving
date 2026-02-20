import time

from datetime import datetime
from transformers import AutoProcessor, AutoTokenizer, LlavaNextProcessor, LlavaNextVideoProcessor
from transformers import PreTrainedTokenizerFast, Qwen2TokenizerFast, Qwen2VLProcessor
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset
from vllm.assets.video import VideoAsset_for_Qwen_plus_CleverSample

from llmperf.config.approaches import get_approach_by_name
from llmperf.config.models import get_model_by_name
from llmperf.config.workloads import get_workload_by_name, Workload, WORKLOADS
from llmperf.postprocessing.output import RequestOutput, ExperimentOutput
from llmperf.utils import load_image

import os
import json
import sys

if __name__ == '__main__':
    START_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    FRAMES = 64

    GPU_UTIL = 0.95
    SWAP_SPACE = 0
    approach = get_approach_by_name("Isolation")

    SAVE_LLM_RESPONSES_FOLDER = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses"
    
    # watch out in the next two lists they have to be in the same order and match workload names with models
    workload_names = [
        # "Text Conversations",
        # "Image Reasoning",
        # "Video Description",
        # "Audio Captioning",
        # "Image Conversation",
        # "Image Detail",
        # "Video Description 0-30",
        "Video Multi-Choice 0-30",
        # "Video Open-Ended 0-30",
        # "A-OKVQA Image Classification",
        # "A-OKVQA Grayscale Image Classification"
        # "A-OKVQA Image Classification Trial 30",
        # "A-OKVQA Image Classification Common Shapes 225"
    ]

    # always have to have at least one model
    model_names = [
        # "Mistral-7b",
        # "LLaVA 1.6 (Mistral-7b)",
        "LLaVA-Next-Video (Mistral-7b)",
        # "Qwen2-Audio-7b",
        # "Qwen2-VL-2B-Instruct",
        # "Qwen2-VL-7B-Instruct",
        # "llava_onevision_qwen2_7b_ov",
        # "llava_onevision_qwen2_0.5b_ov"
        # "pixtral_12b"
    ]

    image_aliases = ["image-static_350", "img-conv_350", "img-det_350", "aokvqa_350", "aokvqa-grayscale_350", "aokvqa_30", "aokvqa-common-shapes_225"]
    out_save = None # do I want to save the outputs somewhere specifically?
    responses_files = None  # will I take the responses from somewhere specific?
    
    if len(sys.argv) > 2: # I am executing for aokvqa from notebook to create workloads
        if sys.argv[2] == "RUN_FROM_NTBK":
            
            # create workload
            if sys.argv[3] == "grayscale":
                
                responses_files = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses/grayscale_compr_responses"
                path_use="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/workloads/static/aokvqa_grayscale_comprs"
                workload_new_name = "A-OKVQA Image Classification Grayscale " + sys.argv[1]
                name_search_alias = "aokvqa-grayscale_" + sys.argv[1]
                out_save = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/folder_350_aokvqa_grayscale"
            
            elif sys.argv[3] == "common_shapes_225":
                
                responses_files = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses/rgb_compr_responses"
                path_use="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/workloads/static/aokvqa_rgb_comprs"
                workload_new_name = "A-OKVQA Image Classification Common Shapes " + sys.argv[1]
                name_search_alias = "aokvqa-common-shapes_" + sys.argv[1]
                out_save = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/folder_350_aokvqa_rgb"
            
            else:
                
                responses_files = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses/rgb_compr_responses"
                path_use="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/workloads/static/aokvqa_rgb_comprs"
                workload_new_name = "A-OKVQA Image Classification " + sys.argv[1]
                name_search_alias = "aokvqa_" + sys.argv[1]
                out_save = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/folder_350_aokvqa_rgb"
            
            workload_names = [workload_new_name]
            
            image_aliases.append(name_search_alias)
            workload_new =  Workload(
                    name=workload_new_name,
                    path=path_use,
                    alias=name_search_alias,
                    modalities="image",
                    modality_pct=1.0

            )
            WORKLOADSIN = workload_new
            
            # this workload has already been created
            for resp_file in os.listdir(responses_files):
                if name_search_alias in resp_file:
                    print(f"From subprocess: {name_search_alias}, exists already")
                    exit(0)
            

    for workload_name, model_name in zip(workload_names, model_names):
        print(110*"*", "\n", workload_name, " --- ", model_name, "\n", 110*"*", "\n", sep="")
        start_time_loop = time.time()
        try:
            model = get_model_by_name(model_name)

            if model_name in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"]:
                llm = LLM(
                    model=model.path,
                    gpu_memory_utilization=GPU_UTIL,
                    swap_space=SWAP_SPACE,
                    scheduling_policy=approach.scheduling_policy,
                    enable_custom_scheduler=approach.enable_custom_scheduler,
                    enable_chunked_prefill=approach.enable_chunked_prefill,
                    ####################################
                    max_num_seqs=5,
                    limit_mm_per_prompt={"image": 64, "video": 1}
                    ####################################
                )
            elif "pixtral" in model_name:
                llm = LLM(
                    model=model.path,
                    max_model_len=model.max_model_len,
                    gpu_memory_utilization=GPU_UTIL,
                    swap_space=SWAP_SPACE,
                    scheduling_policy=approach.scheduling_policy,
                    enable_custom_scheduler=approach.enable_custom_scheduler,
                    enable_chunked_prefill=approach.enable_chunked_prefill
                )
            else:
                # llava_onevision_qwen2_7b_ov so far goes here
                llm = LLM(
                    model=model.path,
                    gpu_memory_utilization=GPU_UTIL,
                    swap_space=SWAP_SPACE,
                    scheduling_policy=approach.scheduling_policy,
                    # enable_custom_scheduler=approach.enable_custom_scheduler,
                    # enable_chunked_prefill=approach.enable_chunked_prefill
                )

            if len(sys.argv) == 1:
                workload = get_workload_by_name(workload_name)
            else:
                workload = WORKLOADSIN
            workload.load()
            requests = workload.requests

            outputs = []
            llm_responses = []
            modality_token_index = -1
            start_time = time.time()
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

                # if workload.alias == "text-static":
                if workload.alias == "text-static_350":
                    modality_token_index = -1
                    tokenizer = AutoTokenizer.from_pretrained(model.path)
                    prompt = [
                        {"role": "user", "content": request.input}
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                    final_prompt = {
                        "prompt": formatted_prompt
                    }

                # if workload.alias == "image-static":
                if workload.alias in image_aliases:
                    modality_token_index = model.image_token_index
                    if isinstance(tokenizer, Qwen2TokenizerFast):
                        processor = Qwen2VLProcessor.from_pretrained(model.path)
                    # elif isinstance(tokenizer, PreTrainedTokenizerFast):
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
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    
                    # here I can play with the image
                    image = load_image(request.modality_path)

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": image}
                    }

                # if workload.alias == "video-static":
                if workload.alias in ["video-static_350", "vid-desc-0-30_350", "vid-mc-0-30_350", "vid-oe-0-30_350", "vid-mc-0-30_799"]: # the vid-mc-0-30_799 is trial
                    modality_token_index = model.video_token_index
                    if isinstance(tokenizer, Qwen2TokenizerFast):
                        processor = AutoProcessor.from_pretrained(model.path)
                        # processor = AutoProcessor.from_pretrained(model.path, do_rescale=False, image_size=(224, 224))
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
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    if model_name in ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct"]:
                        video = VideoAsset_for_Qwen_plus_CleverSample(name=request.modality_path, num_frames=FRAMES, param_dict=None).np_ndarrays
                    else:
                        video = VideoAsset(name=request.modality_path, num_frames=FRAMES).np_ndarrays

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"video": video}
                    }

                # if workload.alias == "audio-static":
                if workload.alias == "audio-static_350":
                    modality_token_index = model.audio_token_index
                    processor = AutoProcessor.from_pretrained(model.path)

                    audio_in_prompt = "".join([
                        f"Audio 1: "
                        f"<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                    ])
                    prompt = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio"},
                                {"type": "text", "text": audio_in_prompt + request.input}
                            ]
                        }
                    ]
                    formatted_prompt = processor.apply_chat_template(
                        prompt,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    audio = AudioAsset(request.modality_path).audio_and_sample_rate

                    final_prompt = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"audio": audio}
                    }

                req_output = llm.generate(
                    final_prompt,
                    sampling_params,
                    use_tqdm=False
                )[0]

                # this is the model's response to the prompt
                # req_output.outputs[0].text
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
                    )
                )

        finally:
            elapsed_time = now - start_time

            experiment_output = ExperimentOutput(
                id=f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}",
                elapsed_time=elapsed_time,
                request_outputs=outputs
            )
            # experiment_output.save()
            if out_save:
                experiment_output.save(out_save)
            else:
                experiment_output.save()

            llm = None

            # save llm responses
            if responses_files:
                save_path_resp = os.path.join(responses_files, f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}-responses.jsonl")
            else:
                save_path_resp = os.path.join(SAVE_LLM_RESPONSES_FOLDER, f"{workload.alias}-{model.alias}-{approach.alias}-{START_TIME}-responses.jsonl")
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