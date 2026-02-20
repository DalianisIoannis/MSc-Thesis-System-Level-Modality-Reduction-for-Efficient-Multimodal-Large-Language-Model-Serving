from plot_utils import *

WORKLOAD_DICT = {techn: [] for techn in FRAME_SAMPLING_TECHNIQUE}

# uniform
for max_fram in MAX_FRAMES:
    new_work = {"max_frames": max_fram}
    WORKLOAD_DICT["uniform"].append(new_work)

# fixed_rate
for fps in FPS:
    new_work = {"target_fps": fps}
    WORKLOAD_DICT["fixed_rate"].append(new_work)

# scene_change
for thresh in CONTENT_THRESHOLD:
    for max_fram in MAX_FRAMES:
        new_work = {"content_threshold": thresh, "max_frames": max_fram}
        WORKLOAD_DICT["scene_change"].append(new_work)

# sharpness_based
for sharp_thresh in SHARPNESS_THRESHOLD:
    for max_fram in MAX_FRAMES:
        new_work = {"sharpness_threshold": sharp_thresh, "max_frames": max_fram}
        WORKLOAD_DICT["sharpness_based"].append(new_work)

# motion_based
for mot_thresh in MOTION_THRESHOLD:
    for max_fram in MAX_FRAMES:
        new_work = {"motion_threshold": mot_thresh, "max_frames": max_fram}
        WORKLOAD_DICT["motion_based"].append(new_work)

print_dictionary_content(WORKLOAD_DICT, limit_print=100)

for strategy in WORKLOAD_DICT.keys():
    print(f"Strategy: {strategy}")
    for workload in WORKLOAD_DICT[strategy]:
        print(f"\tWorkload: {workload}")

        create_resized_video_n_jsonl_frame_sampling(
            init_jsonl_request_pth=JSONL_FILE,
            strategy=strategy,
            sub_workload=workload,
            # model_alias="qwen-2b-instruct" # halps define if we are working on a model other than the default
            # model_alias="qwen-7b-instruct" # halps define if we are working on a model other than the default
            model_alias="llava_onevision_qwen2_0.5b_ov" # halps define if we are working on a model other than the default
        )


# nohup
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env/bin/python
# or
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env_v1_konpap/bin/python
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vid_experiments_clever_sampling.py
# >
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vids.log 2>&1 < /dev/null &


# nohup /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env/bin/python /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vid_experiments_clever_sampling.py > /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vids.log 2>&1 < /dev/null &
# or
# nohup /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env_v1_konpap/bin/python /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vid_experiments_clever_sampling.py > /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_vids.log 2>&1 < /dev/null &