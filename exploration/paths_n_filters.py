import os
from PIL import Image
import json

# Get absolute path to the directory of this file
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go one directory up to get the project folder
PTH_TO_PROJECT_FOLDER = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))

# for run static img resizing
SAVE_LLM_RESPONSES_FOLDER = os.path.join(PTH_TO_PROJECT_FOLDER, "artifacts/outputs/llm_responses/rgb_compr_responses")
RESIZE_PROCEDURE_STATS = os.path.join(PTH_TO_PROJECT_FOLDER, "exploration/util_files/resizing_times.jsonl")

# files and paths and parameters only for img experimentation
# Paths
OUTPUTS_FOLDER = os.path.join(PTH_TO_PROJECT_FOLDER, "artifacts/outputs")
OUTPUTS_350_FOLDER = os.path.join(OUTPUTS_FOLDER, "folder_350")
OUTPUTS_350_FOLDER_AOKVQA = os.path.join(OUTPUTS_FOLDER, "folder_350_aokvqa_rgb")

# # this defines the number of samples that I will save from every workload as returned from the clever sampling
# # MAX_SAMPLED_FRAMES_TO_SAVE = 25
# MAX_SAMPLED_FRAMES_TO_SAVE = 5
# SAVED_FRAMES_FOLDER = os.path.join(PTH_TO_PROJECT_FOLDER, "exploration/saved_sampled_frames/")

LLM_RESPONSES_FOLDER = os.path.join(OUTPUTS_FOLDER, "llm_responses")
LLM_RESPONSES_RGB_AOKVQA = os.path.join(LLM_RESPONSES_FOLDER, "rgb_compr_responses")

STATIC_WORKLOAD_FOLDER = os.path.join(PTH_TO_PROJECT_FOLDER, "artifacts/workloads/static")
FULL_STATIC_WORKLOADS = os.path.join(STATIC_WORKLOAD_FOLDER, "created_full")
AOKVQA_RGB_STATICS = os.path.join(STATIC_WORKLOAD_FOLDER, "aokvqa_rgb_comprs")

AOKVQA_FOLDER = "/srv/muse-lab/datasets/A-OKVQA"
AOKVQA_IMAGES_FOLDER = os.path.join(AOKVQA_FOLDER, "images")

# Files
AOKVQA_FULL_WORKLOAD = os.path.join(FULL_STATIC_WORKLOADS, "aokvqa_1000.jsonl")
AOKVQA_350_WORKLOAD = os.path.join(STATIC_WORKLOAD_FOLDER, "aokvqa_350.jsonl")
AOKVQA_350_STATISTICS = os.path.join(OUTPUTS_FOLDER, "aokvqa_350-image-mistral-iso-20250404-094306.jsonl")
AOKVQA_350_RESPONSES = os.path.join(LLM_RESPONSES_FOLDER, "aokvqa_350-image-mistral-iso-20250404-094306-responses.jsonl")

# Other models
# AOKVQA_350_QWEN_TWO_STATISTICS = os.path.join(OUTPUTS_FOLDER, "aokvqa_350-qwen-2b-instruct-iso-20250923-165520.jsonl")
# THEY ARE INITIALLY LISTS SO THAT THAY CAN BE CHECKED IN exploration/img_experimentation.ipynb IF THERE ARE MORE THAN ONE
AOKVQA_350_QWEN_TWO_STATISTICS = [os.path.join(OUTPUTS_FOLDER, f) for f in os.listdir(OUTPUTS_FOLDER) if "aokvqa_350-qwen-2b-instruct" in f and f.endswith(".jsonl")]
AOKVQA_350_QWEN_TWO_RESPONSES = [os.path.join(LLM_RESPONSES_FOLDER, f) for f in os.listdir(LLM_RESPONSES_FOLDER) if "aokvqa_350-qwen-2b-instruct" in f and f.endswith(".jsonl")]

# AOKVQA_350_QWEN_SEVEN_RESPONSES = os.path.join(LLM_RESPONSES_FOLDER, "aokvqa_350-qwen-7b-instruct-iso-20250923-183139-responses.jsonl")
AOKVQA_350_QWEN_SEVEN_STATISTICS = [os.path.join(OUTPUTS_FOLDER, f) for f in os.listdir(OUTPUTS_FOLDER) if "aokvqa_350-qwen-7b-instruct" in f and f.endswith(".jsonl")]
AOKVQA_350_QWEN_SEVEN_RESPONSES = [os.path.join(LLM_RESPONSES_FOLDER, f) for f in os.listdir(LLM_RESPONSES_FOLDER) if "aokvqa_350-qwen-7b-instruct" in f and f.endswith(".jsonl")]

# AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS = os.path.join(OUTPUTS_FOLDER, "aokvqa_350-llava-ov-qwen2-0.5b-iso-20250924-152059.jsonl")
AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS = [os.path.join(OUTPUTS_FOLDER, f) for f in os.listdir(OUTPUTS_FOLDER) if "aokvqa_350-llava-ov-qwen2-0.5b" in f and f.endswith(".jsonl")]
# AOKVQA_350_LLAVA_OV_QWEN2_0_5_RESPONSES = os.path.join(LLM_RESPONSES_FOLDER, "aokvqa_350-llava-ov-qwen2-0.5b-iso-20250924-152059-responses.jsonl")
AOKVQA_350_LLAVA_OV_QWEN2_0_5_RESPONSES = [os.path.join(LLM_RESPONSES_FOLDER, f) for f in os.listdir(LLM_RESPONSES_FOLDER) if "aokvqa_350-llava-ov-qwen2-0.5b" in f and f.endswith(".jsonl")]

# AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS = os.path.join(OUTPUTS_FOLDER, "aokvqa_350-llava-ov-qwen2-7b-iso-20250924-131820.jsonl")
AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS = [os.path.join(OUTPUTS_FOLDER, f) for f in os.listdir(OUTPUTS_FOLDER) if "aokvqa_350-llava-ov-qwen2-7b" in f and f.endswith(".jsonl")]
# AOKVQA_350_LLAVA_OV_QWEN2_7_RESPONSES = os.path.join(LLM_RESPONSES_FOLDER, "aokvqa_350-llava-ov-qwen2-7b-iso-20250924-131820-responses.jsonl")
AOKVQA_350_LLAVA_OV_QWEN2_7_RESPONSES = [os.path.join(LLM_RESPONSES_FOLDER, f) for f in os.listdir(LLM_RESPONSES_FOLDER) if "aokvqa_350-llava-ov-qwen2-7b" in f and f.endswith(".jsonl")]

# AOKVQA_350_PIXTRAL_12B_STATISTICS = os.path.join(OUTPUTS_FOLDER, "ao`kvqa_350-pixtral_12b-iso-20250930-122934.jsonl")
AOKVQA_350_PIXTRAL_12B_STATISTICS = [os.path.join(OUTPUTS_FOLDER, f) for f in os.listdir(OUTPUTS_FOLDER) if "aokvqa_350-pixtral_12b" in f and f.endswith(".jsonl")]
# AOKVQA_350_PIXTRAL_12B_RESPONSES = os.path.join(LLM_RESPONSES_FOLDER, "aokvqa_350-pixtral_12b-iso-20250930-122934-responses.jsonl")
AOKVQA_350_PIXTRAL_12B_RESPONSES = [os.path.join(LLM_RESPONSES_FOLDER, f) for f in os.listdir(LLM_RESPONSES_FOLDER) if "aokvqa_350-pixtral_12b" in f and f.endswith(".jsonl")]

map_mod_to_stats = {
    "qwen-2b-instruct": AOKVQA_350_QWEN_TWO_STATISTICS,
    "qwen-7b-instruct": AOKVQA_350_QWEN_SEVEN_STATISTICS,
    "llava-ov-qwen2-0.5b": AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS,
    "llava-ov-qwen2-7b": AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS,
}

map_mod_to_responses = {
    "qwen-2b-instruct": AOKVQA_350_QWEN_TWO_RESPONSES,
    "qwen-7b-instruct": AOKVQA_350_QWEN_SEVEN_RESPONSES,
    "llava-ov-qwen2-0.5b": AOKVQA_350_LLAVA_OV_QWEN2_0_5_RESPONSES,
    "llava-ov-qwen2-7b": AOKVQA_350_LLAVA_OV_QWEN2_7_RESPONSES,
}

# script run by call_script_run_static_img_resizing
# RUN_SCRIPT_PATH = os.path.join(PTH_TO_PROJECT_FOLDER, "scripts/run_static_workloads_img_vid_resizing.py")
# EXECUTABLE_PYTHON = os.path.join(PTH_TO_PROJECT_FOLDER, "env/bin/python")
RUN_SCRIPT_PATH = os.path.join(PTH_TO_PROJECT_FOLDER, "scripts/universal_static_workloads.py")
EXECUTABLE_PYTHON = os.path.join(PTH_TO_PROJECT_FOLDER, "env_v1_konpap/bin/python")

####################################################################################################################################
####################################################################################################################################
# LMUDATAPTH = "/home/ioannis.dalianis/LMUData"
LMUDATAPTH = "/srv/muse-lab/datasets/VLMEvalKitdata/LMUData"

MMBENCH_DEV_EN_original = os.path.join(LMUDATAPTH, "MMBench_DEV_EN.tsv")
MMBENCH_DEV_EN_img_folder = os.path.join(LMUDATAPTH, "images/MMBench/")

AOKVQA_original = os.path.join(LMUDATAPTH, "AOKVQA_original.tsv")
AOKVQA_original_img_folder = os.path.join(LMUDATAPTH, "images/AOKVQA/")

COCO_VAL_original = os.path.join(LMUDATAPTH, "COCO_VAL.tsv")
COCO_VAL_original_img_folder = os.path.join(LMUDATAPTH, "images/COCO/")

LLaVABench_original = os.path.join(LMUDATAPTH, "LLaVABench.tsv")
LLaVABench_original_img_folder = os.path.join(LMUDATAPTH, "images/LLaVABench/")

# ORIGINAL_TSV = COCO_VAL_original
# ORIGINAL_img_folder = COCO_VAL_original_img_folder

# ORIGINAL_TSV = MMBENCH_DEV_EN_original
# ORIGINAL_img_folder = MMBENCH_DEV_EN_img_folder

# ORIGINAL_TSV = AOKVQA_original
# ORIGINAL_img_folder = AOKVQA_original_img_folder

ORIGINAL_TSV = LLaVABench_original
ORIGINAL_img_folder = LLaVABench_original_img_folder
####################################################################################################################################
####################################################################################################################################

# this contains from the file with the 350 requests only those that have the most common sizes -> (640, 480)
AOKVQA_350_WORKLOAD_MOST_COMMON_SIZE = os.path.join(STATIC_WORKLOAD_FOLDER, "aokvqa_83.jsonl")
AOKVQA_350_STATISTICS_COMMON_SIZE = os.path.join(OUTPUTS_FOLDER, "aokvqa_83-image-mistral-iso-20250404-094306.jsonl")

RESAMPLING_FILTERS = { # Techniques for Resizing
    "LANCZOS": "lan",
    # "NEAREST": "ner",
    # "BILINEAR": "bil",
    # "BOX": "box",
    # "HAMMING": "ham",
    # "BICUBIC": "bic"
    }

FILTERS_OBJECTS = {
    "LANCZOS": Image.Resampling.LANCZOS,
    # "NEAREST": Image.Resampling.NEAREST,
    # "BILINEAR": Image.Resampling.BILINEAR,
    # "BOX": Image.Resampling.BOX,
    # "HAMMING": Image.Resampling.HAMMING,
    # "BICUBIC": Image.Resampling.BICUBIC
}

DIMENSION_RESIZE_TECHNIQUES = {
    "Thumbnail": "thu",
    # "Hard Coded": "hcd",
    # "Hard Coded Width": "hcw",
    # "Hard Coded Height": "hch",
    # "Average": "avg",
}
DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY = {
    "Both Dimensions Proportionally": "bdp",
    ## "Width Proportionally": "wdp",
    ## "Height Proportionally": "hdp",
    # "Thumbnail Proportionally": "thp",
    ## "Thumbnail Proportionally Width": "thw",
    ## "Thumbnail Proportionally Height": "thh",
}

DIMENSIONS_DICT = {
    "00": (100,100),
    "01": (200,200),
    "02": (300,300),
    # "03": (400,400),
    # "04": (500,500),
    # "05": (150,250),
    # "06": (250,150),
    # "07": (250,350),
    # "08": (350,250),
    # "09": (150,450),
    # "10": (450,150),
}

DIMENSIONS_DICT_PROPORTIONALITY = {
    "00": (5,5),
    "01": (10,10),
    "02": (20,20),
    "03": (30,30),
    "04": (40,40),
    "05": (50,50),
    "06": (60,60),
    "07": (70,70),
    "08": (80,80),
    "09": (90,90),
}

AVG_DIMENSION_ALIASES_PATH = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/util_files/avg_dimensions_dict.json"
# it gets created on the go - this is what we want to see about avg
with open(AVG_DIMENSION_ALIASES_PATH, "r") as f:
    AVG_DIMENSIONS_DICT = json.load(f)

COLORS = [
    # "gray",
    "rgb"
]

####################################################################################################################################
####################################################################################################################################
# files and paths and parameters only for video experimentation

OUTPUTS_350_FOLDER_LLAVA_VID = os.path.join(OUTPUTS_FOLDER, "folder_350_llava_vid_rgb")
LLAVA_VID_RGB_STATICS = os.path.join(STATIC_WORKLOAD_FOLDER, "llava_vid_rgb_comprs")
LLAVA_VID_FOLDER = "/srv/muse-lab/datasets/LLaVA-Video/videos"

LLAVA_VID_SRV_PTH = "/srv/muse-lab/datasets/LLaVA-Video"

MC_0_30_350_JSONL       = os.path.join(STATIC_WORKLOAD_FOLDER, "vid-mc-0-30_350.jsonl")
MC_0_30_10_JSONL       = os.path.join(STATIC_WORKLOAD_FOLDER, "vid-mc-0-30_10.jsonl")
# MC_0_30_350_STATISTICS  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-video-mistral-iso-20250526-153404.jsonl")
# MC_0_30_350_RESPONSES   = os.path.join(LLM_RESPONSES_FOLDER, "vid-mc-0-30_350-video-mistral-iso-20250526-153404-responses.jsonl")
MC_0_30_350_STATISTICS  = os.path.join(OUTPUTS_FOLDER, "back_up/vid-mc-0-30_350-video-mistral-iso-20250526-153404.jsonl") # backup
MC_0_30_350_RESPONSES   = os.path.join(LLM_RESPONSES_FOLDER, "back_up/vid-mc-0-30_350-video-mistral-iso-20250526-153404-responses.jsonl") # backup
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/outputs/llm_responses/back_up/vid-mc-0-30_350-video-mistral-iso-20250526-153404-responses.jsonl

# MC_0_30_350_STATISTICS_QWEN_TWO  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-qwen-2b-instruct-iso-20250930-190827.jsonl")
# MC_0_30_350_RESPONSES_QWEN_TWO   = os.path.join(LLM_RESPONSES_FOLDER, "vid-mc-0-30_350-qwen-2b-instruct-iso-20250930-190827-responses.jsonl")
MC_0_30_350_STATISTICS_QWEN_TWO  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-qwen-2b-instruct-iso-20251004-191539.jsonl")
MC_0_30_350_RESPONSES_QWEN_TWO   = os.path.join(LLM_RESPONSES_FOLDER, "vid-mc-0-30_350-qwen-2b-instruct-iso-20251004-191539-responses.jsonl")

MC_0_30_350_STATISTICS_QWEN_SEVEN  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-qwen-7b-instruct-iso-20251005-222838.jsonl")
MC_0_30_350_RESPONSES_QWEN_SEVEN   = os.path.join(LLM_RESPONSES_FOLDER, "vid-mc-0-30_350-qwen-7b-instruct-iso-20251005-222838-responses.jsonl")

MC_0_30_350_STATISTICS_LLAVA_OV_0_5  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-llava-ov-qwen2-0.5b-iso-20251007-154043.jsonl")
# MC_0_30_350_RESPONSES_LLAVA_OV_0_5   = os.path.join(LLM_RESPONSES_FOLDER, "vid-mc-0-30_350-qwen-7b-instruct-iso-20251005-222838-responses.jsonl")

MC_0_30_350_STATISTICS_LLAVA_OV_7  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-llava-ov-qwen2-7b-iso-20251007-163440.jsonl")

MC_0_30_350_STATISTICS_PIXTRAL  = os.path.join(OUTPUTS_FOLDER, "vid-mc-0-30_350-pixtral_12b-iso-20251008-011654.jsonl")

JSONL_FILE = MC_0_30_350_JSONL
# JSONL_FILE = MC_0_30_10_JSONL

# STATS_FILE = MC_0_30_350_STATISTICS
# RESPS_FILE = MC_0_30_350_RESPONSES
# MODEL_ALIAS = None

STATS_FILE = MC_0_30_350_STATISTICS_QWEN_TWO
RESPS_FILE = MC_0_30_350_RESPONSES_QWEN_TWO
MODEL_ALIAS = "qwen-2b-instruct"

RESAMPLING_FILTERS_VID = { # Techniques for Resizing
    "LANCZOS": "lan",
    }

FRAME_SAMPLING_TECHNIQUE = {
    "uniform": "uni",
    "fixed_rate": "frs",
    "scene_change": "scc",
    "motion_based": "mbd",
    "sharpness_based": "shb",
}

FRAME_SAMPLING_TECHNIQUE_NAME = {
    "uni": "Uniform",
    "frs": "Frame Sampling",
    "scc": "Scene Change",
    "mbd": "Motion Based",
    "shb": "Sharpness Based",
}

DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY_VID = {
    "Both Dimensions Proportionally": "bdp",
}

DIMENSIONS_DICT_PROPORTIONALITY_VID = {
    "00": (5,5),
    "01": (10,10),
    "02": (20,20),
    # "03": (30,30),
    # "04": (40,40),
    # "05": (50,50),
    # "06": (60,60),
    # "07": (70,70),
    # "08": (80,80),
    # "09": (90,90),
}

MAX_FRAMES = {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32,
    "4": 64,
}

# Test fixed-rate sampling (consistent temporal density)
FPS = {
    "0": 0.25,
    "1": 0.5,
    "2": 1.0,
    "3": 1.5,
    "4": 2.0,
}

# The dummy video has hard cuts at t=3s (red->blue) and t=6s (blue->green). ContentDetector should find these.
CONTENT_THRESHOLD = {
    # "0": 10.0,
    # "1": 20.0,
    "2": 27.0,
    # "3": 34.0,
    # "4": 44.0,
}

# Test motion-based sampling (good for action-heavy videos). The dummy video has motion from 6-10s. Default motion_threshold is 1.0.
MOTION_THRESHOLD = {
    # "0": 0.5,
    "1": 1.0,
    # "2": 2.0,
    # "4": 5.0,
    # "0": 0.1,
}

# Test sharpness-based sampling (good for prioritizing clear visual information). Sharpness threshold needs tuning. Text is sharper than plain color
SHARPNESS_THRESHOLD = {
    # "0": 50.0,
    "1": 100.0,
    # "2": 200.0,
    # "3": 500.0,
}
####################################################################################################################################
####################################################################################################################################

param_dictionary ={
    "title_size": 26.4,
    # "figsize_mul": 1.13,
    "figsize_mul": 2,
    "params_label_size": 21.5,
    "legend_font_size": 19,
    "label_size": 26,
    "plot_line_width": 2.72,
    "plot_marker_size": 11.5,
    "move_y_title_label": 0.42,
    "ymax_value": None,
    "ymin_value": 0.0,
    "mul_col_size": 6.45,
    "mul_row_size": 5.45,
    "x_ticks": [1, 2, 3, 4, 5],
    "y_ticks": None,
    "xmin_value": None,
    "axis_title_size": 28,
    "legend_size": 19,
    # "xlabel_size": 23,
    "xlabel_size": 29,
    # "ylabel_size": 23,
    "ylabel_size": 29,
    # "y_params_label_size": 20,
    # "x_params_label_size": 20,
    "y_params_label_size": 29.5,
    "x_params_label_size": 29.5,
    # "line_width": 2.3,
    "line_width": 4,
    "bar_width": 0.81
}

plot_colors_by_num = {
    0: "#e6194b",  # red
    1: "#3cb44b",  # green
    2: "#ffe119",  # yellow
    3: "#4363d8",  # blue
    4: "#f58231",  # orange
    5: "#911eb4",  # purple
    6: "#46f0f0",  # cyan
    7: "#f032e6",  # magenta
    8: "#bcf60c",  # lime
    9: "#fabebe",  # pink
    10: "#008080", # teal
    11: "#e6beff", # lavender
    12: "#9a6324", # brown
    13: "#fffac8", # light yellow
    14: "#800000", # maroon
    15: "#aaffc3", # mint
    16: "#808000", # olive
    17: "#ffd8b1", # apricot
    18: "#000075", # navy
    19: "#808080", # gray
    20: "#ffffff", # white
    21: "#000000", # black
}

# COLORS_ALL = get_colormap_colors(len(x_zip), cmap_name="Blues", vmin=0.2, vmax=0.7)
# COLORS_ALL = ["#E2B890", "#66f", "#E6E600", "#A9B3BD", "#E3A0FF", "#F7B1DD", "#94E594"]            
COLORS_ALL = ["#F2DFA8", "#E6AE65", "#C8685B", "#9a6324", "#923B62", "#69105E", "#1E256C", "#0E0721"]