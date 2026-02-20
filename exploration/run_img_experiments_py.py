from plot_utils import *
import itertools

for resampl_fil, dim_res_techn, dims, colors in list(itertools.product(
    list(RESAMPLING_FILTERS.keys()), list(DIMENSION_RESIZE_TECHNIQUES.keys()), list(DIMENSIONS_DICT.keys()), COLORS
    )):
    print(f"Filter: {resampl_fil}, dimension resize technique: {dim_res_techn}, dimensions: {dims}")

    create_resized_img_folder_n_jsonl(init_jsonl_request_pth=AOKVQA_350_WORKLOAD,
                                      resampl_fil=resampl_fil,
                                      dim_res_techn=dim_res_techn,
                                      dimens=dims,
                                      color=colors
                                      )
    print()
####################################################################################################################################
####################################################################################################################################
for resampl_fil, dim_res_techn, dims, colors in list(itertools.product(
    list(RESAMPLING_FILTERS.keys()), list(DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY.keys()), list(DIMENSIONS_DICT_PROPORTIONALITY.keys()), COLORS
    )):
    print(f"Filter: {resampl_fil}, dimension resize technique: {dim_res_techn}, dimensions: {dims}")

    create_resized_img_folder_n_jsonl(init_jsonl_request_pth=AOKVQA_350_WORKLOAD,
                                      resampl_fil=resampl_fil,
                                      dim_res_techn=dim_res_techn,
                                      dimens=dims,
                                      color=colors
                                      )
    print()
####################################################################################################################################
####################################################################################################################################
for resampl_fil, dim_res_techn, dims, colors in list(itertools.product(
    list(RESAMPLING_FILTERS.keys()), list(DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY.keys()), list(DIMENSIONS_DICT_PROPORTIONALITY.keys()), COLORS
    )):
    print(f"Filter: {resampl_fil}, dimension resize technique: {dim_res_techn}, dimensions: {dims}")

    # only creates compressed workloads for VLMEvalKit
    create_resized_img_folder_n_tsv_file(init_tsv_request_pth=ORIGINAL_TSV,
                                      resampl_fil=resampl_fil,
                                      dim_res_techn=dim_res_techn,
                                      dimens=dims,
                                      color=colors,
                                      created_img_folder_pth=ORIGINAL_img_folder,
                                      created_jsonl_pth=LMUDATAPTH
                                      )
    print()

# nohup
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env/bin/python
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_img_experiments_py.py
# >
# /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_imgs.log 2>&1
# < /dev/null &

# nohup /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/env/bin/python /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_img_experiments_py.py > /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/run_imgs.log 2>&1 < /dev/null &
