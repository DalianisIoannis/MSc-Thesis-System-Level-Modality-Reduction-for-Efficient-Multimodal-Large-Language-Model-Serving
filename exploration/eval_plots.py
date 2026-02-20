from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker # Imported for Y-axis formatting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as mtick
from typing import Callable, Dict, List
import re
from llmperf.postprocessing.output import RequestOutput
from llmperf.config.workloads import Request
import evaluate
import os
import ast
import json
from collections import defaultdict

model_rename_map = {
    'qwen-2': 'qwen',
    'qwen-2-small': 'qwen-small',
    'llava-ov': 'llava',
    'llava-ov-small': 'llava-small',
    # 'pixtral' is left as is
}
# Define the exact order you want the models and frame parameters to appear
PREFERRED_MODEL_ORDER = [
    'qwen-small', 'qwen', 'llava-small', 'llava', 'pixtral',
    # Add renamed versions if they are the ones present in the 'Model' column after renaming
    'Qwen-Small', 'Qwen-VL', 'Llava-Small', 'Llava', 'Pixtral' 
]
PREFERRED_FRAME_ORDER = [
    # 'frame4', 'frame8', 'frame16', 'frame32', 'frame64'
    'frame64', 'frame32', 'frame16', 'frame8', 'frame4'
]
frame_rename_map = {
    'frame4': '4 frames',
    'frame8': '8 frames',
    'frame16': '16 frames',
    'frame32': '32 frames',
    'frame64': '64 frames'
}
technique_rename_map = {
    'motion_based': 'motion',
    'scene_change': 'scene',
    'sharpness_based': 'sharpness',
}

param_dictionary ={
    "title_size": 26.4,
    "legend_font_size": 19,
    "y_params_label_size": 26,
    "x_params_label_size": 26,
    # "title_size": 35,
    # "legend_font_size": 35,
    # "y_params_label_size": 35,
    # "x_params_label_size": 35,

    "figsize_mul": 2,
    "params_label_size": 21.5,
    "label_size": 26,
    "mul_row_size": 5.45,
    "x_ticks": [1, 2, 3, 4, 5],
    "legend_size": 19,
    "xlabel_size": 29,
    "ylabel_size": 29,
    "line_width": 4,
    "bar_width": 0.81
}
LATENCY_COLORS = {
    'Preprocess': '#31511E',
    'Encoder': '#859F3D',
    'LLM': '#C9DD84',
}
colors_all = [
    "#F2DFA8",  # 0% - Very Light Tan (Original)
    "#E6AE65",  # 5% - Light Orange-Brown (Original)
    "#C8685B",  # 10% - Terracotta (Original)
    "#9a6324",  # 20% - Sienna Brown (Original)
    "#923B62",  # 30% - Deep Rose (Original)
    "#69105E",  # 40% - Maroon/Deep Plum (Original)
    "#381460",  # 50% - Vivid Dark Purple (NEW contrast)
    "#1E256C",  # 60% - Navy Blue (Original)
    "#103E5C",  # 70% - Dark Teal/Slate Blue (NEW contrast)
    "#05203A",  # 80% - Very Dark Teal/Midnight (NEW contrast)
    "#000A1A"   # 90% - Deepest Blue-Black (NEW contrast - Darkest)
]

def save_figure_as_pdf(fig, plot_name):
    """
    Saves a matplotlib figure as a pdf to a specified directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to be saved.
    plot_name : str
        The name of the plot to be saved.

    Notes
    -----
    If the directory does not exist, it will be created.
    If a file with the same name already exists, it will not be overwritten.
    """
    PLOTS_SAVE_PATH = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/figures"
    if not os.path.exists(PLOTS_SAVE_PATH):
        os.makedirs(PLOTS_SAVE_PATH)
    # if not os.path.exists(os.path.join(PLOTS_SAVE_PATH, plot_name+".pdf")):
    print(f"Saving plot to {os.path.join(PLOTS_SAVE_PATH, plot_name+'.pdf')}")
    fig.savefig(os.path.join(PLOTS_SAVE_PATH, plot_name+".pdf"), bbox_inches='tight')

def parse_logs(filepath: str):
    """
    Parse the experiment IDs from the benchmark logs.

    Args:
        filepath (str): The path to the benchmark logs file.

    Returns:
        experiment_ids (defaultdict): A dictionary of dictionaries.
            The outer dictionary is keyed by workload aliases.
            The inner dictionary is keyed by model aliases.
            The value is the experiment ID.
    """
    experiment_ids = defaultdict(lambda: defaultdict(dict))

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
                continue

            params = record["id"].split("__")
            workload_alias = params[0]
            model_alias = params[1]

            if len(params) == 10:
                cr = 0.0
            else:
                cr = float(params[10][2:])

            experiment_ids[workload_alias][model_alias][cr] = record["id"]

    return experiment_ids

def parse_logs_videos(filepath: str):
    """
    Parse the experiment IDs from the benchmark logs for videos.

    Args:
        filepath (str): The path to the benchmark logs file.

    Returns:
        experiment_ids (defaultdict): A dictionary of dictionaries.
            The outer dictionary is keyed by workload aliases.
            The inner dictionary is keyed by model aliases.
            The value is another dictionary keyed by the technique name and the frame number.
            The value of the inner dictionary is the experiment ID.
    """
    experiment_ids = defaultdict(lambda: defaultdict(dict))

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line}")
                continue

            params = record["id"].split("__")
            workload_alias = params[0]
            model_alias = params[1]

            cr = params[-2].split('strat')[1]

            frame_num = params[-1]

            # experiment_ids[workload_alias][model_alias][cr] = record["id"]
            try:
                experiment_ids[workload_alias][model_alias][cr][frame_num] = record["id"]
            except KeyError:
                experiment_ids[workload_alias][model_alias][cr] = {frame_num: record["id"]}
    return experiment_ids

def score_rougel_tempcompass_captioning(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [r.output for r in requests] # "A. solidifying"
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"] # or results["rougeLsum"]

def score_rougel_cocoval_captioning(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [ast.literal_eval(r.output) for r in requests] # "['A row of parked cars sitting next to parking meters.', 'A row of cars parked on a street with parking meters.', 'A series of parking meters and cars are located next to each other. ', 'A parking meter on a street by a car with traffic.', 'A parking meter on a street with cars']"
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    
    ########################################################
    # # trials
    # print("Rouge-L Results:", results)
    # # bleu = evaluate.load("sacrebleu")
    # # bleu_results = bleu.compute(predictions=preds, references=refs)
    # # print("BLEU Results:", bleu_results)
    # # from pycocoevalcap.cider.cider import Cider
    # # cider_scorer = Cider()
    # # score, scores = cider_scorer.compute_score(refs, preds)
    # # print("Cider Results:", score)
    ########################################################
    
    return results["rougeL"] # or results["rougeLsum"]  

def score_rougel_qna(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    rouge = evaluate.load("rouge")
    refs = [r.output for r in requests] # "A creative painting of a dog dressed as the famous Mona Lisa." or # "It's called \"We Are More\"."
    preds = [ro.outputs[0] for ro in request_outputs]
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"] # or results["rougeLsum"]

def score_acc_mc(requests: List[Request], request_outputs: List[RequestOutput]) -> float:
    """
    Compute accuracy for multiple-choice questions.

    Parameters
    ----------
    requests : List[Request]
        List of requests where each request has a ground-truth answer.
    request_outputs : List[RequestOutput]
        List of request outputs where each output contains the model's answer.

    Returns
    -------
    float
        The accuracy of the model, computed as the number of correct answers divided by the total number of requests.
    """
    correct_answers = 0
    for r, ro in zip(requests, request_outputs):
        gt = r.output # "A"
        answer = ro.outputs[0]
        matches = re.findall(r'\b([A-D])\b', answer, flags=re.IGNORECASE)
        answer = matches[0] if matches else ""

        if gt == answer:
            correct_answers += 1

    return correct_answers / len(requests)

_METRIC_FN_MAP: Dict[str, Dict[str, Callable[[List[Request], List[RequestOutput]], float]]] = {
    "accuracy": {
        "mmbench-mc": score_acc_mc,
        "videomme-mc": score_acc_mc,
    },
    "rouge-l": {
        "llavabench-qna": score_rougel_qna,
        "mmbench-video-qna": score_rougel_qna,
        "cocoval-captioning": score_rougel_cocoval_captioning,
        "tempcompass-captioning": score_rougel_tempcompass_captioning,
    }
}

def get_metric_fn(metric: str, workload_alias: str) -> Callable[[List[Request], List[RequestOutput]], float]:
    """
    Returns the metric function associated with the given metric and workload alias.

    Parameters
    ----------
    metric : str
        The name of the metric.
    workload_alias : str
        The alias of the workload.

    Returns
    -------
    Callable[[List[Request], List[RequestOutput]], float]
        The metric function.

    Raises
    -------
    ValueError
        If the metric is not valid for the given workload alias.
    """
    try:
        return _METRIC_FN_MAP[metric][workload_alias]
    except KeyError:
        raise ValueError(f"Metric '{metric}' is not valid for workload '{workload_alias}'")

def create_suffix_formatter(base_formatter_func, suffix=''):
    """Creates a FuncFormatter that applies a base formatter and then appends a suffix."""
    def suffix_formatter(x, pos):
        # 1. Apply the original formatting logic
        formatted_string = base_formatter_func(x, pos)
        
        # 2. Append the suffix
        return formatted_string + suffix
    
    return ticker.FuncFormatter(suffix_formatter)

def format_y_tick_decimal_aware(y, pos):
    """
    Custom Formatter for Decimal-Aware Y-Tick values:
    - If the value is an integer, returns the integer (e.g., 1.0 -> '1').
    - If rounding to ONE decimal place results in a loss of precision, 
      returns TWO decimal places (e.g., 0.15 -> '0.15').
    - Otherwise, returns ONE decimal place (e.g., 0.9 -> '0.9', 0.80 -> '0.8').
    """
    
    # 1. Integer Check: If the number is effectively an integer
    if abs(y - round(y)) < 1e-6:
        return f"{int(round(y))}"
    
    # 2. Precision Check: Determine if one decimal is enough
    
    # Round the value to one decimal place
    y_rounded_one = round(y, 1)
    
    # Check if rounding to one decimal loses significant precision 
    # (i.e., if the original value is substantially different from the one-decimal rounded value).
    # We use a threshold of 1e-3 (or 0.001) for this comparison.
    if abs(y - y_rounded_one) > 1e-3:
        # Example: y=0.15, y_rounded_one=0.2. Difference is ~0.05 ( > 1e-3). 
        # Needs two decimals for precision.
        return f"{y:.2f}"
    else:
        # Example: y=0.9, y_rounded_one=0.9. Difference is 0. 
        # Example: y=0.801, y_rounded_one=0.8. Difference is ~0.001 (not > 1e-3). 
        # One decimal is enough.
        return f"{y:.1f}"

def plot_grouped_bar_chart(data_variable, dataset_names, tit="Accuracy", show_legend=False, add_suffix=False):
    """
    Generates a stacked grouped bar chart for one or more specified datasets (tasks).
    Plots tasks in a vertical stack, sharing X-axis (Model) labels.

    Args:
        data_variable (defaultdict): The nested defaultdict containing the results.
        dataset_names (str or list): The key(s) for the dataset(s) (tasks) to plot.
        tit (str): The Y-axis title.
        show_legend (bool): Whether to show the legend.
        add_suffix (bool): Whether to add a suffix (e.g., 'x') to Y-tick labels.
    """
    
    # 1. Standardize dataset_names input to a list
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    N = len(dataset_names) # Number of rows (Tasks)
    
    df_list = []
    
    # 2. Extract and format data for all tasks
    for task_name in dataset_names:
        if task_name not in data_variable:
            print(f"Error: Dataset '{task_name}' not found in the data.")
            continue
            
        dataset_data = data_variable[task_name]
        
        # Convert dict to DataFrame: Model names become index, percentage values become columns.
        df_task = pd.DataFrame(dataset_data).T
        
        # Add Task name column
        df_task['Task'] = task_name
        
        # --- Model Renaming ---
        df_task.rename(index=model_rename_map, inplace=True)
        
        # --- Column Renaming (Percentage values) ---
        # Get column names (which are the raw percentage/value keys)
        original_cols = df_task.columns.drop('Task', errors='ignore') 
        
        col_rename_map = {}
        for col in original_cols:
            if isinstance(col, (int, float)):
                 # For numerical columns (like 0.0, 5.0, 10.0 for images)
                col_rename_map[col] = f"{int(col)}%"
            else:
                 # For non-numerical columns (like frame-pixel keys for videos)
                col_rename_map[col] = f"{col}%" # Keep original string and append %
                
        df_task.rename(columns=col_rename_map, inplace=True)
        
        df_list.append(df_task)
        
    if not df_list:
        print("No valid data found for plotting.")
        return
        
    df_combined = pd.concat(df_list)
    
    # Identify the columns representing the bars (should be the percentages)
    bar_columns = [col for col in df_combined.columns if col not in ['Task']]

    # --- 3. SUBPLOTS SETUP ---
    
    fig_width = param_dictionary["figsize_mul"] * 6.4
    fig_height = 4.8 * N * param_dictionary["figsize_mul"] * 0.74 # Scale height by N
    # fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    fig, axes = plt.subplots(
        N, 1,
        figsize=[fig_width, fig_height],
        constrained_layout=True,
        sharex=True,  # Share X-axis across all rows
        sharey=False  # Do NOT share Y-axis
    )
    
    if N == 1:
        axes = [axes]

    # --- 4. ITERATE AND PLOT ---
    
    y_label_text = tit

    for i, task_name in enumerate(dataset_names):
        ax = axes[i]
        
        # Filter the combined DataFrame for the current task
        df_plot = df_combined[df_combined['Task'] == task_name].drop(columns=['Task'])
        
        # Plotting: index (models) are groups, columns (percentages) are bars
        df_plot.plot(kind='bar', ax=ax, rot=0, color=colors_all)

        # --- Formatting and Labeling ---
        
        # 🎯 Plot Title (Task Name) 🎯
        ax.set_title(f"{task_name}", size=param_dictionary["xlabel_size"] * 1.0, pad=10)
        
        # Y-Axis Formatting
        if add_suffix:
            # ax.yaxis.set_major_formatter(formatter_func)
            custom_formatter = create_suffix_formatter(format_y_tick_decimal_aware, suffix="x")
            ax.yaxis.set_major_formatter(custom_formatter)
        else:
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter_func))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_tick_decimal_aware))
        ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
        
        # 🎯 Y-Axis Label: Only show on the first plot 🎯
        if i == 0:
            ax.set_ylabel(y_label_text, size=param_dictionary["ylabel_size"])
        else:
            ax.set_ylabel("") # Suppress Y-label for other plots
            # ax.set_ylabel("Rouge-L", size=param_dictionary["ylabel_size"])

        # X-Axis Ticks & Labels: Only show on the bottom plot
        if i == N - 1:
            ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"], rotation=40)
            ax.set_xlabel('Model', size=param_dictionary["xlabel_size"]) # Add X-label
        else:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("") # Suppress X-axis title for upper plots
            
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove individual legend
        if ax.get_legend():
            ax.get_legend().remove()

    # --- 5. GLOBAL LEGEND ---
    if show_legend and N > 0 and bar_columns:
        # Get labels and handles from the top-most plot
        legend_handles, legend_labels = axes[0].get_legend_handles_labels()

        if legend_handles and legend_labels:
            ncol = len(bar_columns) # Number of columns in the legend
            
            fig.legend(
                handles=legend_handles, 
                labels=legend_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.05), # Position above the titles
                # ncol=ncol,
                ncol=5,
                fontsize=param_dictionary["legend_font_size"],
                frameon=False
            )

    plt.show()
    return fig

def flatten_data_to_dataframe(data_variable, dataset_name):
    """Flattens the nested dictionary structure into a pandas DataFrame."""
    if dataset_name not in data_variable:
        raise ValueError(f"Error: Dataset '{dataset_name}' not found in the data.")
    
    records = []
    # Flatten the deeply nested dictionary into a list of records
    dataset_data = data_variable[dataset_name]

    for model_key, techniques in dataset_data.items():
        # Apply model renaming immediately
        model_name = model_rename_map.get(model_key, model_key)
        for technique, frames in techniques.items():
            for frame_param, score in frames.items():
                records.append({
                    'Model': model_name,
                    'Technique': technique,
                    'FrameParam': frame_param,
                    'Score': score
                })
                
    df = pd.DataFrame(records)
    return df

def plot_latency_breakdown(preprocess_data, encoder_data, ttft_data, dataset_name):
    """
    Generates a stacked bar chart using the '0.0' key from three component dictionaries.
    Handles plotting for one or multiple datasets (tasks).

    Args:
        preprocess_data (defaultdict): Latency data for 'Preprocess'.
        encoder_data (defaultdict): Latency data for 'Encoder'.
        ttft_data (defaultdict): Latency data for 'LLM'.
        dataset_name (str or list): The specific dataset key(s) to plot.
    """
    
    # 1. Standardize dataset_name input to a list
    if isinstance(dataset_name, str):
        dataset_names = [dataset_name]
    else:
        dataset_names = dataset_name
        
    N = len(dataset_names)
    component_order = ['Preprocess', 'Encoder', 'LLM']
    colors = [LATENCY_COLORS[comp] for comp in component_order]

    # --- 2. Create Figure and Subplots (N rows, 1 column) ---
    
    # Scale height by N, keep width constant (or slightly adjust)
    fig_width = param_dictionary["figsize_mul"] * 6.4
    fig_height = param_dictionary["figsize_mul"] * 4.8 * N * 0.75 
    
    fig, axes = plt.subplots(
        N, 1,
        figsize=[fig_width, fig_height],
        constrained_layout=True,
        sharex=True,  # Share X-axis across all rows
        sharey=False  # Do NOT share Y-axis
    )
    
    # Ensure axes is iterable even if N=1
    if N == 1:
        axes = np.array([axes])
        
    # --- 3. Process Data and Plot Each Task ---
    
    # Find the global maximum latency across all tasks for consistent Y-limits
    global_max_latency = 0
    
    for i, task_name in enumerate(dataset_names):
        ax = axes[i]
        
        if task_name not in preprocess_data:
            print(f"Error: Dataset '{task_name}' not found. Skipping.")
            ax.set_visible(False)
            continue
            
        # Get the list of models for the specified dataset
        models = list(preprocess_data[task_name].keys())
        
        # Build the dictionary for the plotting DataFrame
        plot_data = {}
        for model in models:
            # Extract the value ONLY at the 0.0 key for each component
            prep_time = preprocess_data[task_name].get(model, {}).get(0.0, 0)
            enc_time = encoder_data[task_name].get(model, {}).get(0.0, 0)
            llm_time = ttft_data[task_name].get(model, {}).get(0.0, 0)
            
            # Structure the data for the stacked bar plot
            plot_data[model] = {
                'Preprocess': prep_time,
                'Encoder': enc_time,
                'LLM': llm_time
            }
        
        df_latency = pd.DataFrame.from_dict(plot_data, orient='index')
        df_latency.rename(index=model_rename_map, inplace=True)

        # Ensure order and find max latency
        df_latency = df_latency[component_order]
        max_latency = df_latency.sum(axis=1).max()
        if max_latency > global_max_latency:
            global_max_latency = max_latency

        # Create the stacked bar plot
        df_latency.plot(
            kind='bar', 
            stacked=True, 
            ax=ax, 
            color=colors,
            rot=45,
            legend=False # Suppress individual subplot legends
        )
        
        # --- Formatting and Labeling ---
        
        # 🎯 Set Task Title (Above each plot) 🎯
        ax.set_title(f"{task_name}", size=param_dictionary["title_size"] * 0.9, pad=10)
        # ax.set_title("Latency Breakdown for Image", size=param_dictionary["title_size"])
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
        ax.set_ylabel('Latency (seconds)', size=param_dictionary["ylabel_size"])
        
        # 🎯 X-Axis Labeling: Only show on the bottom plot 🎯
        if i == N - 1:
            ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"], rotation=45)
            # ax.set_xlabel('Model', size=param_dictionary["xlabel_size"]) # Add global X-label
        else:
            # Hide X-tick labels for all but the bottom plot
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("") # Suppress X-axis title for upper plots

    # --- 4. Final Adjustments (Y-limits and Legend) ---
    
    # # Set uniform Y-limit for all plots based on global maximum
    # y_limit = global_max_latency * 1.1 
    # for ax in axes.flat:
    #     if ax.get_visible():
    #          ax.set_ylim(0, y_limit)
        
    # Get handles/labels from the last processed plot (df_latency from i=N-1)
    # Use fig.legend for a single legend above the entire figure
    if N > 0 and axes[0].get_visible():
        handles, labels = axes[0].get_legend_handles_labels() # Use the top plot to get legend info
        
        fig.legend(
            handles, 
            labels,
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.02), # Position above the titles
            ncol=len(component_order),
            fontsize=param_dictionary["legend_font_size"],
            frameon=False,
        )

    plt.show()
    return fig

def plot_latency_breakdown_videos(preprocess_data, encoder_data, ttft_data, dataset_name):
    """
    Generates a stacked bar chart showing latency breakdown for video tasks.
    The function extracts data specifically at the 'uniform' -> 'frame64' path.
    Handles plotting for one or multiple datasets (tasks) in a vertical stack.

    Args:
        preprocess_data (defaultdict): Latency data for 'Preprocess'.
        encoder_data (defaultdict): Latency data for 'Encoder'.
        ttft_data (defaultdict): Latency data for 'LLM'.
        dataset_name (str or list): The specific dataset key(s) to plot.
    """
    
    # 1. Standardize dataset_name input to a list
    if isinstance(dataset_name, str):
        dataset_names = [dataset_name]
    else:
        dataset_names = dataset_name
        
    N = len(dataset_names) # Number of rows (Tasks)
    component_order = ['Preprocess', 'Encoder', 'LLM']
    colors = [LATENCY_COLORS[comp] for comp in component_order]

    # --- 2. Create Figure and Subplots (N rows, 1 column) ---
    
    # Scale height by N, adjust factor for better spacing
    fig_width = param_dictionary["figsize_mul"] * 6.4
    fig_height = param_dictionary["figsize_mul"] * 4.8 * N * 0.75
    
    fig, axes = plt.subplots(
        N, 1,
        figsize=[fig_width, fig_height],
        constrained_layout=True,
        sharex=True,  # Share X-axis across all rows
        sharey=False  # Do NOT share Y-axis
    )
    
    # Ensure axes is iterable even if N=1
    if N == 1:
        axes = np.array([axes])

    # --- 3. Process Data and Plot Each Task ---
    
    # Find the global maximum latency across all tasks for consistent Y-limits
    global_max_latency = 0
    
    for i, task_name in enumerate(dataset_names):
        ax = axes[i]
        
        if task_name not in preprocess_data:
            print(f"Error: Dataset '{task_name}' not found. Skipping.")
            ax.set_visible(False)
            continue
            
        # Get the list of models for the specified dataset
        models = list(preprocess_data[task_name].keys())
        
        # Build the dictionary for the plotting DataFrame
        plot_data = {}
        for model in models:
            try:
                # 1. Extract the value ONLY at the fixed path: uniform -> frame64
                prep_time = preprocess_data[task_name][model]["uniform"]["frame64"]
                enc_time = encoder_data[task_name][model]["uniform"]["frame64"]
                llm_time = ttft_data[task_name][model]["uniform"]["frame64"]
            except KeyError:
                print(f"Warning: Missing data path for {task_name} -> {model} at [uniform][frame64]. Skipping model.")
                continue

            # 2. Structure the data for the stacked bar plot
            plot_data[model] = {
                'Preprocess': prep_time,
                'Encoder': enc_time,
                'LLM': llm_time
            }
        
        # If no models were successfully processed, skip the subplot
        if not plot_data:
            ax.set_visible(False)
            continue
            
        df_latency = pd.DataFrame.from_dict(plot_data, orient='index')
        df_latency.rename(index=model_rename_map, inplace=True)

        # Ensure order and find max latency
        df_latency = df_latency[component_order]
        max_latency = df_latency.sum(axis=1).max()
        if max_latency > global_max_latency:
            global_max_latency = max_latency

        # Create the stacked bar plot
        df_latency.plot(
            kind='bar', 
            stacked=True, 
            ax=ax, 
            color=colors,
            rot=45,
            legend=False # Suppress individual subplot legends
        )
        
        # --- Formatting and Labeling ---
        
        # 🎯 Set Task Title (Above each plot) 🎯
        ax.set_title(f"{task_name}", size=param_dictionary["title_size"] * 0.9, pad=10)
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
        ax.set_ylabel('Latency (seconds)', size=param_dictionary["ylabel_size"])
        
        # 🎯 X-Axis Labeling: Only show on the bottom plot 🎯
        if i == N - 1:
            ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"], rotation=45)
            # ax.set_xlabel('Model', size=param_dictionary["xlabel_size"]) # Add global X-label
        else:
            # Hide X-tick labels for all but the bottom plot
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("") # Suppress X-axis title for upper plots

    # --- 4. Final Adjustments (Y-limits and Legend) ---
    
    # # Set uniform Y-limit for all visible plots based on global maximum
    # y_limit = global_max_latency * 1.1 
    # for ax in axes.flat:
    #     if ax.get_visible():
    #          ax.set_ylim(0, y_limit)
        
    # Get handles/labels from the first visible plot
    legend_handles = []
    legend_labels = []
    for ax in axes.flat:
        if ax.get_visible():
            h, l = ax.get_legend_handles_labels()
            if h and l:
                legend_handles = h
                legend_labels = l
                break
        
    # Use fig.legend for a single legend above the entire figure
    if legend_handles:
        fig.legend(
            legend_handles, 
            legend_labels,
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.03), # Position above the titles
            ncol=len(component_order),
            fontsize=param_dictionary["legend_font_size"],
            frameon=False,
            handler_map={tuple: HandlerTuple(ndivide=None)} # Use handler_map if necessary
        )
        # Add a title for the entire figure to maintain context
        # fig.suptitle("Latency Breakdown for Video", size=param_dictionary["title_size"] * 1.1, y=1.0)


    plt.show()
    return fig

def rename_value(col_name, value):
    if col_name == "FrameParam":
        return frame_rename_map.get(value, value)
    if col_name == "Technique":
        return technique_rename_map.get(value, value)
    return value  # Model or unknown, keep original

PLOT_MODES = { # Defines how to pivot based on the mode
    "model-technique|frame": ("Model", "Technique", "FrameParam"),
    "model-frame|technique": ("Model", "FrameParam", "Technique"),
    "technique-model|frame": ("Technique", "Model", "FrameParam"),
    "technique-frame|model": ("Technique", "FrameParam", "Model"),
    "frame-model|technique": ("FrameParam", "Model", "Technique"),
    "frame-technique|model": ("FrameParam", "Technique", "Model"),
}

def plot_grouped_video_bars(data, dataset_name, mode="model-technique|frame", title="Accuracy", show_legend=False, log_threshold=100, color_map="continuous",
                            add_plot_title=True, add_suffix=False):
    """
    Universal grouped bar chart generator for video experiments.
    Plots one or more tasks (dataset_name list) in a single vertical column.
    """
    
    # 1. Standardize dataset_name input to a list
    if not isinstance(dataset_name, list):
        dataset_names = [dataset_name]
    else:
        dataset_names = dataset_name
        
    all_dfs = []
    
    # --- Data Collection and Flattening (Per Task) ---
    
    for task_name in dataset_names:
        try:
            df_task = flatten_data_to_dataframe(data, task_name)
            if not df_task.empty:
                df_task['Task'] = task_name # Add task identifier
                all_dfs.append(df_task)
            else:
                print(f"No data found for task: {task_name}. Skipping.")
        except Exception as e:
            print(f"Error processing task {task_name}: {e}. Skipping.")

    if not all_dfs:
        print("No valid data found for plotting.")
        return
        
    df_base = pd.concat(all_dfs, ignore_index=True)

    if mode not in PLOT_MODES:
        print(f"Invalid mode: {mode}. Choose from:\n{list(PLOT_MODES.keys())}")
        return
    
    x_key, group_key, subplot_key = PLOT_MODES[mode]

    # --- Ordering (Applied to combined DataFrame) ---
    ordering_map = {
        "Model": PREFERRED_MODEL_ORDER,
        "FrameParam": PREFERRED_FRAME_ORDER,
        "Technique": sorted(df_base["Technique"].unique()),
    }
    for col, order in ordering_map.items():
        if col in df_base.columns:
            valid = [x for x in order if x in df_base[col].unique()]
            df_base[col] = pd.Categorical(df_base[col], categories=valid, ordered=True)

    # --- Subplot Setup ---
    
    subplot_values_per_task = df_base[subplot_key].unique()
    # Filter subplot values based on preferred order
    subplot_values_per_task = [v for v in ordering_map.get(subplot_key, subplot_values_per_task) if v in subplot_values_per_task]
    
    N_rows_per_task = len(subplot_values_per_task)
    N_tasks = len(dataset_names)
    
    Total_Rows = N_tasks * N_rows_per_task

    fig_width = param_dictionary["figsize_mul"] * 6.4
    fig_height = 4 * Total_Rows * param_dictionary["figsize_mul"] * 0.9 # Scale height by total rows
    
    # Create one long column of subplots
    fig, axes = plt.subplots(
        Total_Rows, 1,
        figsize=[fig_width, fig_height],
        constrained_layout=False, # Manual layout for more control
        sharex=True,  # All plots share the model names (X-axis)
        sharey=False  # Y-axis scales should be independent
    )
    
    # Ensure axes is a flat array
    axes = np.ravel(axes)

    # Manual subplot adjustments (Tighter spacing, top space for legend/titles)
    plt.subplots_adjust(
        hspace=0.25, # Space between subplots (rows)
        top=0.9,     # Space for legend/global title
        bottom=0.15,
        left=0.1,
        right=0.98
    )

    # --- 4. Nested Iteration and Plotting ---
    
    global_index = 0
    # The X-axis title only goes on the very last subplot
    
    # Prepare the formatter function
    formatter_func = format_y_tick_decimal_aware
    if add_suffix:
        formatter_func = create_suffix_formatter(format_y_tick_decimal_aware, suffix="x")
        
    legend_set = False # Flag to ensure legend is only handled once

    for k, task_name in enumerate(dataset_names): # Task loop (Vertical Stacking)
        for i, row_value in enumerate(subplot_values_per_task): # Subplot loop (Frames/Sub-dimension)
            
            ax = axes[global_index]
            
            # Filter to the specific task and subplot row
            df_plot = df_base[
                (df_base['Task'] == task_name) &
                (df_base[subplot_key] == row_value)
            ].pivot(index=x_key, columns=group_key, values="Score")

            # Apply renaming
            df_plot.index = [rename_value(x_key, idx) for idx in df_plot.index]
            df_plot.columns = [rename_value(group_key, col) for col in df_plot.columns]

            # Plotting
            plot_kwargs = {'kind': 'bar', 'ax': ax, 'rot': 0, 'width': 0.82}
            plot_kwargs['color'] = colors_all if color_map != "categorical" else None
            df_plot.plot(**plot_kwargs)

            # --- Log Scaling Check ---
            # (Keeping your original log scale check structure but simplified slightly)
            values = [v for v in df_plot.values.flatten() if v is not None and not pd.isna(v) and v > 0]
            
            needs_log = len(values) > 0 and max(values) / min(values) > log_threshold
            
            if needs_log:
                ax.set_yscale("log")

            # --- Labeling and Formatting ---
            
            # 🎯 Task Title (Above the first subplot of this task group) 🎯
            if i == 0:
                ax.set_title(f"{task_name}", size=param_dictionary["xlabel_size"] * 1.1, pad=10)
            
            # 🎯 Subplot Row Title (Frame Label) 🎯 (Above the plot, left-aligned)
            if add_plot_title:
                pretty_value = rename_value(subplot_key, row_value)
                ax.text(
                    -0.01, 1.05, f"{pretty_value}", # X, Y in axes coordinates
                    transform=ax.transAxes, 
                    ha='right', va='bottom', 
                    size=param_dictionary["xlabel_size"] * 0.9,
                    weight='bold'
                )

            # if i == 0:
            #     if k == 0:
            #         y_label_text = title + (" (log scale)" if needs_log else "")
            #         ax.set_ylabel(y_label_text, size=param_dictionary["ylabel_size"])
            #     else:
            #         ax.set_ylabel("Rouge-L", size=param_dictionary["ylabel_size"])
            # else:
            #     ax.set_ylabel("")
            # 🎯 Y-Axis Label 🎯 (Only on the first subplot of the task group)
            if i == N_rows_per_task // 2: # Place Y-label roughly in the middle plot of the task group
                y_label_text = title + (" (log scale)" if needs_log else "")
                ax.set_ylabel(y_label_text, size=param_dictionary["ylabel_size"])
            else:
                ax.set_ylabel("")
            
            # X-Axis Ticks & Labels: Only show on the very last plot
            if global_index == Total_Rows - 1:
                ax.tick_params(axis="x", labelsize=30, rotation=40)
                # ax.set_xlabel(rename_value("axis", x_key), size=param_dictionary["xlabel_size"])
            else:
                ax.tick_params(axis="x", labelbottom=False)
                ax.set_xlabel("")
            
            # Apply formatter and grid
            ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter_func))
            ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # --- Legend Management (Global Legend) ---
            if not legend_set and show_legend:
                # Use the first plot that has data to draw the legend
                if ax.get_legend():
                    legend_handles, legend_labels = ax.get_legend_handles_labels()
                    length = len(legend_labels)
                    ncol = length if length <= 4 else (length // 2 + length % 2)
                    
                    fig.legend(
                        handles=legend_handles,
                        labels=legend_labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.98), 
                        ncol=ncol,
                        fontsize=22,
                        frameon=False,
                    )
                    legend_set = True
                
            if ax.get_legend():
                ax.get_legend().remove()
            
            global_index += 1

    plt.show()
    return fig

def plot_grouped_video_bars_multi_data(data, dataset_names, mode="model-technique|frame", title="Accuracy", show_legend=False, log_threshold=100, color_map="continuous",
                            add_plot_title=True, add_suffix=False):
    """
    Universal grouped bar chart generator for video experiments.
    Plots data for multiple tasks (dataset_names) in columns.
    mode format: "Xaxis-GroupBy|Subplot"
    """
    
    # Ensure dataset_names is a list if a single task name was passed
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names]
        
    M = len(dataset_names) # Number of columns (Tasks)
    
    # Flatten data for all tasks and combine into a single DataFrame
    df_list = []
    for task_name in dataset_names:
        try:
            df_task = flatten_data_to_dataframe(data, task_name)
            if not df_task.empty: # Only add if data was found
                df_task['Task'] = task_name # Add a 'Task' column for grouping
                df_list.append(df_task)
            else:
                print(f"No data found for task: {task_name}")
        except Exception as e: # Catch any error during flattening
            print(f"Error processing task {task_name}: {e}")
            
    if not df_list:
        print("No valid data found for plotting.")
        return fig
        
    df_base = pd.concat(df_list, ignore_index=True)


    if mode not in PLOT_MODES:
        print(f"Invalid mode: {mode}. Choose from:\n{list(PLOT_MODES.keys())}")
        return
    
    x_key, group_key, subplot_key = PLOT_MODES[mode]

    # --- 2. ORDERING ---
    
    ordering_map = {
        "Model": PREFERRED_MODEL_ORDER,
        "FrameParam": PREFERRED_FRAME_ORDER,
        "Technique": sorted(df_base["Technique"].unique()),
        "Task": dataset_names
    }

    # Apply ordering (using df_base)
    for col, order in ordering_map.items():
        if col in df_base.columns:
            valid = [x for x in order if x in df_base[col].unique()]
            df_base[col] = pd.Categorical(df_base[col], categories=valid, ordered=True)

    # --- 3. SUBPLOTS SETUP ---
    
    subplot_values = df_base[subplot_key].unique()
    subplot_values = [v for v in ordering_map.get(subplot_key, subplot_values) if v in subplot_values]
    N = len(subplot_values) # Number of rows (Subplot Key values, e.g., 64 frames, 32 frames)

    if N == 0:
        print("No data available for the given mode.")
        return

    # Adjust figsize for overall plot. Make it taller if N is large.
    # Adjust width for M columns. Reduced figsize_mul to make default size smaller.
    fig_width = param_dictionary["figsize_mul"] * 5.4 * M * 0.9 # Reduce default width factor
    fig_height = 5 * N * param_dictionary["figsize_mul"] # Keep height scaling with N
    
    # Use subplot_kw to set common properties like sharex/sharey
    fig, axes = plt.subplots(
        N, M,
        figsize=[fig_width, fig_height],
        constrained_layout=False, # Disable constrained_layout to allow manual adjustments
        sharex='col', # Share X-axis only within columns
        sharey=False,
        # sharey='row'  # Share Y-axis only within rows
    )
    
    # Ensure axes is always a 2D array for consistent indexing
    if N == 1 and M == 1:
        axes = np.array([[axes]])
    elif N == 1: # One row, multiple columns
        axes = np.array([axes])
    elif M == 1: # Multiple rows, one column
        axes = np.array([axes]).T # Transpose to make it N rows, 1 col

    # Adjust spacing between subplots
    plt.subplots_adjust(
        # wspace=0.2,  # Horizontal space between plots (columns)
        wspace=0.25,  # Horizontal space between plots (columns)
        hspace=0.25, # Vertical space between plots (rows)
        top=0.85,    # Space for main title and legend
        bottom=0.15, # Space for X-axis labels and global X-label
        left=0.08,   # Space for Y-axis label
        right=0.98   # Right margin
    )

    # --- 4. ITERATE AND PLOT ---
    
    # Use a set to track which column has already applied log scale
    log_scale_applied_cols = set() 
    
    for i, row_value in enumerate(subplot_values): # Iterate over rows (e.g., FrameParams)
        for j, task_name in enumerate(dataset_names): # Iterate over columns (Tasks)
            
            ax = axes[i, j]
            
            # Filter by Subplot Key (Row) AND by Task (Column)
            df_slice = df_base[
                (df_base[subplot_key] == row_value) & 
                (df_base['Task'] == task_name)
            ]
            
            # If df_plot is empty, plot an empty frame or handle as needed
            if df_slice.empty:
                # Optionally, you can set an empty title or clear the axis if no data
                ax.set_visible(False) # Hide empty subplots
                continue

            df_plot = df_slice.pivot(index=x_key, columns=group_key, values="Score")
            
            # If df_plot is empty after pivot (e.g., no matching x_key for this task/frame)
            if df_plot.empty:
                ax.set_visible(False)
                continue

            # Apply renaming to column labels (legend) and x-axis labels
            df_plot.index = [rename_value(x_key, idx) for idx in df_plot.index]
            df_plot.columns = [rename_value(group_key, col) for col in df_plot.columns]

            # Plotting
            if color_map == "categorical":
                df_plot.plot(kind="bar", ax=ax, rot=0, width=0.8)
            else:
                # df_plot.plot(kind="bar", ax=ax, rot=0, color=colors_all, width=0.82)
                df_plot.plot(kind="bar", ax=ax, rot=0, color=colors_all, width=0.8)

            # --- LOG SCALING (Applied per COLUMN for consistency) ---
            values = df_plot.values.flatten()
            values = [v for v in values if v is not None and not pd.isna(v) and v > 0] # Filter out None/NaN/non-positive
            
            # Determine if log scale is needed for this column based on values in the first row
            if i == 0 and len(values) > 0 and max(values) / min(values) > log_threshold:
                log_scale_applied_cols.add(j)
            
            # Apply log scale if it was determined for this column (j)
            if j in log_scale_applied_cols:
                 ax.set_yscale("log")
                 
            # --- TITLES AND LABELS ---

            # 🎯 Add Column Title (Task Name) 🎯
            if i == 0: # Only on the top row
                ax.set_title(f"{task_name}", size=40 * 1.0, pad=10) # Adjust pad

            # 🎯 Add Row Title (Frame Label) - above the first column plot 🎯
            if j == 0: # Only on the first column
                pretty_value = rename_value(subplot_key, row_value)
                ax.text(
                    -0.01, 1.05, f"{pretty_value}", # X, Y in axes coordinates. Y=1.05 puts it above the plot.
                    transform=ax.transAxes, 
                    ha='right', va='bottom', # Align to top-right of the text block
                    size=40 * 0.9,
                    weight='bold'
                )
                
            # 🎯 Y-Axis Label ('Accuracy' or 'Sampling + Inference(s)') 🎯
            if j == 0: # Only for the first column of plots
                y_label_text = title
                # Check if this column needs a log scale suffix
                if j in log_scale_applied_cols:
                    y_label_text += " (log scale)"
                ax.set_ylabel(y_label_text, size=40)
            else:
                ax.set_ylabel("") # Suppress Y-label for other columns


            # --- X-Axis Ticks & Labels ---
            if i == N - 1: # Only on the bottom row
                ax.tick_params(axis="x", labelsize=40, rotation=42)
                ax.set_xlabel("") # Suppress subplot-level x-label as we'll add a global one if needed
            else: # Hide x-labels for all but the bottom row
                ax.tick_params(axis='x', labelbottom=False)

            # --- Grid and Tick Formatting ---
            ax.tick_params(axis="y", labelsize=40)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            
            # Apply custom formatter (log scale or not, with suffix)
            formatter_func = format_y_tick_decimal_aware
            if add_suffix:
                formatter_func = create_suffix_formatter(format_y_tick_decimal_aware, suffix="x")
                
            ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter_func))

            # --- Remove individual subplot legends ---
            if ax.get_legend():
                ax.get_legend().remove()
                
    # --- GLOBAL LEGEND ---
    if show_legend:
        # Get labels and handles from one of the plots (e.g., top-left)
        # Ensure df_plot is not empty for this (can happen if first plot is invisible)
        legend_labels = []
        legend_handles = []
        for col_idx in range(M):
            if not axes[0, col_idx].get_visible(): # Skip if subplot is hidden
                continue
            # Try to get handles/labels from the first visible plot in the top row
            h, l = axes[0, col_idx].get_legend_handles_labels()
            if h and l:
                legend_handles = h
                legend_labels = l
                break # Found a source for legend

        if legend_handles and legend_labels:
            # Determine appropriate ncol for the legend
            length = len(legend_labels)
            ncol = length if length <= 4 else (length // 2 + length % 2)
            
            fig.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper center", # Position the legend at the top center of the figure
                # bbox_to_anchor=(0.5, 0.98), # Adjust Y to be higher than column titles
                bbox_to_anchor=(0.5, 0.90), # Adjust Y to be higher than column titles
                ncol=ncol,
                fontsize=40,
                frameon=False,
                borderaxespad=0. # Remove padding from axes
            )

    # --- GLOBAL X-AXIS LABEL (Optional) ---
    # Add a global X-axis label if desired (uncomment if needed)
    # fig.supxlabel(rename_value("axis", x_key), size=param_dictionary["xlabel_size"], y=0.03)
    plt.show()
    return fig

def filter_nested_data(data, remove_models=None, remove_frames=None, remove_techniques=None, remove_tasks=None):
    """
    Creates a new nested dictionary containing data, excluding specified models,
    techniques, or frames.

    :param data: The original nested dictionary (tasks -> models -> techniques -> frames).
    :param remove_models: A set or list of model keys to remove (e.g., {'qwen'}).
    :param remove_frames: A set or list of frame keys to remove (e.g., {'frame4'}).
    :param remove_techniques: A set or list of technique keys to remove (e.g., {'uniform'}).
    :return: A new dictionary with the specified data points removed.
    """
    # Use sets for O(1) average time complexity lookups
    models_to_remove = set(remove_models or [])
    tasks_to_remove = set(remove_tasks or [])
    frames_to_remove = set(remove_frames or [])
    techniques_to_remove = set(remove_techniques or [])
    
    # Initialize the new dictionary structure
    new_data = defaultdict(lambda: defaultdict(dict))

    # Task Level (Level 1)
    for task_key, models_data in data.items():
        if task_key in tasks_to_remove:
            continue
        
        # Model Level (Level 2)
        for model_key, techniques_data in models_data.items():
            
            # Skip the entire model if it's on the removal list
            if model_key in models_to_remove:
                continue
            
            new_techniques_data = {}
            
            # Technique Level (Level 3)
            for tech_key, frames_data in techniques_data.items():
                
                # Skip the entire technique if it's on the removal list
                if tech_key in techniques_to_remove:
                    continue
                
                new_frames_data = {}
                
                # Frame Level (Level 4)
                # if we have video, there are frames, otherwise, it's images
                if type(frames_data) == float or isinstance(frames_data, np.float64):
                    new_frames_data = frames_data
                    new_techniques_data[tech_key] = new_frames_data
                    continue
                else:
                    for frame_key, score in frames_data.items():
                        
                        # Skip the frame if it's on the removal list
                        if frame_key in frames_to_remove:
                            continue
                        
                        # Keep the frame and its score
                        new_frames_data[frame_key] = score
                
                # Only add the technique if it still has frames after filtering
                if new_frames_data:
                    new_techniques_data[tech_key] = new_frames_data

            # Only add the model if it still has techniques after filtering
            if new_techniques_data:
                # We can't use the defaultdict structure directly here,
                # as the inner dicts might be empty, but it works for the new_data.
                # Use standard dicts for clarity.
                new_data[task_key][model_key] = new_techniques_data

    return dict(new_data) # Convert the outermost defaultdict back to a standard dict for cleaner printing/usage

def calculate_percentage_change(previous, current):
    """Calculates the percentage change from previous to current value."""
    if previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100

# Use a helper function to determine the numerical value of the key.
def get_numerical_key(k):
    """Attempts to convert the key to a float. Falls back to string sorting if conversion fails."""
    try:
        # Try to convert the key directly to a float (works for '5.0' or '4')
        return float(k)
    except ValueError:
        # If it's a non-numeric string (like 'frame4'), extract the number part.
        # This retains compatibility with the old keys if they still exist.
        num_part = ''.join(filter(str.isdigit, k))
        if num_part:
            return float(num_part)
        # If no number, return the string itself for alphabetical sorting
        return k

def analyze_nested_data_change(data_structure, direction='small_to_big', comparison_mode='previous', show_keys=None):
    """
    Traverses a nested dictionary structure and prints the value and 
    percentage change based on specified direction and comparison mode.

    :param data_structure: The nested data (tasks -> models -> techniques -> frames).
    :param direction: 'small_to_big' (frame4 -> frame64) or 'big_to_small' (frame64 -> frame4).
    :param comparison_mode: 'previous' (compare to the adjacent frame) or 'first' (compare to the start frame).
    :param show_keys: Optional set of specific keys (frames) to display. If None, shows all.
    """
    
    def recursive_analysis(current_data, level_keys):
        """
        Recursively walks the dictionary to find the deepest numerical data.
        """
        # Base Case: Check if the current data is the deepest level (Frames/Pixels)
        if current_data and isinstance(list(current_data.values())[0], (int, float)):
            
            print(f"\n--- Analysis for: {' -> '.join(level_keys)} ---")
            
            # --- 1. Determine Iteration Order ---
            
            # Sort keys numerically (e.g., 'frame4' -> 4, 'frame8' -> 8)
            try:
                # Sort keys using the robust helper function
                sorted_keys = sorted(current_data.keys(), key=get_numerical_key)
            except TypeError:
                # This catches a rare case where the mixed sorting fails (e.g., comparing string to float)
                # Fallback to simple alphabetical sorting
                sorted_keys = sorted(current_data.keys())

            # Reverse the keys if direction is 'big_to_small'
            if direction == 'big_to_small':
                keys_to_iterate = sorted_keys[::-1]
            else: # Default: 'small_to_big'
                keys_to_iterate = sorted_keys

            # --- 2. Initialize Comparison Values ---
            
            # The *actual* first value, regardless of iteration direction
            first_key_actual = sorted_keys[0] if direction == 'small_to_big' else sorted_keys[-1]
            first_value = current_data[first_key_actual]
            
            previous_value = None

            # --- 3. Iterate and Print ---
            
            for key in keys_to_iterate:
                current_value = current_data[key]
                
                # Determine the comparison value based on the mode
                if comparison_mode == 'first':
                    # Compare against the fixed first value of the sequence
                    comparison_value = first_value if key != first_key_actual else None
                else: # Default: 'previous'
                    # Compare against the previous value in the iteration order
                    comparison_value = previous_value

                # Calculate percentage change
                percent_change = calculate_percentage_change(comparison_value, current_value)
                
                # --- Printing Logic ---
                if comparison_value is None:
                    # This is the starting point of the comparison (first value in sequence)
                    print(f"  {key}: {current_value:.4f} (Start Value)")
                else:
                    if percent_change is not None:
                        sign = "+" if percent_change >= 0 else ""
                        change_str = f"({sign}{percent_change:.2f}%)"
                    else:
                        change_str = "(N/A)"
                        
                    comparison_tag = f"vs {comparison_value:.4f}"
                    
                    if comparison_mode == 'first':
                        # Clean up the output to mention the initial frame for the comparison
                        comparison_tag = f"vs {first_key_actual}"
                    
                    if show_keys is None or key in show_keys:
                        print(f"  {key}: {current_value:.4f} {change_str} {comparison_tag}")
                
                # Update previous value for the next iteration if using 'previous' mode
                previous_value = current_value
            
            return # Stop recursion here

        # Recursive Step
        if isinstance(current_data, dict):
            for key, nested_data in current_data.items():
                recursive_analysis(nested_data, level_keys + [key])
        
    # Start the analysis
    return recursive_analysis(data_structure, [])

def normalize_nested_data_to_base_frame(data, direction='small_to_big'):
    """
    Traverses a nested dictionary structure (Task -> Model -> Tech -> Frame/Pixel)
    and normalizes the scores at the deepest level. 
    
    The highest frame/pixel value's score is set as the base (1.0).
    Other scores are set as the ratio to this base score.
    
    :param data: The original nested dictionary.
    :return: A new defaultdict containing the normalized scores.
    """
    
    new_data = defaultdict(lambda: defaultdict(dict))

    def recursive_normalize(current_data, level_keys):
        """
        Recursively walks the dictionary and returns the normalized segment.
        """
        # Base Case: Check if the current data is the deepest level (Frames/Pixels)
        if current_data and isinstance(list(current_data.values())[0], (int, float)):
            
            # --- 1. Identify the Reference (Base) Frame ---
            
            # Sort all keys numerically to find the largest frame/pixel value
            try:
                sorted_keys = sorted(current_data.keys(), key=get_numerical_key)
            except TypeError:
                sorted_keys = sorted(current_data.keys())

            # Reverse the keys if direction is 'big_to_small'
            if direction == 'big_to_small':
                keys_to_iterate = sorted_keys[::-1]
            else: # Default: 'small_to_big'
                keys_to_iterate = sorted_keys
            
            # The reference key is the one with the largest numerical value (e.g., 'frame64' or '10.0')
            # reference_key = sorted_keys[-1]
            reference_key = keys_to_iterate[-1]
            reference_score = current_data[reference_key]
            
            normalized_frames = {}
            
            # --- 2. Perform Normalization ---
            
            if reference_score == 0:
                # Avoid division by zero: all scores become 0.0
                for key, score in current_data.items():
                    normalized_frames[key] = 0.0
            else:
                for key, score in current_data.items():
                    # Normalized Score = Current Score / Reference Score
                    normalized_frames[key] = score / reference_score
            
            return normalized_frames

        # Recursive Step: If the current data contains more nested dictionaries
        if isinstance(current_data, dict):
            new_segment = {}
            
            for key, nested_data in current_data.items():
                # Recursively call and get the normalized sub-structure
                result = recursive_normalize(nested_data, level_keys + [key])
                
                # Only add results if the segment returned content
                if result:
                    new_segment[key] = result
            
            return new_segment
        
        # If it's not a dict or the base case, return None or an empty dict
        return {}
        
    # Start the normalization process at the top level
    normalized_results = recursive_normalize(data, [])

    # Reconstruct the final output using defaultdict for consistency if needed, 
    # though it should already be a standard nested dict structure.
    
    # We use copy and update to avoid issues with deep recursion and defaultdict creation
    final_output = defaultdict(lambda: defaultdict(dict))
    
    # This loop populates the final defaultdict structure from the returned dict
    for task_key, models_data in normalized_results.items():
        for model_key, techniques_data in models_data.items():
            final_output[task_key][model_key] = techniques_data

    return final_output

# def get_numerical_key(k):
#     """Safely converts key strings (like 'frame4' or '10.0') to floats for sorting."""
#     try:
#         # Handles keys like 0.0, 5.0, 10.0 as floats
#         return float(k)
#     except ValueError:
#         # Handles keys like 'frame4' by extracting digits
#         num_part = ''.join(filter(str.isdigit, k))
#         return float(num_part) if num_part else k

def calculate_average_change_from_base(data_structure, comparison_key):
    """
    Calculates the grand average percentage change across all models/techniques,
    relative to a specified comparison_key.

    :param data_structure: The nested data (Task -> Model -> [Tech] -> [Frame/Value]).
    :param comparison_key: The key (e.g., 'frame64' or 0.0) to use as the baseline (100%).
    :return: The average percentage change across all model/technique groups.
    """
    all_group_changes = []
    comparison_key = str(comparison_key) # Ensure the key is a string for comparison

    def recursive_comparison(current_data):
        """Recursively traverses and collects percentage changes."""
        
        # Base Case: Check if the current data is the deepest level (scores are numbers)
        if current_data and isinstance(list(current_data.values())[0], (int, float)):
            
            # --- 1. Identify Base Score ---
            
            # Find the numerical base key in the current dictionary
            base_score = None
            for key, score in current_data.items():
                if str(key) == comparison_key:
                    base_score = score
                    break
            
            if base_score is None:
                print(f"Warning: Comparison key '{comparison_key}' not found in group: {list(current_data.keys())}. Skipping.")
                return
            
            if base_score == 0:
                print(f"Warning: Base score for key '{comparison_key}' is zero. Skipping division.")
                return
            
            # --- 2. Calculate Group Average Change ---
            
            total_change = 0.0
            count = 0
            
            for key, score in current_data.items():
                if str(key) == comparison_key:
                    continue # Skip comparison with itself

                # Percentage Change = ((Current - Base) / Base) * 100
                percent_change = ((score - base_score) / base_score) * 100
                total_change += percent_change
                count += 1
            
            if count > 0:
                group_average_change = total_change / count
                all_group_changes.append(group_average_change)
            
            return

        # Recursive Step
        if isinstance(current_data, dict) or isinstance(current_data, defaultdict):
            for key, nested_data in current_data.items():
                # For the top levels, we just traverse
                recursive_comparison(nested_data)

    # Start the analysis
    recursive_comparison(data_structure)

    # --- 3. Calculate Grand Average ---
    
    if not all_group_changes:
        return 0.0, [] # Return 0.0 if no averages were calculated
    
    grand_average = sum(all_group_changes) / len(all_group_changes)
    
    # Return the grand average and the list of individual model/group averages
    return grand_average, all_group_changes

from collections import defaultdict
import math

def calculate_average_relative_score(data_structure, comparison_key):
    """
    Calculates the grand average relative score (ratio) across all groups/models,
    where the score is normalized relative to the specified comparison_key.

    :param data_structure: The nested data (Task -> Model -> [Tech] -> [Frame/Value]).
    :param comparison_key: The key (e.g., 'frame64' or 0.0) to use as the baseline (1.0).
    :return: The average relative score (grand mean of ratios).
    """
    all_group_ratios = []
    comparison_key = str(comparison_key) # Ensure the key is a string for comparison

    def recursive_ratio_calculation(current_data):
        """Recursively traverses and collects score ratios."""
        
        # Base Case: Check if the current data is the deepest level (scores are numbers)
        if current_data and isinstance(list(current_data.values())[0], (int, float)):
            
            # --- 1. Identify Base Score ---
            
            base_score = None
            for key, score in current_data.items():
                if str(key) == comparison_key:
                    base_score = score
                    break
            
            if base_score is None:
                # print(f"Warning: Comparison key '{comparison_key}' not found in group: {list(current_data.keys())}. Skipping.")
                return
            
            if base_score == 0:
                # Avoid division by zero: treat ratios as 0 for this group
                base_score = 1 # Set to 1 temporarily to calculate ratios as 0 if all are 0
                
            # --- 2. Calculate Group Average Ratio ---
            
            total_ratio = 0.0
            count = 0
            
            for key, score in current_data.items():
                if str(key) == comparison_key:
                    continue # Skip the base key itself
                    
                # Calculate the Ratio (Relative Score) = Current Score / Base Score
                ratio = score / base_score
                total_ratio += ratio
                count += 1
            
            if count > 0:
                group_average_ratio = total_ratio / count
                all_group_ratios.append(group_average_ratio)
            
            return

        # Recursive Step
        if isinstance(current_data, dict) or isinstance(current_data, defaultdict):
            for key, nested_data in current_data.items():
                recursive_ratio_calculation(nested_data)

    # Start the analysis
    recursive_ratio_calculation(data_structure)

    # --- 3. Calculate Grand Average ---
    
    if not all_group_ratios:
        return 0.0 # Return 0.0 if no averages were calculated
    
    grand_average_ratio = sum(all_group_ratios) / len(all_group_ratios)
    
    return grand_average_ratio