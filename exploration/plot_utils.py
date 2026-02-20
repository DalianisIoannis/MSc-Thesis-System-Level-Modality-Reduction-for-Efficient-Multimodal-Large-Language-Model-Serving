import os
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
from collections import Counter
import cv2 # For video processing
import pickle
import shutil

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

import skimage
import re
import subprocess
import random
import time
from matplotlib.ticker import PercentFormatter, FuncFormatter
from paths_n_filters import *
import matplotlib.colors as mcolors

####################################################################################################################################
# scripts

def call_script_run_static_img_resizing(new_work_nam, type="image", strategy="None", current_par_dict=None):
    """
    Calls the script run_static_workloads_img_vid_resizing.py with the arguments new_work_nam, type, strategy, and current_par_dict.
    
    Parameters
    ----------
    new_work_nam : str
        The name of the new workload.
    type : str, optional
        The type of the workload, either "image" or "video". Default is "image".
    strategy : str, optional
        The strategy to use for the workload. Default is "None".
    current_par_dict : dict, optional
        The dictionary of parameters to use for the workload. Default is None.
    
    Returns
    -------
    None
    """
    # script_path = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/scripts/run_static_workloads_img_vid_resizing.py"
    param_dict_str = json.dumps(current_par_dict)
    subprocess.run(
        [
            EXECUTABLE_PYTHON,
            # "python",
            # script_path, new_work_nam, type, strategy, param_dict_str,
            RUN_SCRIPT_PATH, new_work_nam, type, strategy, param_dict_str,
        ],
        check=True,
    )
####################################################################################################################################
####################################################################################################################################
# jsonls

def all_unique_values_jsonl_column(pjsonl, column_name_outer, index=None, column_name_inner_if_app=None):
    """
    Reads a JSONL file and returns a list of all unique values of a given column.
    
    Parameters
    ----------
    pjsonl : str
        The path to the JSONL file.
    column_name_outer : str
        The name of the outer column.
    index : int, optional
        The index of the column if the column is a list. Default is None.
    column_name_inner_if_app : str, optional
        The name of the inner column if the column is a dictionary. Default is None.
    
    Returns
    -------
    list
        A list of all unique values of the given column.
    """
    set_values = set()
    with open(pjsonl) as f:
        lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            if index is not None:
                if column_name_inner_if_app is not None:
                    set_values.add(data[column_name_outer][index][column_name_inner_if_app])
                else:
                    set_values.add(data[column_name_outer][index])
            else:
                if column_name_inner_if_app is not None:
                    if not isinstance(data[column_name_outer][column_name_inner_if_app], list):
                        set_values.add(data[column_name_outer][column_name_inner_if_app])
                    else:
                        set_values.add(str(
                            data[column_name_outer][column_name_inner_if_app][0]
                         ) + "_" +\
                         str(
                                data[column_name_outer][column_name_inner_if_app][1]
                            ))
                else:
                    set_values.add(data[column_name_outer])
        f.close()
    return list(set_values)

def display_last_element(element, tabs_num):
    # if it is string, don't print all of it and keep in mind to take care of the newline characters
    """
    Prints the last element in a nested list or dict, taking care of not printing too much if it is a string.

    If the element is a string, only the first 300 characters are printed and
    newline characters are handled properly. If the string is longer than 300
    characters, "..." is printed after the first 300 characters.

    Parameters
    ----------
    element : str or int or None
        The element to print.
    tabs_num : int
        The number of tabs to print before the element.

    """
    if isinstance(element, str):
        
        # print(tabs_num*"\t", element[:300].replace("\n", "\n"+tabs_num*"\t"))
        print("\n".join((tabs_num*"\t") + line for line in element[:300].splitlines()))

        if len(element) > 300:
            if len(element) > 300:
                print(tabs_num*"\t", "...")
    else:
        # int or None
        print(tabs_num*"\t", element)

def display_element(element, tabs_num=1):

    """
    Recursively prints the structure of a given element with indentation.

    The function handles elements that are dictionaries, lists, or other types.
    For dictionaries, it prints each key and recursively calls itself for the
    key's value. For lists, it prints all elements if the list contains fewer
    than 15 items, otherwise it only prints the first element. If the element
    is neither a dictionary nor a list, it calls `display_last_element` to
    handle the printing.

    Parameters
    ----------
    element : dict, list, or other types
        The element to be displayed. It can be a dict, list, int, None, or str.
    tabs_num : int, optional
        The number of tabs to print before the element, default is 1.

    Notes
    -----
      Handles the "image_size_pixel" key in a special way by printing its value directly.
      If the element is a dict, it will print the keys and the length of the values if they are lists.
      If the element is a list, it will print the first element and then recursively call itself on it.
      If the element is a string, it will print the first 300 characters and then "..." if it is longer than 300.
      If the element is an int or None, it will simply print it.
    """

    if isinstance(element, dict):
    
        for keys in element.keys():
            
            # Might be an inner key of data so check to print number of samples:
            if isinstance(element[keys], list):
                print(tabs_num*"\t", keys, " contains: ", len(element[keys]), sep='')
            else:
                print(tabs_num*"\t", keys, sep='')
            if keys != "image_size_pixel":
                display_element(element[keys], tabs_num+1)
            else:
                print((tabs_num+1)*"\t", element[keys], sep='')
    
    elif isinstance(element, list):
        # SOS
        # if they are a lot probably means they are a lot of samples so display only one
        if len(element) < 15:
            for list_item in element:
                display_element(list_item, tabs_num+1)
        else:
            display_element(element[0], tabs_num+1)
    else:
        display_last_element(element, tabs_num)

def universal_json_jsonl_printer(path_to_file):
    # Print the total path to the file
    """
    Prints the contents of a JSON or JSONL file, with indentation and prettiness.

    The function prints the total number of samples (lines) in the file.
    For each JSON object, it checks if any value is a list and prints
    the list elements and their keys. If not a list, it prints the key
    and value.

    Parameters
    ----------
    path_to_file : str
        The path to the JSON or JSONL file.

    Notes
    -----
        If the file is a JSONL file, it prints the total number of samples (lines) in the file.
        It then prints the first sample in a prettified format.
        If the file is a JSON file, it prints the contents of the file in a prettified format.
    """

    if not path_to_file.startswith("/"):
        if not path_to_file.startswith(".."):
            print(110*"*", "\n", 110*"*", "\n", os.path.join(os.getcwd(), path_to_file), sep='')
        else:
            print(110*"*", "\n", 110*"*", "\n", 
                  os.path.join(
                      os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                      path_to_file[3:]
                      ),
                      sep='')
    else:
        print(110*"*", "\n", 110*"*", "\n", path_to_file, sep='')

    # Open the file and check if it is json or jsonl
    with open(path_to_file, "r") as f:
        
        if path_to_file.endswith(".jsonl"):
            
            data = f.readlines()
            print(f"File samples are {len(data)} in the format:")
            display_element(json.loads(data[0]))

        else:
            data = json.load(f)
            print(f"File samples are {len(data)} in the format:")
            display_element(data, 0)
            
        f.close()
    print(110*"*", "\n", 110*"*", sep='')

def get_aokvqa_sample_with_id(aokvqa_jsonl, id, is_request=False):
    """
    Retrieves a single entry from a JSONL file based on a specified ID.

    Parameters
    ----------
    aokvqa_jsonl : str
        Path to the JSONL file containing AOKVQA data.
    id : str
        The ID of the entry to retrieve.
    is_request : bool, optional
        Determines whether to search for the ID in the "request" field (default is False).

    Returns
    -------
    dict
        The entry from the JSONL file that matches the specified ID. Returns None if no match is found.
    """

    with open(aokvqa_jsonl, "r") as f:
        for line in f:
            entry = json.loads(line)

            if is_request:
                item = entry["request"]["id"]
            else:
                item = entry["id"]

            if item == id:
                f.close()
                return entry

def get_value_from_aokvqa_request(request, key, is_request=False):
    """
    Retrieves a value from an AOKVQA request based on a specified key.

    Parameters
    ----------
    request : dict
        The AOKVQA request from which the value is to be retrieved.
    key : str
        The key associated with the desired value in the request.
    is_request : bool, optional
        Indicates whether the key is within the nested "request" field. Defaults to False.

    Returns
    -------
    The value associated with the specified key within the request. 
    Returns the value from the "request" field if is_request is True, otherwise returns the value directly from the request.
    """

    if is_request:
        return request["request"][key]
    else:
        return request[key]

def create_subworkload(file_initial,
                       subset_samples,
                       folder_initial,
                       folder_to,
                       shuffle=True,
                       ):
    new_f = os.path.join(folder_initial, file_initial.split(".")[0].split("_")[0]+"_"+str(subset_samples)+".jsonl")
    """
    Creates a subset of a JSONL file, optionally shuffled, and saves it to a new file.

    Parameters
    ----------
    file_initial : str
        The name of the initial JSONL file.
    subset_samples : int
        The number of samples to include in the subset.
    folder_initial : str
        The path to the folder containing the initial JSONL file.
    folder_to : str
        The path to the folder where the subset JSONL file should be saved.
    shuffle : bool, optional
        Whether to shuffle the samples before selecting the subset. Defaults to True.

    Returns
    -------
    None
    """
    print(new_f)
    if not os.path.exists(new_f):
        data_save = []
        with open(os.path.join(folder_initial, file_initial), "r") as f:
            lines_init = f.readlines()

            if shuffle:
                random.Random(42).shuffle(lines_init)
            count = 0
            for record in lines_init:
                data_read = json.loads(record)
                data_save.append(data_read)

                count += 1
                if count == subset_samples:
                    break
            
            f.close()
        
        with open(os.path.join(folder_to, file_initial.split(".")[0].split("_")[0]+"_"+str(subset_samples)+".jsonl"), "w") as f:
            for item in data_save:
                f.write(json.dumps(item) + "\n")
            f.close()

def get_requests_in_list(jsonl_request_or_stat_pth):
    """
    Reads a JSONL file and returns all entries as a list of dictionaries.

    Parameters
    ----------
    jsonl_request_or_stat_pth : str
        The path to the JSONL file containing the requests or statistics data.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a parsed JSON object from each line of the file.
    """
    with open(jsonl_request_or_stat_pth, "r") as f:
        requests = [json.loads(line) for line in f]
        f.close()
    return requests

def check_if_all_paths_exist(jsonl_path, modality_col="modality_path"):
    """
    Checks if all image paths specified in a JSONL file exist on the filesystem.

    Parameters
    ----------
    jsonl_path : str
        The path to the JSONL file containing entries with image paths.
    modality_col : str, optional
        The column name containing the image paths. Defaults to "modality_path".

    Returns
    -------
    None
        Prints a message for each image path that does not exist.
    """
    with open(jsonl_path, "r") as fread:
        for line in fread:
            line = json.loads(line)
            img_pth = get_value_from_aokvqa_request(line, modality_col, is_request=True)
            if not os.path.exists(img_pth):
                print(f"Image path {img_pth} does not exist")
        fread.close()

def count_value_apperances(pjsonl, column_name_outer, index=None, column_name_inner_if_app=None):
    """
    Counts the occurrences of values in a specified column of a JSONL file.

    This function reads a JSONL file and counts the occurrences of values
    within a specified column. If the column contains nested data and an index
    is provided, it counts the values at that specific index. Additionally, if
    a nested column name is provided, it counts values from that nested column.

    Parameters
    ----------
    pjsonl : str
        Path to the JSONL file to be processed.
    column_name_outer : str
        The outer column name from which values are to be counted.
    index : int, optional
        An optional index for nested lists within the outer column. Default is None.
    column_name_inner_if_app : str, optional
        An optional inner column name for nested dictionaries within the outer column. Default is None.

    Returns
    -------
    dict
        A dictionary where keys are the unique values found in the specified column,
        and values are the counts of those unique values.
    """

    dict_values = dict()
    with open(pjsonl) as f:
        lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            if index is not None:
                if column_name_inner_if_app is not None:
                    if data[column_name_outer][index][column_name_inner_if_app] not in dict_values:
                        dict_values[data[column_name_outer][index][column_name_inner_if_app]] = 1
                    else:
                        dict_values[data[column_name_outer][index][column_name_inner_if_app]] += 1
                else:
                    if data[column_name_outer][index] not in dict_values:
                        dict_values[data[column_name_outer][index]] = 1
                    else:
                        dict_values[data[column_name_outer][index]] += 1
            else:
                if column_name_inner_if_app is not None:
                    if data[column_name_outer][column_name_inner_if_app] not in dict_values:
                        dict_values[data[column_name_outer][column_name_inner_if_app]] = 1
                    else:
                        dict_values[data[column_name_outer][column_name_inner_if_app]] += 1
        f.close()
    return dict_values

def get_all_column_values(pjsonl, column_name):
    """
    Retrieves all values from a specific column in a JSONL file.

    Parameters
    ----------
    pjsonl : str
        Path to the JSONL file to be processed.
    column_name : str
        The name of the column from which values are to be retrieved.

    Returns
    -------
    list
        A list of values from the specified column.
    """
    reqs = get_requests_in_list(pjsonl)
    return [req[column_name] for req in reqs]
####################################################################################################################################
####################################################################################################################################
# accuracy

def check_response_truth(response):
    """
    Checks whether the llm_response matches the correct answer for a single response.

    Parameters
    ----------
    response : dict
        A dictionary containing the following keys:
        - "llm_response": str, the response from the LLM
        - "correct_answer": str, the correct answer

    Returns
    -------
    bool
        True if the LLM response matches the correct answer, False otherwise
    """
    # if response["llm_response"] in [" A ", "A ", " A"] and response["correct_answer"] == "0":
    if response["llm_response"] in [" A ", "A ", " A"] and response["correct_answer"] in ["0", "A", "A."]:
        return True
    # elif response["llm_response"]  in [" B ", "B ", " B"] and response["correct_answer"] == "1":
    elif response["llm_response"]  in [" B ", "B ", " B"] and response["correct_answer"] in ["1", "B", "B."]:
        return True
    # elif response["llm_response"]  in [" C ", "C ", " C"] and response["correct_answer"] == "2":
    elif response["llm_response"]  in [" C ", "C ", " C"] and response["correct_answer"] in ["2", "C", "C."]:
        return True
    # elif response["llm_response"]  in [" D ", "D ", " D"] and response["correct_answer"] == "3":
    elif response["llm_response"]  in [" D ", "D ", " D"] and response["correct_answer"] in ["3", "D", "D."]:
        return True
    else:
        # print("in check_response_truth ", response["llm_response"], "correct ", response["correct_answer"])
        return False

def letter_to_numeric(letter):
    """
    Maps a letter (A, B, C, D) to its corresponding numeric value (0, 1, 2, 3).

    Parameters
    ----------
    letter : str
        The letter to be mapped.

    Returns
    -------
    int
        The numeric value corresponding to the letter. Returns -1 if the letter is not A, B, C, or D.
    """

    # remove spaces
    letter = letter.strip()
    if letter in [" A ", "A ", " A", "A.", "A"]:
        return 0
    elif letter in [" B ", "B ", " B", "B.", "B"]:
        return 1
    elif letter in [" C ", "C ", " C", "C.", "C"]:
        return 2
    elif letter in [" D ", "D ", " D", "D.", "D"]:
        return 3
    else:
        print("\tin letter_to_numeric ", letter)
        return -1

def count_aokvqa_accuracy(response_jsonl):
    """
    Counts the accuracy of AOKVQA responses in a given JSONL file.

    Parameters
    ----------
    response_jsonl : str
        The path to the JSONL file containing the responses.

    Returns
    -------
    tuple
        A tuple containing:
        - The accuracy of the responses (float)
        - The number of correct responses (int)
    """
    y_true = []
    y_pred = []
    count_corrects = 0
    with open(response_jsonl, "r") as f:
        responses = [json.loads(line) for line in f]
        
        for response in responses:
            count_corrects += int(check_response_truth(response))

            # if response["llm_response"].strip(".").strip() not in [" A ", "A ", " A", "A", " B ", "B ", " B", "B", " C ", "C ", " C", "C", "D", " D ", "D ", " D"]:
            if response["llm_response"].strip(".").strip() not in ["A", "B", "C", "D"]:
                print("\tNOT appropriate ", response["llm_response"])

            y_true.append(int(response["correct_answer"]))
            y_pred.append(letter_to_numeric(response["llm_response"]))

            # # for displaying some wrong ones
            # if int(response["correct_answer"]) != letter_to_numeric(response["llm_response"]):
            #     print("wrong ", response["correct_answer"], response["llm_response"], response["id"])

    return np.sum(np.equal(y_true, y_pred)) / len(y_true), count_corrects

def count_aokvqa_accuracy_and_get_wrong(response_jsonl):
    """
    Counts the accuracy of AOKVQA responses in a given JSONL file and returns the ids of the wrong ones.

    Parameters
    ----------
    response_jsonl : str
        The path to the JSONL file containing the responses.

    Returns
    -------
    tuple
        A tuple containing:
        - The accuracy of the responses (float)
        - The number of correct responses (int)
        - The list of ids of the wrong responses (list of str)
    """
    y_true = []
    y_pred = []
    list_wrong = []
    count_corrects = 0
    with open(response_jsonl, "r") as f:
        responses = [json.loads(line) for line in f]
        
        for response in responses:
            count_corrects += int(check_response_truth(response))

            # if response["llm_response"].strip(".").strip() not in [" A ", "A ", " A", "A", " B ", "B ", " B", "B", " C ", "C ", " C", "C", "D", " D ", "D ", " D"]:
            if response["llm_response"].strip(".").strip() not in ["A", "B", "C", "D"]:
                print("NOT appropriate ", response["llm_response"])

            y_true.append(int(response["correct_answer"]))
            y_pred.append(letter_to_numeric(response["llm_response"]))

            if int(response["correct_answer"]) != letter_to_numeric(response["llm_response"]):
                list_wrong.append(response["id"])

    return np.sum(np.equal(y_true, y_pred)) / len(y_true), count_corrects, list_wrong

def print_jsonl_length(jsonl_file_path):
    """
    Reads a JSONL file and prints its length.

    Parameters
    ----------
    jsonl_file_path : str
        The path to the JSONL file.

    Returns
    -------
    None
    """
    with open(jsonl_file_path, "r") as f:
        print(len([json.loads(line) for line in f]))
        f.close()

def compare_accuracy_before_n_after_resize(initial_responses, final_resized_responses, final_resized_requests, initial_requests,
                                           only_fixed=False):
    """
    Compares the accuracy of the responses between the initial responses and the ones after resizing.

    Parameters
    ----------
    initial_responses : str
        The path to the JSONL file containing the initial responses.
    final_resized_responses : str
        The path to the JSONL file containing the responses after resizing.
    final_resized_requests : str
        The path to the JSONL file containing the requests after resizing.
    initial_requests : str
        The path to the JSONL file containing the initial requests.
    only_fixed : bool, optional
        If True, then only see responses that were wrong and became correct after resizing. Defaults to False.

    Returns
    -------
    None
    """
    with open(initial_responses, "r") as finit:
        responses_init = [json.loads(line) for line in finit]
        finit.close()
    
    with open(final_resized_responses, "r") as ffinal:
        responses_final = [json.loads(line) for line in ffinal]
        ffinal.close()
    
    for init_response, final_response in zip(responses_init, responses_final):
        
        print_res = False
        if not only_fixed:
            # if letter_to_numeric(init_response["llm_response"]) != letter_to_numeric(final_response["llm_response"]):
            if letter_to_numeric(init_response["llm_response"]) == letter_to_numeric(final_response["llm_response"]):
                
                print_res = True
        else:
            # initial response is wrong
            if letter_to_numeric(init_response["llm_response"]) != int(init_response["correct_answer"]) and \
            letter_to_numeric(final_response["llm_response"]) == int(final_response["correct_answer"]): # final response is correct
            
            # # initial response is correct and final response is correct
            # if letter_to_numeric(init_response["llm_response"]) == int(init_response["correct_answer"]) and \
            # letter_to_numeric(final_response["llm_response"]) == int(final_response["correct_answer"]): # final response is correct
                
                print_res = True
        
        if print_res:
            print("Correct: ", init_response["correct_answer"], " initial response:", init_response["llm_response"], " and resized response:", final_response["llm_response"])

            # this part here displays images before and after resizing
            
            # get id of the request - it would be the same if I used the resized responses
            request_id = get_value_from_aokvqa_request(init_response, "id", is_request=False)
            
            # the initial request itself
            initial_sample_by_id = get_aokvqa_sample_with_id(initial_requests, id=request_id, is_request=True)
            # the resized request itself
            final_sample_by_id = get_aokvqa_sample_with_id(final_resized_requests, id=request_id, is_request=True)

            # the input text question
            in_ques = get_value_from_aokvqa_request(initial_sample_by_id, "input", is_request=True)
            print(in_ques)

            # path to original_img
            or_img_path = get_value_from_aokvqa_request(initial_sample_by_id, "modality_path", is_request=True)
            # print(or_img_path)
            # print("Image dimensions:", get_img_dimensions_given_path(or_img_path))
            print("Byte size of the image:", file_size(or_img_path, convert_to = "KB"))
            display_matplot_lib_img(or_img_path)
            # display_img_given_path(or_img_path)

            # path to resized_img
            res_img_path = get_value_from_aokvqa_request(final_sample_by_id, "modality_path", is_request=True)
            # print(res_img_path)
            # print("Image dimensions:", get_img_dimensions_given_path(res_img_path))
            print("Byte size of the image:", file_size(res_img_path, convert_to = "KB"))
            # display_img_given_path(res_img_path)
            display_matplot_lib_img(res_img_path)

            break

def count_vid_mc_accuracy(response_jsonl):
    """
    Counts the accuracy of Video MC responses in a given JSONL file.

    Parameters
    ----------
    response_jsonl : str
        The path to the JSONL file containing the responses.

    Returns
    -------
    tuple
        A tuple containing:
        - The accuracy of the responses (float)
        - The number of correct responses (int)
    """
    y_true = []
    y_pred = []
    count_corrects = 0
    with open(response_jsonl, "r") as f:
        responses = [json.loads(line) for line in f]
        
        for response in responses:
            count_corrects += int(check_response_truth(response))
            # if response["llm_response"].strip(".").strip() not in [" A ", "A ", " A", "A", " B ", "B ", " B", "B", " C ", "C ", " C", "C", "D", " D ", "D ", " D"]:
            if response["llm_response"].strip(".").strip() not in ["A", "B", "C", "D"]:
                print("NOT appropriate ", response["llm_response"])

            y_true.append(letter_to_numeric(response["correct_answer"]))
            y_pred.append(letter_to_numeric(response["llm_response"]))

    return np.sum(np.equal(y_true, y_pred)) / len(y_true), count_corrects

def count_vid_mc_accuracy_and_get_wrong(response_jsonl):
    """
    Counts the accuracy of Video MC responses in a given JSONL file and returns the ids of the wrong ones.

    Parameters
    ----------
    response_jsonl : str
        The path to the JSONL file containing the responses.

    Returns
    -------
    tuple
        A tuple containing:
        - The accuracy of the responses (float)
        - The number of correct responses (int)
        - The list of ids of the wrong responses (list of str)
    """
    y_true = []
    y_pred = []
    list_wrong = []
    count_corrects = 0
    with open(response_jsonl, "r") as f:
        responses = [json.loads(line) for line in f]
        
        for response in responses:
            count_corrects += int(check_response_truth(response))
            # if response["llm_response"].strip(".").strip() not in [" A ", "A ", " A", "A", " B ", "B ", " B", "B", " C ", "C ", " C", "C", "D", " D ", "D ", " D"]:
            if response["llm_response"].strip(".").strip() not in ["A", "B", "C", "D"]:
                print("NOT appropriate ", response["llm_response"])

            y_true.append(letter_to_numeric(response["correct_answer"]))
            y_pred.append(letter_to_numeric(response["llm_response"]))

            if letter_to_numeric(response["correct_answer"]) != letter_to_numeric(response["llm_response"]):
                list_wrong.append(response["id"])

    return np.sum(np.equal(y_true, y_pred)) / len(y_true), count_corrects, list_wrong

def print_used_parameters(**param_dicts):
    """
    Prints the used parameters for each category.

    Parameters
    ----------
    **param_dicts : dict
        Dictionaries that contain all the parameters that I want to print
    """
    print("\nUsed Parameters by Category:\n")
    for category, options in param_dicts.items():
        print(f"{category}:")
        for key, val in options.items():
            print(f"  {key} → {val}")
        print()

def total_files_with_parameter(parameter, parameter_dict=RESAMPLING_FILTERS, dict_other=None):
    """
    Returns a list of file names in the dictionary_with_workloads that contain the given parameter filter alias.

    Parameters
    ----------
    parameter : str
        The parameter to search for in the file names.
    parameter_dict : dict, optional
        A dictionary that maps a parameter to its corresponding filter alias. Defaults to RESAMPLING_FILTERS.
    dict_other : dict, optional
        A dictionary to search in. Defaults to dictionary_with_workloads.

    Returns
    -------
    list
        A list of file names that contain the given parameter filter alias.
    """
    filter_alias = parameter_dict[parameter]

    if dict_other is None:
        dict_other = dictionary_with_workloads
    file_names = []
    for i in dict_other.keys(): # aokvqa_350_hdp_ner_rgb_10

        if filter_alias in i:
            file_names.append(i)

    print(f"Files with {parameter} filter: {len(file_names)}")
    return file_names
####################################################################################################################################
####################################################################################################################################
# resize stuff

"""
dictionary that will contain all the workloads and the corresponding files and paths and statistics
    aokvqa_350_thu_lan_rgb_01
        new_request_jsonl 
        responses_new 
        stats_new 
        initial_requests_jsonl
        avg pixel reduction
        new modality tokens list
        modality tokens avg reduction
"""
dictionary_with_workloads = {}

"""
like the above but only with the most common size -> [640, 480]
"""
dictionary_with_workloads_common_size = {}

def find_key_by_value(d, target_value):
    """
    Finds all keys in a dictionary that correspond to a given target value.

    Parameters
    ----------
    d : dict
        The dictionary to search within.
    target_value : any
        The value to search for in the dictionary.

    Returns
    -------
    list
        A list of keys whose corresponding values match the target value.
    """

    return [key for key, value in d.items() if value == target_value]

def add_procedure_resize_stats(workload_nam, statistics):
    """
    Updates the statistics of the workload with the given name in the RESIZE_PROCEDURE_STATS file.
    
    Parameters
    ----------
    workload_nam : str
        The name of the workload to be updated.
    statistics : dict
        A dictionary containing the statistics to be updated.
    """
    existing_stats = get_requests_in_list(RESIZE_PROCEDURE_STATS)

    # if the workload key exists, renew the statistics, otherwise add it
    found_already = False
    for stat in existing_stats:
        if workload_nam in stat.keys():
            for stats_count in statistics:
                stat[workload_nam].update({stats_count: statistics[stats_count]})
            found_already = True
            break
    if not found_already:
        existing_stats.append({workload_nam: {stat_cat: statistics[stat_cat] for stat_cat in statistics}})

    with open(RESIZE_PROCEDURE_STATS, "w") as fwrite:
        for stat in existing_stats:
            fwrite.write(json.dumps(stat) + "\n")
        fwrite.close()

def avg_resize(
        img_create_path, jsonl_create_path, init_jsonl_request_pth, dimens, resampl_fil
):
    """
    Creates a new folder of images with the average dimensions of the images in the given init_jsonl_request_pth.
    The new images are saved in the img_create_path folder.
    The new jsonl file is saved in the jsonl_create_path file.
    The function also calculates the time taken to create the new folder and jsonl and adds it to the RESIZE_PROCEDURE_STATS file.
    The function also executes the workload to get the responses and statistics.
    The function returns the responses and statistics files.
    
    Parameters
    ----------
    img_create_path : str
        The path to the new folder of images.
    jsonl_create_path : str
        The path to the new jsonl file.
    init_jsonl_request_pth : str
        The path to the init jsonl file.
    dimens : tuple
        The dimensions to resize the images to.
    resampl_fil : str
        The resampling filter to use.
    """
    if create_folder_if_no_exists(img_create_path):
        requests = get_requests_in_list(init_jsonl_request_pth)

        # get the avg dimensions of all the images in the request file - We calculate them again
        _, pixel_tuples = aokvqa_img_details(jsonl_file_path=init_jsonl_request_pth, print_details=True)

        created_width = int(sum(pixel_tuples[0])/len(pixel_tuples[0]))
        created_height = int(sum(pixel_tuples[1])/len(pixel_tuples[1]))

        t0 = time.time()
        with open(jsonl_create_path, "w") as fwrite:
            for request in requests:

                img_pth = get_value_from_aokvqa_request(request, "modality_path", is_request=True)
                img = Image.open(img_pth)
                img = img.resize((created_width, created_height), resample=FILTERS_OBJECTS[resampl_fil])

                img_save_path = img_create_path + "/" + request["request"]["id"] + ".jpg"
                img.save(img_save_path)

                item = request # change the modality path of the dict and save it to the new jsonl
                item["request"].update({'modality_path': img_save_path})
                item["request"].update({'modality_size': img.size})
                fwrite.write(json.dumps(item) + "\n")
            
            fwrite.close()
    
        t1 = time.time()
        total_n = t1-t0

        statistics_workload_creation = {"time": total_n}
        # img_create_path.split("/")[-1] = aokvqa_25_hcd_lan_00
        add_procedure_resize_stats(img_create_path.split("/")[-1], statistics_workload_creation)

        # execute the workload to get the responses and statistics
        call_script_run_static_img_resizing(new_work_nam = img_create_path.split("/")[-1])

    # return the statistics of the workload
    found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, img_create_path.split("/")[-1])
    statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_AOKVQA, img_create_path.split("/")[-1])

    return found_resp, statf

def hard_coded_or_proportional_technique(img_create_path, jsonl_create_path, init_jsonl_request_pth, dimens, resampl_fil,
                                         dimensions_to_mess="all", technique="Hard Coded"):
    """
    This function creates a new jsonl file from an init jsonl file.
    The function resizes the images in the init jsonl file according to the technique and dimensions provided.
    The function also calculates the time taken to create the new folder and jsonl and adds it to the RESIZE_PROCEDURE_STATS file.
    The function also executes the workload to get the responses and statistics.
    The function returns the responses and statistics files.

    Parameters
    ----------
    img_create_path : str
        The path to the new folder of images.
    jsonl_create_path : str
        The path to the new jsonl file.
    init_jsonl_request_pth : str
        The path to the init jsonl file.
    dimens : tuple
        The dimensions to resize the images to.
    resampl_fil : str
        The resampling filter to use.
    dimensions_to_mess : str
        The dimension to mess with. "all", "width", "height".
    technique : str
        The technique to use. "Hard Coded", "Proportional", "Thumbnail", "Thumbnail Proportionally".
    """
    if create_folder_if_no_exists(img_create_path):
        requests = get_requests_in_list(init_jsonl_request_pth)

        # created_width, created_height = DIMENSIONS_DICT[dimens] # 00 -> (200,300)
        if technique=="Hard Coded" or technique=="Thumbnail":
            created_width, created_height = DIMENSIONS_DICT[dimens] # 00 -> (200,300)
        elif technique=="Proportional":
            created_width, created_height = DIMENSIONS_DICT_PROPORTIONALITY[dimens] # 00 -> (200,300)

        t0 = time.time()
        print(f"Count image creation time")
        if not os.path.exists(jsonl_create_path):
            with open(jsonl_create_path, "w") as fwrite:
                for request in requests:

                    img_pth = get_value_from_aokvqa_request(request, "modality_path", is_request=True)
                    
                    img = Image.open(img_pth)
                    img_init = img.size

                    if technique=="Hard Coded":
                        if dimensions_to_mess == "all":
                            img = img.resize((created_width, created_height), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "width":
                            img = img.resize((created_width, img_init[1]), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "height":
                            img = img.resize((img_init[0], created_height), resample=FILTERS_OBJECTS[resampl_fil])
                    
                    elif technique=="Proportional":
                        if dimensions_to_mess == "all":
                            img = img.resize((
                                int(img_init[0] - (created_width*img_init[0])/100),
                                int(img_init[1] - (created_height*img_init[1])/100),
                                ), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "width":
                            img = img.resize(
                                (
                                    int(img_init[0] - (created_width*img_init[0])/100),
                                    img_init[1]
                                    ),
                                    resample=FILTERS_OBJECTS[resampl_fil]
                                )
                        elif dimensions_to_mess == "height":
                            img = img.resize((img_init[0], int(img_init[1] - (created_height*img_init[1])/100)), resample=FILTERS_OBJECTS[resampl_fil])
                    
                    elif technique=="Thumbnail":
                        if dimensions_to_mess == "all":
                            img.thumbnail(size=(created_width, created_height), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "width":
                            img.thumbnail(size=(created_width, img_init[1]), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "height":
                            img.thumbnail(size=(img_init[0], created_height), resample=FILTERS_OBJECTS[resampl_fil])
                    
                    elif technique=="Thumbnail Proportionally":
                        if dimensions_to_mess == "all":
                            img.thumbnail(size=(
                                int(img_init[0] - (created_width*img_init[0])/100),
                                int(img_init[1] - (created_height*img_init[1])/100)), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "width":
                            img.thumbnail(size=(
                                int(img_init[0] - (created_width*img_init[0])/100), img_init[1]), resample=FILTERS_OBJECTS[resampl_fil])
                        elif dimensions_to_mess == "height":
                            img.thumbnail(size=(img_init[0],
                                                int(img_init[1] - (created_height*img_init[1])/100)), resample=FILTERS_OBJECTS[resampl_fil])

                    img_save_path = img_create_path + "/" + request["request"]["modality_path"].split("/")[-1]
                    img.save(img_save_path)

                    item = request # change the modality path of the dict and save it to the new jsonl
                    item["request"].update({'modality_path': img_save_path})
                    item["request"].update({'modality_size': img.size})
                    fwrite.write(json.dumps(item) + "\n")
                
                fwrite.close()
        
        t1 = time.time()
        total_n = t1-t0

        statistics_workload_creation = {"time": total_n}
        # img_create_path.split("/")[-1] = aokvqa_25_hcd_lan_00
        add_procedure_resize_stats(img_create_path.split("/")[-1], statistics_workload_creation)
        # execute the workload to get the responses and statistics
        call_script_run_static_img_resizing(new_work_nam = img_create_path.split("/")[-1])

    # return the statistics of the workload
    found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, img_create_path.split("/")[-1])
    statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_AOKVQA, img_create_path.split("/")[-1])

    return found_resp, statf

def get_img_full_pth_from_tsv_row(tsv_row, img_col="index"):
    """
    Retrieves the full image path from a TSV row.

    This function determines the full file path for an image based on the information
    provided in a TSV row. It checks for the presence of an "image_path" key and verifies
    if the path exists. If the path doesn't exist or the key is absent, it constructs the 
    path using the provided index or a default column name.

    Parameters
    ----------
    tsv_row : dict
        A dictionary representing a row from a TSV file, containing image information.
    img_col : str, optional
        The column name to use for constructing the image path if "image_path" is not present,
        default is "index".

    Returns
    -------
    str
        The full path to the image file corresponding to the given TSV row.
    """
    if "image_path" in tsv_row:
        # probably only AOKVQA_original has correct full path like this
        if os.path.exists(tsv_row["image_path"]):
            return tsv_row["image_path"]
        else:
            # there is something like 001.jpg
            return os.path.join(ORIGINAL_img_folder, tsv_row["image_path"])
    # else return index + .jpg
    return os.path.join(
        ORIGINAL_img_folder,
        str(tsv_row[img_col]) + ".jpg"
        )

# I don't have to save a new file since only the folder where the images are saved is changed
def hard_coded_or_proportional_technique_tsv(img_create_path, tsv_create_path, init_tsv_request_pth, dimens,
                                             resampl_fil,
                                             dimensions_to_mess="all", technique="Hard Coded"):
    """
    Resizes images based on a given technique and saves them along with an updated TSV file.

    This function processes images specified in a TSV file, resizing them according to either
    a hard-coded or proportional technique. The resized images are saved in the specified directory,
    and an updated copy of the TSV file is also saved.

    Parameters
    ----------
    img_create_path : str
        The directory path where the resized images should be saved.
    tsv_create_path : str
        The file path where the updated TSV should be saved.
    init_tsv_request_pth : str
        The path to the initial TSV file containing image requests.
    dimens : str
        The dimension key used for resizing images when using the proportional technique.
    resampl_fil : str
        The resampling filter to use for resizing images.
    dimensions_to_mess : str, optional
        Specifies which dimensions to adjust when using the proportional technique (default is "all").
    technique : str, optional
        The technique to use for resizing images ("Hard Coded" or "Proportional", default is "Hard Coded").

    Returns
    -------
    None
    """

    if create_folder_if_no_exists(img_create_path):
        tsv_requests = pd.read_csv(init_tsv_request_pth, sep = '\t')

        if technique=="Proportional":
            created_width, created_height = DIMENSIONS_DICT_PROPORTIONALITY[dimens]

        t0 = time.time()
        for index, row in tsv_requests.iterrows():

            img_pth = get_img_full_pth_from_tsv_row(row)
            
            img = Image.open(img_pth)
            img_init = img.size

            if technique=="Proportional":
                if dimensions_to_mess == "all":
                    img = img.resize((
                        int(img_init[0] - (created_width*img_init[0])/100),
                        int(img_init[1] - (created_height*img_init[1])/100),
                        ), resample=FILTERS_OBJECTS[resampl_fil])

            img_save_path = img_create_path + "/" + img_pth.split("/")[-1]
            img.save(img_save_path)
            tsv_requests.at[index, "image_path"] = img_save_path

        t1 = time.time()
        total_n = t1-t0

        # save a copy of the tsv with correct name
        tsv_requests.to_csv(tsv_create_path, sep = '\t', index = False)

        statistics_workload_creation = {"time": total_n}
        add_procedure_resize_stats(img_create_path.split("/")[-1], statistics_workload_creation)

def create_resized_img_folder_n_tsv_file(init_tsv_request_pth,
                                         resampl_fil, dim_res_techn, dimens, color,
                                         created_img_folder_pth,
                                         created_jsonl_pth):
    """
    Creates a resized image folder and TSV file based on specified parameters.

    This function processes the initial TSV request path to generate resized images according
    to the given resampling filter, dimension resize technique, dimensions, and color. It creates
    a new folder for the resized images and saves the corresponding TSV file with updated metadata.

    Parameters
    ----------
    init_tsv_request_pth : str
        The path to the initial TSV file containing the requests for image processing.
    resampl_fil : str
        The resampling filter to apply during image resizing.
    dim_res_techn : str
        The dimension resize technique to use, which can influence whether resizing is proportional.
    dimens : str
        The dimension identifier for resizing the image.
    color : str
        The color format of the image, e.g., RGB.
    created_img_folder_pth : str
        The path to the directory where the resized images should be saved.
    created_jsonl_pth : str
        The path to the directory where the updated TSV file should be saved.

    Returns
    -------
    None
    """

    init_tsv_alias = init_tsv_request_pth.split("/")[-1].split(".")[0] # /path/MMBench_DEV_EN.tsv -> MMBench_DEV_EN
    
    # is it for proportional or no?
    if dim_res_techn in DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY.keys():
        dimension_resize_technique = DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY[dim_res_techn] # Thumbnail -> thu etc
    
    resampl_fil_alias = RESAMPLING_FILTERS[resampl_fil] # LANCZOS -> lan etc
    alias_dim_filter = f"{init_tsv_alias}_{dimension_resize_technique}_{resampl_fil_alias}_{color}" # aokvqa_350_hcd_lan_rgb

    created_folder = f"{alias_dim_filter}_{dimens}" # aokvqa_350_hcd_lan_rgb_01
    created_folder_path = os.path.join(created_img_folder_pth, created_folder) # /srv/muse-lab/datasets/A-OKVQA/images/aokvqa_350_hcd_lan_rgb_01
    created_jsonl_path = os.path.join(created_jsonl_pth, f"{created_folder}.tsv")
    
    if dim_res_techn == "Both Dimensions Proportionally":
        # found_resp, statf = 
        hard_coded_or_proportional_technique_tsv(created_folder_path, created_jsonl_path,
                                                                 init_tsv_request_pth, dimens, resampl_fil,
                                                                 dimensions_to_mess="all",
                                                                 technique="Proportional")

# Function to Create Resized Workloads and Corresponding Files
def create_resized_img_folder_n_jsonl(init_jsonl_request_pth, resampl_fil, dim_res_techn, dimens, color,
                                      created_img_folder_pth=AOKVQA_FOLDER,
                                      created_jsonl_pth=AOKVQA_RGB_STATICS):

    """
    Function to create resized workload for a given initial jsonl workload and a specific dimension, color, and resampling filter.
    
    Parameters
    ----------
    init_jsonl_request_pth : str
        The path to the initial jsonl workload.
    resampl_fil : str
        The resampling filter to use. It can be LANCZOS, BICUBIC, BILINEAR, or NEAREST.
    dim_res_techn : str
        The dimension resize technique to use. It can be Hard Coded, Hard Coded Width, Hard Coded Height, 
        Both Dimensions Proportionally, Width Proportionally, Height Proportionally, Thumbnail, Thumbnail Proportionally, or Average.
    dimens : int or str
        The dimension to resize the image to. If it is an int, it is the dimension to resize the shorter side of the image to.
        If it is a str, it is the alias of the dimension in the AVG_DIMENSIONS_DICT.
    color : str
        The color of the image. It can be RGB or GRAY.
    
    Returns
    -------
    found_resp : str
        The path to the new jsonl workload with the resized images.
    statf : str
        The path to the new jsonl workload with the statistics of the resized images.
    """
    init_jsonl_alias = init_jsonl_request_pth.split("/")[-1].split(".")[0] # /path/aokvqa_350.jsonl -> aokvqa_350
    
    # is it for proportional or no?
    if dim_res_techn in DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY.keys():
        dimension_resize_technique = DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY[dim_res_techn] # Thumbnail -> thu etc
    else:
        dimension_resize_technique = DIMENSION_RESIZE_TECHNIQUES[dim_res_techn] # Thumbnail -> thu etc
    
    resampl_fil_alias = RESAMPLING_FILTERS[resampl_fil] # LANCZOS -> lan etc
    alias_dim_filter = f"{init_jsonl_alias}_{dimension_resize_technique}_{resampl_fil_alias}_{color}" # aokvqa_350_hcd_lan_rgb

    created_folder = f"{alias_dim_filter}_{dimens}" # aokvqa_350_hcd_lan_rgb_01
    created_folder_path = os.path.join(created_img_folder_pth, created_folder) # /srv/muse-lab/datasets/A-OKVQA/images/aokvqa_350_hcd_lan_rgb_01
    created_jsonl_path = os.path.join(created_jsonl_pth, f"{created_folder}.jsonl")
    
    if dim_res_techn == "Hard Coded":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="all",
                                                                 technique="Hard Coded")
    elif dim_res_techn == "Hard Coded Width":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="width",
                                                                 technique="Hard Coded")
    elif dim_res_techn == "Hard Coded Height":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="heigth",
                                                                 technique="Hard Coded")
    elif dim_res_techn == "Both Dimensions Proportionally":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="all",
                                                                 technique="Proportional")
    elif dim_res_techn == "Width Proportionally":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="width",
                                                                 technique="Proportional")
    elif dim_res_techn == "Height Proportionally":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="heigth",
                                                                 technique="Proportional")
    elif dim_res_techn == "Thumbnail":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="all",
                                                                 technique="Thumbnail")
    elif dim_res_techn == "Thumbnail Proportionally":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="all",
                                                                 technique="Thumbnail")
    elif dim_res_techn == "Thumbnail Proportionally Width":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="width",
                                                                 technique="Thumbnail")
    elif dim_res_techn == "Thumbnail Proportionally Height":
        found_resp, statf = hard_coded_or_proportional_technique(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil, dimensions_to_mess="heigth",
                                                                 technique="Thumbnail")
    elif dim_res_techn == "Average":

        # first get the avg dimensions of all the images in the request file
        # _, pixel_tuples = aokvqa_img_details(jsonl_file_path=init_jsonl_request_pth, print_details=True)
        _, pixel_tuples = aokvqa_img_details(jsonl_file_path=init_jsonl_request_pth, print_details=False)
        # those are the new dimensions we want
        created_width = int(sum(pixel_tuples[0])/len(pixel_tuples[0]))
        created_height = int(sum(pixel_tuples[1])/len(pixel_tuples[1]))

        if (created_width, created_height) in list(AVG_DIMENSIONS_DICT.values()):
            dimens_code = find_key_by_value(AVG_DIMENSIONS_DICT, (created_width, created_height))[0]
        else:
            so_far = len(AVG_DIMENSIONS_DICT)
            if so_far > 0:
                so_far = so_far - 1 # start from zero
            if so_far < 10:
                dimens_code = "0" + str(so_far)
            else:
                dimens_code = str(so_far)

            AVG_DIMENSIONS_DICT[dimens_code] = (created_width, created_height)

            with open(AVG_DIMENSION_ALIASES_PATH, "w") as f:
                json.dump(AVG_DIMENSIONS_DICT, f, indent=4)
        
        created_folder = f"{alias_dim_filter}_{dimens_code}" # aokvqa_25_avg_lan_rgb_00
        created_folder_path = os.path.join(created_img_folder_pth, created_folder) # /srv/muse-lab/datasets/A-OKVQA/aokvqa_25_avg_lan_rgb_00
        created_jsonl_path = os.path.join(created_jsonl_pth, f"{created_folder}.jsonl") # /home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/workloads/static/aokvqa_rgb_comprs/aokvqa_25_avg_lan_rgb_00.jsonl

        found_resp, statf = avg_resize(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens=dimens_code, resampl_fil=resampl_fil)

    dictionary_with_workloads[created_folder] = {
        "new_request_jsonl": created_jsonl_path,
        "responses_new": found_resp,
        "stats_new": statf,
        "initial_requests_jsonl": init_jsonl_request_pth
        }

def alias_to_parameter(alias_given, only_percentage_change=False, type="img", keep_only_general_category=False):
    """
    Convert an alias into its corresponding parameter representation.

    This function interprets a given alias and returns the associated parameter string
    representation based on the specified type ('img' or 'video'). For images, it calls
    the `correspond_alias_to_parameter_string` function. For videos, it parses the alias
    to extract and map parameters such as frame sampling technique and thresholds.

    Args:
        alias_given (str): The alias string to be converted into parameters.
        only_percentage_change (bool, optional): Flag to indicate if only percentage 
            changes should be considered. Default is False.
        type (str, optional): The type of alias, either 'img' or 'video'. Default is 'img'.
        keep_only_general_category (bool, optional): Flag to indicate if only the general
            technique category should be kept and the rest should be discarded. Default is
            False.

    Returns:
        str: A string representation of the parameters derived from the alias.
    """
    if alias_given == "00":
        return alias_given
    if type == "img":
        return correspond_alias_to_parameter_string(alias_given, only_percentage_change=only_percentage_change)
    elif type == "video":
        str_ret = ""
        if "_350_" in alias_given:
            find_division = "_350_"
        elif "_10_" in alias_given:
            find_division = "_10_"
        alias_given = alias_given.split(find_division)[-1]

        fram_sampl = alias_given.split("_")[0]  # frs or uni etc
        first_param_alias = alias_given.split("_")[1]
        if fram_sampl not in ["frs", "uni"]:
            second_param_alias = alias_given.split("_")[2]

        if fram_sampl in FRAME_SAMPLING_TECHNIQUE_NAME.keys():
                res_techn_nam = FRAME_SAMPLING_TECHNIQUE_NAME[fram_sampl]   # technique name

        if "uni" in fram_sampl:
            first_param = str(MAX_FRAMES[first_param_alias])
        elif "frs" in fram_sampl:
            first_param = str(FPS[first_param_alias])
        elif "scc" in fram_sampl:
            first_param = str(CONTENT_THRESHOLD[first_param_alias])
        elif "mbd" in fram_sampl:
            first_param = str(MOTION_THRESHOLD[first_param_alias])
        elif "shb" in fram_sampl:
            first_param = str(SHARPNESS_THRESHOLD[first_param_alias])

        if fram_sampl not in "uni":
            str_ret += res_techn_nam + " " + first_param
        else:
            str_ret += res_techn_nam
        if keep_only_general_category:
            return str_ret
        if fram_sampl not in ["frs", "uni"]:
            second_param = str(MAX_FRAMES[second_param_alias])
            str_ret = str_ret + " frames:" + second_param

        return str_ret

def correspond_alias_to_parameter_string(alias_given, only_percentage_change=False):
    # aokvqa_350_avg_lan_rgb_00
    """
    This function takes an alias and returns a string that contains the actual parameter string used when calling the create_resized_img_folder_n_jsonl function.
    The alias is expected to have the following format: aokvqa_350_avg_lan_rgb_00
    The function will return a string like this: 'Average LANCZOS (100,100)'

    only_percentage_change: if True, only the percentage change will be returned
    """

    if alias_given == "00":
        return alias_given

    str_ret = ""
    if "_350_" in alias_given:
        alias_given = alias_given.split("aokvqa_350_")[-1]                                  # avg_lan_rgb_00
    elif "_83_" in alias_given:
        alias_given = alias_given.split("aokvqa_83_")[-1]                                    # avg_lan_rgb_00

    dim_res_alias = alias_given.split("_")[0]                                               # avg
    resample_tec_alias = alias_given.split("_")[1]                                          # lan
    dimensions_dict_key = alias_given.split("_")[-1]                                        # 00

    if only_percentage_change:
        # return str(DIMENSIONS_DICT_PROPORTIONALITY[dimensions_dict_key][0]) + "%"
        return str(DIMENSIONS_DICT_PROPORTIONALITY[dimensions_dict_key][0])

    if dim_res_alias in DIMENSION_RESIZE_TECHNIQUES.values():                               # is it in this dictionary?
        res_techn_nam = find_key_by_value(DIMENSION_RESIZE_TECHNIQUES, dim_res_alias)[0]    # get the actual name ['Hard Coded'] -> 'Hard Coded'
        if dim_res_alias != "avg":
            dim1, dim2 = DIMENSIONS_DICT[dimensions_dict_key]                                   # get the actual dimensions from key '00' -> (100,100)
        else: # for average we take dimensions from the file
            dim1, dim2 = AVG_DIMENSIONS_DICT[dimensions_dict_key]
    else:
        res_techn_nam = find_key_by_value(DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY, dim_res_alias)[0]
        dim1, dim2 = DIMENSIONS_DICT_PROPORTIONALITY[dimensions_dict_key]
    
    str_ret += res_techn_nam
    resample_tec_param = find_key_by_value(RESAMPLING_FILTERS, resample_tec_alias)[0]
    str_ret += " " + resample_tec_param + f" {dim1, dim2}"
    
    return str_ret

def get_dimensions_str_from_work_nam(work_str_whole):
    """
    Takes a workload name string and returns the dimensions as a string.

    Parameters
    ----------
    work_str_whole : str
        The workload name string. Expected to be in the format
        'aokvqa_350_avg_lan_rgb_00'.

    Returns
    -------
    str
        The dimensions as a string, e.g. "(100,100)".
    """
    whole_str = correspond_alias_to_parameter_string(work_str_whole)
    only_dims = "(" + whole_str.split(" (")[-1]
    return only_dims

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
# Videos

def get_video_metadata_from_jsonl_standard(jsonl_filepath: str):
    """
    Reads a standard JSONL file, extracts video paths, and gathers video metadata
    (file size, number of frames, frame dimensions).

    Args:
        jsonl_filepath (str): The path to the JSONL file.

    Returns:
        tuple: A tuple containing lists of metadata:
               - video_ids (list of str): Unique IDs for each entry.
               - video_paths (list of str): Absolute paths to video files.
               - video_kbs (list of float): File size of each video in kilobytes.
               - num_frames_list (list of int): Number of frames in each video.
               - frame_sizes_list (list of tuple): (width, height) of frames for each video.
               - fps_list (list of float): Frames per second of each video.
               - duration_seconds_list (list of float): Duration of each video in seconds.
               - question_texts (list of str): The 'input' (question) text for each entry.
               - correct_answers (list of str): The 'output' (correct answer) for each entry.
    """
    video_ids = []
    video_paths = []
    video_kbs = []
    num_frames_list = []
    frame_sizes_list = [] # Stores (width, height)
    fps_list = [] # Frames per second
    duration_seconds_list = [] # Duration in seconds
    question_texts = []
    correct_answers = []
    modality_size_file_list = []

    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line {line_num + 1}: {e} - {line.strip()}")
                    continue
                
                # The 'request' key contains the dictionary with input, output, id, etc.
                request_data = data.get('request', {})

                # Extract required fields from the 'request' dictionary
                modality_path = request_data.get('modality_path')
                modality_size_file = request_data.get('modality_size')
                entry_id = request_data.get('id')
                question_text = request_data.get('input') 
                correct_answer = request_data.get('output') 

                # Skip if any crucial field is missing
                if not all([modality_path, entry_id, question_text, correct_answer]):
                    print(f"Warning: Skipping line {line_num + 1} due to missing critical fields within 'request': {line.strip()}")
                    continue
                
                video_ids.append(entry_id)
                question_texts.append(question_text)
                correct_answers.append(correct_answer)
                modality_size_file_list.append(modality_size_file)
                
                # Ensure absolute path for video file system access
                full_video_path = os.path.abspath(modality_path) 
                video_paths.append(full_video_path)

                # Get video metadata from the file system and video file
                if os.path.exists(full_video_path):
                    # File size in KB
                    file_size_bytes = os.path.getsize(full_video_path)
                    video_kbs.append(file_size_bytes / 1024)

                    # Using OpenCV to get frames, dimensions, and FPS
                    cap = cv2.VideoCapture(full_video_path)
                    if cap.isOpened():
                        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = float(cap.get(cv2.CAP_PROP_FPS))
                        cap.release()

                        num_frames_list.append(num_frames)
                        frame_sizes_list.append((width, height))
                        fps_list.append(fps)
                        
                        # Calculate duration
                        if fps > 0:
                            duration_seconds_list.append(num_frames / fps)
                        else:
                            print("FPS is 0, cannot determine duration.")
                            duration_seconds_list.append(0.0) # Cannot determine duration if FPS is 0
                    else:
                        print(f"Warning: Could not open video file for metadata: {full_video_path}")
                        num_frames_list.append(0)
                        frame_sizes_list.append((0, 0)) # Indicate failure
                        fps_list.append(0.0)
                        duration_seconds_list.append(0.0)
                else:
                    print(f"Warning: Video file not found: {full_video_path}")
                    video_kbs.append(0.0) # Indicate file not found
                    num_frames_list.append(0)
                    frame_sizes_list.append((0, 0)) # Indicate file not found
                    fps_list.append(0.0)
                    duration_seconds_list.append(0.0)

    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_filepath}")
        return [], [], [], [], [], [], [], [], [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], [], [], [], [], [], [], [], [], []

    return (video_ids, video_paths, video_kbs, num_frames_list,
            frame_sizes_list, fps_list, duration_seconds_list,
            question_texts, correct_answers, modality_size_file_list)

def do_video_fram_sampling(
        vid_create_pth,
        jsonl_create_path, init_jsonl_request_pth,
        strategy, sub_workload,
        model_alias=None):
    """
    Creates a new workload for video frame sampling based on the given strategy and parameters.

    Parameters
    ----------
    vid_create_pth : str
        The path to the new folder of the created workload.
    jsonl_create_path : str
        The path to the JSONL file to create.
    init_jsonl_request_pth : str
        The path to the init JSONL file.
    strategy : str
        The strategy to use for video frame sampling.
    sub_workload : dict
        The dictionary containing the parameters for the given strategy.
    model_alias : str, optional
        The alias of the model to use for the workload creation.

    Returns
    -------
    found_resp : str
        The path to the response file of the workload.
    statf : str
        The path to the statistics file of the workload.
    """
    print(f"Creating workload: {jsonl_create_path}")
    """
    first will check if the vid-mc-0-30_350_uni_0.jsonl for example file exists
    """
    if not os.path.exists(jsonl_create_path):
        requests = get_requests_in_list(init_jsonl_request_pth)

        with open(jsonl_create_path, "w") as fwrite:
            print(f"Writing to: {jsonl_create_path}")
                
            for request in requests:
                item = request # change the modality path of the dict and save it to the new jsonl
                fwrite.write(json.dumps(item) + "\n")
            fwrite.close()

        # Change and try the timing in here because otherwise it erases the previous one
        # if I haven't given get_only_existing
        t0 = time.time()    # also time the workload creation

        # execute the workload to get the responses and statistics - 
        call_script_run_static_img_resizing(new_work_nam = vid_create_pth.split("/")[-1],
                                                type="video_clever_sampling", strategy=strategy, current_par_dict=sub_workload)
        t1 = time.time()
        total_n = t1-t0
        statistics_workload_creation = {"time": total_n}
        add_procedure_resize_stats(vid_create_pth.split("/")[-1], statistics_workload_creation) # -> vid-mc-0-30_30_bdp_lan_rgb_00

    """
    the jsonl_create_path file might exist but now we have to check the model alias
    """
    # return the statistics of the workload
    found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, vid_create_pth.split("/")[-1], model_alias)
    statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_LLAVA_VID, vid_create_pth.split("/")[-1], model_alias)

    if not found_resp:
        """
        follow the same procedure because the response file does not exist for the model we want
        """
        print(f"Error: Could not find the response file for {vid_create_pth}")
        requests = get_requests_in_list(init_jsonl_request_pth)

        with open(jsonl_create_path, "w") as fwrite:
            print(f"Writing to: {jsonl_create_path}")                
            for request in requests:
                item = request # change the modality path of the dict and save it to the new jsonl
                fwrite.write(json.dumps(item) + "\n")
            fwrite.close()
        t0 = time.time()    # also time the workload creation
        call_script_run_static_img_resizing(new_work_nam = vid_create_pth.split("/")[-1],
                                                type="video_clever_sampling", strategy=strategy, current_par_dict=sub_workload)
        t1 = time.time()
        total_n = t1-t0
        statistics_workload_creation = {"time": total_n}
        add_procedure_resize_stats(vid_create_pth.split("/")[-1], statistics_workload_creation) # -> vid-mc-0-30_30_bdp_lan_rgb_00
        found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, vid_create_pth.split("/")[-1], model_alias)
        statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_LLAVA_VID, vid_create_pth.split("/")[-1], model_alias)

    return found_resp, statf

def get_only_existing_video_frame_sampling(vid_create_pth, jsonl_create_path, model_alias=None):
    """
    Checks if the specified JSONL file path exists and returns the associated response and statistics files
    for the given video creation path.

    Parameters
    ----------
    vid_create_pth : str
        The path to the created video workload.
    jsonl_create_path : str
        The path to the JSONL file to check for existence.

    Returns
    -------
    found_resp : str or None
        The path to the response file if the JSONL file exists, otherwise None.
    statf : str or None
        The path to the statistics file if the JSONL file exists, otherwise None.
    """

    # print(f"Looking for workload: {jsonl_create_path}")
    if os.path.exists(jsonl_create_path):
        # return the statistics of the workload
        found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, vid_create_pth.split("/")[-1], model_alias=model_alias)
        statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_LLAVA_VID, vid_create_pth.split("/")[-1], model_alias=model_alias)

        # print(found_resp)
        # print(statf)
        
        return found_resp, statf
    print(f"Error: Could not find the workload file: {jsonl_create_path}")
    return None, None

def create_resized_video_n_jsonl_frame_sampling(init_jsonl_request_pth, strategy, sub_workload,
                                                get_only_existing=False, model_alias=None):
    """
    Creates a new video workload with the specified frame sampling strategy and saves it to disk, 
    together with the associated JSONL file containing the requests. The strategy can be one of 
    "uniform", "fixed_rate", "scene_change", "motion_based", or "sharpness_based". The sub_workload 
    dictionary contains the parameters for the chosen strategy.

    Parameters
    ----------
    init_jsonl_request_pth : str
        The path to the JSONL file containing the initial requests.
    strategy : str
        The frame sampling strategy to use, one of "uniform", "fixed_rate", "scene_change", 
        "motion_based", or "sharpness_based".
    sub_workload : dict
        A dictionary containing the parameters for the chosen strategy.
    get_only_existing : bool, optional
        If True, only return the existing video workload and its associated JSONL file, 
        otherwise create a new workload and JSONL file.
    model_alias : str, optional
        The alias of the model to use for the workload creation.

    Returns
    -------
    dictionary_with_workloads : dict
        A dictionary containing the new video workload, the associated JSONL file, the response 
        file, the statistics file, and the initial requests JSONL file.
    """
    dictionary_with_workloads = {}
    init_jsonl_alias = init_jsonl_request_pth.split("/")[-1].split(".")[0] # /path/vid-mc-0-30_350.jsonl -> vid-mc-0-30_350
    strategy_alias = FRAME_SAMPLING_TECHNIQUE[strategy] # strategy abbreviation "uniform": "uni"

    if strategy == "uniform":     # get first abbreviation
        max_frames = sub_workload["max_frames"] # 4
        created_folder = f"{init_jsonl_alias}_{strategy_alias}_{max_frames}" # vid-mc-0-30_350_uni_4
    elif strategy == "fixed_rate":
        target_fps = sub_workload["target_fps"] # 4
        created_folder = f"{init_jsonl_alias}_{strategy_alias}_{target_fps}" # vid-mc-0-30_350_uni_4
    elif strategy == "scene_change":
        content_threshold = sub_workload["content_threshold"] # 0.5
        max_frames = sub_workload["max_frames"] # 4
        created_folder = f"{init_jsonl_alias}_{strategy_alias}_{content_threshold}_{max_frames}" # vid-mc-0-30_350_uni_4
    elif strategy == "motion_based":
        motion_threshold = sub_workload["motion_threshold"] # 0.5
        max_frames = sub_workload["max_frames"] # 4
        created_folder = f"{init_jsonl_alias}_{strategy_alias}_{motion_threshold}_{max_frames}" # vid-mc-0-30_350_uni_4
    elif strategy == "sharpness_based":
        sharpness_threshold = sub_workload["sharpness_threshold"] # 0.5
        max_frames = sub_workload["max_frames"] # 4
        created_folder = f"{init_jsonl_alias}_{strategy_alias}_{sharpness_threshold}_{max_frames}" # vid-mc-0-30_350_uni_4
    created_folder_path = os.path.join(LLAVA_VID_FOLDER, created_folder) # /srv/muse-lab/datasets/LLaVA-Video/videos/vid-mc-0-30_350_uni_4
    created_jsonl_path = os.path.join(LLAVA_VID_RGB_STATICS, f"{created_folder}.jsonl")

    # print(created_jsonl_path)

    if get_only_existing == False:
        # t0 = time.time()    # also time the workload creation
        found_resp, statf = do_video_fram_sampling(created_folder_path, created_jsonl_path, init_jsonl_request_pth, strategy, sub_workload, model_alias=model_alias)
        # t1 = time.time()
        # total_n = t1-t0
        # statistics_workload_creation = {"time": total_n}
        # add_procedure_resize_stats(created_folder_path.split("/")[-1], statistics_workload_creation) # -> vid-mc-0-30_30_bdp_lan_rgb_00
    else:
        found_resp, statf = get_only_existing_video_frame_sampling(created_folder_path, created_jsonl_path, model_alias=model_alias)
    
    # they don't have responses so far
    if model_alias == "llava-ov-qwen2-0.5b" or model_alias == "llava-ov-qwen2-7b" or model_alias == "pixtral_12b":
        if found_resp is None or statf is None:
            # print("GIOOOOOOO")
            dictionary_with_workloads[created_folder] = {
                "new_request_jsonl": created_jsonl_path,
                "responses_new": found_resp,
                "stats_new": statf,
                "initial_requests_jsonl": init_jsonl_request_pth
                }

    if found_resp and statf: # they are not None
        dictionary_with_workloads[created_folder] = {
            "new_request_jsonl": created_jsonl_path,
            "responses_new": found_resp,
            "stats_new": statf,
            "initial_requests_jsonl": init_jsonl_request_pth
            }
    return dictionary_with_workloads
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
# A - OKVQA images

def display_matplot_lib_img(pth):
    """
    Displays an image using matplotlib's imshow function and prints its size in KiloBytes.

    Parameters
    ----------
    pth : str
        The path to the image file to be displayed.

    Returns
    -------
    None
    """
    img = Image.open(pth)
    plt.imshow(img)
    plt.show()
    print("Image KiloBytes:", file_size(pth, convert_to = "KB"))

def display_img_given_path(img_path):
    """
    Displays the image at the given path.

    Parameters
    ----------
    img_path : str
        The path to the image file to be displayed.

    Returns
    -------
    None
    """
    img = Image.open(img_path)
    print(img_path)
    img.show()

def get_img_dimensions_given_path(img_path):
    """
    Returns the dimensions of the image at the given path.

    Parameters
    ----------
    img_path : str
        The path to the image file.

    Returns
    -------
    tuple
        A tuple containing the width and height of the image in pixels.
    """
    img = Image.open(img_path)
    return img.size

def aokvqa_img_details(jsonl_file_path, print_details=True):
    """
    Prints and returns the details of the images in the AOKVQA JSONL file.

    Parameters
    ----------
    jsonl_file_path : str
        The path to the JSONL file containing the AOKVQA requests.
    print_details : bool
        Whether to print the details of the images. Defaults to True.

    Returns
    -------
    tuple
        A tuple containing:
        - The list of image sizes in KB (list of float)
        - The list of image sizes in pixels (list of tuple of int)
    """
    img_KBS = []
    # img_MBs = []
    # first list is the width, second is the height
    pixel_tuples = [[], []]

    with open(jsonl_file_path, "r") as f:
        requests = [json.loads(line) for line in f]
        
        for request in requests:
            if request["request"]["modality_path"] is None:
                print("Has no modality path", request["request"]["id"])
            else:
                pixel_tuples[0].append(request["request"]["modality_size"][0])
                pixel_tuples[1].append(request["request"]["modality_size"][1])
                img_KBS.append(file_size(request["request"]["modality_path"], convert_to = "KB"))
                # img_MBs.append(file_size(request["request"]["modality_path"], convert_to = "MB"))
        f.close()

    if print_details:
        print(len(img_KBS), "\t", max(img_KBS), "\t", min(img_KBS), "\t", sum(img_KBS)/len(img_KBS))
        # print(len(img_MBs), "\t", max(img_MBs), "\t\t", min(img_MBs), "\t\t", sum(img_MBs)/len(img_MBs))
        print(len(pixel_tuples[0]), "\t", max(pixel_tuples[0]), "\t\t", min(pixel_tuples[0]), "\t\t", sum(pixel_tuples[0])/len(pixel_tuples[0]))
        print(len(pixel_tuples[1]), "\t", max(pixel_tuples[1]), "\t\t", min(pixel_tuples[1]), "\t\t", sum(pixel_tuples[1])/len(pixel_tuples[1]))
        print()
    return img_KBS, pixel_tuples

# Used to create subsets of workloads I want to use in order to try plots
def workload_keys_by_filters(resampling_filters=[],
                             dimension_resize_techniques=[],
                             dimension_resize_techniques_proportions=[],
                             dimensions_from_dict=[],
                             dimensions_from_dict_proportions=[],
                             dictionary_workload=dictionary_with_workloads
):
    
    """
    Function to filter workload keys based on multiple criteria.

    Parameters
    ----------
    resampling_filters : list of str
        The list of resampling filters to use.
    dimension_resize_techniques : list of str
        The list of dimension resize techniques to use.
    dimension_resize_techniques_proportions : list of str
        The list of dimension resize techniques with proportional resize.
    dimensions_from_dict : list of tuple of int
        The list of dimensions to use from the DIMENSIONS_DICT.
    dimensions_from_dict_proportions : list of tuple of int
        The list of dimensions to use from the DIMENSIONS_DICT_PROPORTIONALITY.
    dictionary_workload : dict
        The dictionary containing the workloads.

    Returns
    -------
    list of str
        The list of workload keys that satisfy the criteria.
    """
    
    keep_workload_keys = []
    if dimension_resize_techniques_proportions == []:
        for work_keys in dictionary_workload:
            if any(RESAMPLING_FILTERS[resampling_filter] in work_keys for resampling_filter in resampling_filters) or resampling_filters == []:
                if any(DIMENSION_RESIZE_TECHNIQUES[dimension_resize_technique] in work_keys for dimension_resize_technique in dimension_resize_techniques)\
                    or dimension_resize_techniques == []:
                    if any(
                        [find_key_by_value(d=DIMENSIONS_DICT, target_value=dimensions_from_dict_in)[0] in work_keys for dimensions_from_dict_in in dimensions_from_dict]
                        ) or dimensions_from_dict == []:
                        keep_workload_keys.append(work_keys)
    else:
        for work_keys in dictionary_workload:
            if any(RESAMPLING_FILTERS[resampling_filter] in work_keys for resampling_filter in resampling_filters) or resampling_filters == []:
                if any(DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY[dimension_resize_technique] in work_keys for dimension_resize_technique in dimension_resize_techniques_proportions)\
                    or dimension_resize_techniques_proportions == []:
                    if any(
                        [find_key_by_value(d=DIMENSIONS_DICT_PROPORTIONALITY, target_value=dimensions_from_dict_in)[0] in work_keys for dimensions_from_dict_in in dimensions_from_dict_proportions]
                        ) or dimensions_from_dict_proportions == []:
                        keep_workload_keys.append(work_keys)
    
    return keep_workload_keys

def hard_coded_or_proportional_technique_video(vid_create_pth,
    jsonl_create_path,
    init_jsonl_request_pth, dimens, resampl_fil,
    dimensions_to_mess="all", technique="Hard Coded"):
    """
    Processes video resizing based on a specified technique and saves the resized videos 
    along with updated metadata.

    Parameters
    ----------
    vid_create_pth : str
        The path where the resized video will be saved.
    jsonl_create_path : str
        The path to the JSONL file where requests and new video metadata will be stored.
    init_jsonl_request_pth : str
        The initial JSONL file path containing the video processing requests.
    dimens : str
        The dimension identifier for resizing the video.
    resampl_fil : str
        The resampling filter to use during resizing.
    dimensions_to_mess : str, optional
        The scope of dimensions to modify in the video, default is "all".
    technique : str, optional
        The resizing technique to use, either "Hard Coded" or "Proportional", default is "Hard Coded".

    Returns
    -------
    found_resp : str
        The path to the response file containing the results of the resizing operation.
    statf : str
        The path to the statistics file associated with the resizing operation.
    """

    if create_folder_if_no_exists(vid_create_pth):
        requests = get_requests_in_list(init_jsonl_request_pth)

        if technique=="Proportional":
            created_width, created_height = DIMENSIONS_DICT_PROPORTIONALITY_VID[dimens] # 00 -> (200,300)
        
        t0 = time.time()
        with open(jsonl_create_path, "w") as fwrite:
            
            for request in requests:

                vid_pth = get_value_from_aokvqa_request(request, "modality_path", is_request=True)
                vid_save_path = vid_create_pth + "/academic_source" + vid_pth.split("/academic_source")[-1]

                # Ensure output directory exists
                os.makedirs(os.path.dirname(vid_save_path), exist_ok=True)

                # Read original video size using OpenCV
                cap = cv2.VideoCapture(vid_pth)
                original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                # Compute new dimensions
                new_width = int(original_width * (1 - created_width / 100))
                new_height = int(original_height * (1 - created_height / 100))

                # Round down to the nearest even number
                new_width = new_width - new_width % 2
                new_height = new_height - new_height % 2

                if technique=="Proportional":
                    if dimensions_to_mess == "all":
                        # FFmpeg scaling expression
                        scale_expr = f"scale={new_width}:{new_height}:flags=lanczos"
                
                if not os.path.exists(vid_save_path):
                    # Run FFmpeg with the calculated expression
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-loglevel", "error",
                        "-i", vid_pth,
                        "-vf", scale_expr,
                        vid_save_path
                    ])

                item = request # change the modality path of the dict and save it to the new jsonl
                item["request"].update({'modality_path': vid_save_path})
                fwrite.write(json.dumps(item) + "\n")

            fwrite.close()
        
        t1 = time.time()
        total_n = t1-t0

        statistics_workload_creation = {"time": total_n}
        # print(vid_create_pth.split("/")[-1])
        add_procedure_resize_stats(vid_create_pth.split("/")[-1], statistics_workload_creation) # -> vid-mc-0-30_30_bdp_lan_rgb_00

        # execute the workload to get the responses and statistics
        call_script_run_static_img_resizing(new_work_nam = vid_create_pth.split("/")[-1],
                                            type="video")

        # save the pixels of the new videos
        requests = get_requests_in_list(jsonl_create_path)
        with open(jsonl_create_path, "w") as fwrite:
            for item in requests:
                vid_save_path = item["request"]["modality_path"]
                cap = cv2.VideoCapture(vid_save_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                item["request"]["pixels"] = [width, height]
                fwrite.write(json.dumps(item) + "\n")
            fwrite.close()
    
    # return the statistics of the workload
    found_resp = find_file_in_folder_with_str_occurence(LLM_RESPONSES_RGB_AOKVQA, vid_create_pth.split("/")[-1])
    statf = find_file_in_folder_with_str_occurence(OUTPUTS_350_FOLDER_LLAVA_VID, vid_create_pth.split("/")[-1])

    return found_resp, statf

def create_resized_video_n_jsonl(init_jsonl_request_pth, resampl_fil, dim_res_techn, dimens, color):
    """
    Creates a resized video workload and JSONL metadata based on specified parameters.

    This function processes the initial JSONL request path to generate a resized video according 
    to the given resampling filter, dimension resize technique, dimensions, and color. It creates 
    a new folder for the resized video and saves the corresponding JSONL file with updated metadata.

    Parameters
    ----------
    init_jsonl_request_pth : str
        The path to the initial JSONL file containing the requests for video processing.
    resampl_fil : str
        The resampling filter to apply during video resizing.
    dim_res_techn : str
        The dimension resize technique to use, which can influence whether resizing is proportional.
    dimens : str
        The dimension identifier for resizing the video.
    color : str
        The color format of the video, e.g., RGB.

    Returns
    -------
    None
        The function updates the global dictionary_with_workloads with keys for the new 
        request JSONL, responses, statistics, and the initial request JSONL.
    """

    init_jsonl_alias = init_jsonl_request_pth.split("/")[-1].split(".")[0] # /path/vid-mc-0-30_350.jsonl -> vid-mc-0-30_350

    # is it for proportional or no?
    if dim_res_techn in DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY_VID.keys():
        dimension_resize_technique = DIMENSION_RESIZE_TECHNIQUES_PROPORTIONALITY_VID[dim_res_techn] # "Both Dimensions Proportionally": "bdp"

    resampl_fil_alias = RESAMPLING_FILTERS_VID[resampl_fil] # LANCZOS -> lan etc
    alias_dim_filter = f"{init_jsonl_alias}_{dimension_resize_technique}_{resampl_fil_alias}_{color}" # vid-mc-0-30_350_bdp_lan_rgb

    created_folder = f"{alias_dim_filter}_{dimens}" # vid-mc-0-30_350_bdp_lan_rgb_00
    created_folder_path = os.path.join(LLAVA_VID_FOLDER, created_folder) # /srv/muse-lab/datasets/LLaVA-Video/videos/vid-mc-0-30_350_bdp_lan_rgb_00
    created_jsonl_path = os.path.join(LLAVA_VID_RGB_STATICS, f"{created_folder}.jsonl")

    if dim_res_techn == "Both Dimensions Proportionally":
        found_resp, statf = hard_coded_or_proportional_technique_video(created_folder_path, created_jsonl_path, init_jsonl_request_pth, dimens, resampl_fil,
                                                                       dimensions_to_mess="all",
                                                                 technique="Proportional")
    dictionary_with_workloads[created_folder] = {
        "new_request_jsonl": created_jsonl_path,
        "responses_new": found_resp,
        "stats_new": statf,
        "initial_requests_jsonl": init_jsonl_request_pth
        }
####################################################################################################################################
####################################################################################################################################
# plot functions

def get_cdf(data):
    """
    Compute the cumulative distribution function (CDF) of a given dataset.

    Parameters
    ----------
    data : array-like
        The input data.

    Returns
    -------
    x : array-like
        The sorted input data.
    y : array-like
        The cumulative probabilities of the input data.
    """
    N = len(data)

    x = np.sort(data)
    y = np.arange(N) / float(N)

    return x, y

def plot_raw_stats(aokvqa_workloads, stat_list, metric_chosen):
    """
    Plot raw statistics for each AOK-VQA workload.

    This function creates a line plot of the raw statistics for each workload
    provided in the `aokvqa_workloads` dictionary. The average value of each
    statistic is printed in the console.

    Parameters
    ----------
    aokvqa_workloads : dict
        A dictionary with the names of the AOK-VQA workloads as keys and dictionaries
        containing the paths to the JSON Lines files of the requests as values.
    stat_list : list of lists
        A list of lists containing the values of the given metric for each AOK-VQA workload.
    metric_chosen : str
        The name of the metric to plot.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)
    # iterate the stat files
    for (idx, value) in enumerate(aokvqa_workloads):

        plt.plot(stat_list[idx], label=value, color=plot_colors_by_num[idx], linewidth=param_dictionary["line_width"])
        print("avg", value, "stat:", np.mean(stat_list[idx]))
    
    plt.ylabel(metric_chosen, size=param_dictionary["ylabel_size"])
    plt.xlabel("Request", size=param_dictionary["xlabel_size"], x=0.37)
    
    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)
    plt.show()

def plot_entropy_CDF(aokvqa_workloads):
    """
    Plot the CDF of the entropy of the images in the given list of AOK-VQA workloads.

    Parameters
    ----------
    aokvqa_workloads : dict
        A dictionary with the names of the AOK-VQA workloads as keys and dictionaries
        containing the paths to the JSON Lines files of the requests as values.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for (idx, workload_chosen) in enumerate(aokvqa_workloads):
        with open(aokvqa_workloads[workload_chosen]["requests_file"], "r") as f:
            modality_paths = [json.loads(line)["request"]["modality_path"] for line in f]
            f.close()
        
        entropy_list = []
        for modality_path in modality_paths:
            img_skimage = skimage.io.imread(modality_path)
            entropy_list.append(skimage.measure.shannon_entropy(img_skimage))
        
        values, base = np.histogram(entropy_list, bins=40)
        cumulative = np.cumsum(values) # Cumulative sum of values
        plt.plot(base[:-1], cumulative / cumulative[-1], label=workload_chosen, c=plot_colors_by_num[idx], linewidth=param_dictionary["line_width"])  # Normalize to get probabilities
    
    plt.ylabel("Probability", size=param_dictionary["ylabel_size"])
    plt.xlabel("Entropy", size=param_dictionary["xlabel_size"], x=0.37)
    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)
    plt.show()

def cdf_stat(aokvqa_workloads, metric_chosen, metric_dic, accuracy_dict=None):
    """
    Plots the cumulative distribution function (CDF) of a given metric for a set of workloads.

    Parameters
    ----------
    aokvqa_workloads : list
        List of the names of the A - OKVQA workloads.
    metric_chosen : str
        The name of the metric to plot.
    metric_dic : dict
        Dictionary containing the values of the metric for each workload.
    accuracy_dict : dict, optional
        Dictionary containing the accuracy of each workload. Defaults to None.

    Returns
    -------
    None
    """
    is_dual_plot = accuracy_dict is not None

    # Create subplots
    if is_dual_plot:
        fig, (ax_cdf, ax_acc) = plt.subplots(
            1, 2, figsize=(param_dictionary["figsize_mul"]*10, 4.8), 
            constrained_layout=True, gridspec_kw={'width_ratios': [3, 1]}
        )
    else:
        fig, ax_cdf = plt.subplots(
            figsize=[param_dictionary["figsize_mul"]*6.4, 4.8],
            constrained_layout=True
        )
    
    # --- percentage difference of the average of the metrics ---
    measures_dict_dif = {}
    for workload_chosen in aokvqa_workloads:
        if "original" or "common_shapes_225" in workload_chosen:
            basecase = metric_dic[workload_chosen]
            basecase_acc = accuracy_dict[workload_chosen]["acc"]
            basecase_mean = np.mean(basecase)
    for workload_chosen in aokvqa_workloads:
        if "original" not in workload_chosen:
            acc_diff = accuracy_dict[workload_chosen]["acc"] - basecase_acc
            acc_diff_percent = acc_diff / basecase_acc * 100
            
            diff = metric_dic[workload_chosen]
            diff_mean = np.mean(diff)
            
            diff_percent = (diff_mean - basecase_mean) / basecase_mean * 100
            measures_dict_dif[workload_chosen] = {"metric_diff": diff_percent, "acc_diff": acc_diff_percent}
    # --- percentage difference of the average of the metrics ---

    # --- CDF Plot ---
    avg_measure_drop = []
    avg_acc_drop = []
    for idx, workload_chosen in enumerate(aokvqa_workloads):
        if not "original" in workload_chosen:
            # if measures_dict_dif[workload_chosen]["acc_diff"] > -8:
                
                meas_dif = measures_dict_dif[workload_chosen]["metric_diff"]
                acc_dif = measures_dict_dif[workload_chosen]["acc_diff"]
                print(f"{workload_chosen}: {meas_dif:.2f} % and acc: {acc_dif:.2f}%")

                avg_measure_drop.append(meas_dif)
                avg_acc_drop.append(acc_dif)
                
                # metric_dic contains as keys the names of the workloads and as values the values of the metric in lists
                values, base = np.histogram(metric_dic[workload_chosen], bins=40)
                cumulative = np.cumsum(values)

                ax_cdf.plot(
                    base[:-1], cumulative / cumulative[-1],
                    # label=workload_chosen,
                    label=workload_chosen.split("res_225_")[-1].split("_100")[0],
                    c=plot_colors_by_num[idx],
                    linewidth=param_dictionary["line_width"]
                )
        else:
            values, base = np.histogram(metric_dic[workload_chosen], bins=40)
            cumulative = np.cumsum(values)

            ax_cdf.plot(
                base[:-1], cumulative / cumulative[-1],
                label=workload_chosen,
                c=plot_colors_by_num[idx],
                linewidth=param_dictionary["line_width"]
            )

    print(f"Average percentage difference in {metric_chosen} and accuracy: {np.mean(avg_measure_drop):.2f} % and {np.mean(avg_acc_drop):.2f} %")
    
    ax_cdf.set_ylabel("Probability", size=param_dictionary["ylabel_size"])
    ax_cdf.set_xlabel(metric_chosen, size=param_dictionary["xlabel_size"])

    # --- Accuracy Plot (Optional) ---
    if is_dual_plot:
        for idx, workload_name in enumerate(aokvqa_workloads):
            acc = accuracy_dict[workload_name]["acc"]
            ax_acc.plot(0, acc, 'o', markersize=10, color=plot_colors_by_num[idx])
        ax_acc.set_xlim(-1, 1)
        ax_acc.set_xticks([])
        ax_acc.set_ylabel("Accuracy")

    # --- Shared Legend ---
    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)
    
    plt.show()

def cdf_plot_4_modalities(exec_stats, value_collect="TTFT"):
    """
    Plot the CDF of a specific value for each experiment.

    Parameters
    ----------
    exec_stats : list
        A list of dictionaries, where each dictionary contains the following:
        - "label": str, the name of the category (e.g., Text, Audio, Video, Image)
        - "color": str, the color for the experiment in the plot
        - "exp_out": ExperimentOutput, the experiment output
    value_collect : str, optional
        The type of value to collect from the experiment output. Default is "TTFT".

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for static_iso_inst in exec_stats:

        all_req_outputs = static_iso_inst["exp_out"].request_outputs
        if value_collect=="TTFT":
            list_values = [record.ttft for record in all_req_outputs]
        if value_collect=="E2E":
            list_values = [record.e2e for record in all_req_outputs]
        # if value_collect=="PROMPT_TOKENS_COUNT":
        #     list_values = [record.prompt_tokens_cnt for record in all_req_outputs]
        if value_collect=="MODALITY_TOKENS_COUNT":
            list_values = [record.modality_tokens_cnt for record in all_req_outputs]
        # elif value_collect=="Output":

        # Plot line
        xi, yi = get_cdf(list_values)
        plt.plot(xi, yi, label=static_iso_inst["label"], color=static_iso_inst["color"], linewidth=param_dictionary["line_width"])

        # print(static_iso_inst['filenam'], np.average(xi))

    plt.ylabel("Probability (%)", size=param_dictionary["ylabel_size"])
    # plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"])
    # plt.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])

    # plt.xlabel("Input + Output Length (# tokens)", size=xlabel_size, x=0.37
    plt.xlabel(f"{value_collect}", size=param_dictionary["xlabel_size"], x=0.37)
    
    plt.xscale("log", base=10, subs=[])
    
    # plt.xticks([0, 1, 10], ["0", r"$10^0$", r"$10^1$"])
    # plt.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    plt.legend(loc='best', fontsize=param_dictionary["legend_font_size"])

    plt.show()
    return fig

def in_out_token_length(stats_list):
    """
    Plot two CDFs per experiment: one for the prompt/input tokens and one for the output tokens.
    The x-axis is log-scaled and the y-axis is the probability of having a certain number of tokens
    in the input/output.

    Parameters
    ----------
    stats_list : list
        A list of dictionaries, where each dictionary contains the following:
        - "filenam": str, the name of the file
        - "exp_out": ExperimentOutput, the experiment output
        - "color": str, the color for the experiment in the plot
        - "label": str, the label for the experiment in the plot

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    scale = 2
    # fig, axs = plt.subplots(1, 2, figsize=[scale*6.4, 4.8], constrained_layout=True)
    fig, axs = plt.subplots(1, 2, figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    ax0 = axs.flat[0]
    ax1 = axs.flat[1]

    for static_iso_inst in stats_list:
        
        all_req_outputs = static_iso_inst["exp_out"].request_outputs
        print("SOS: It uses both modality and prompt tokens")
        # prompt_tokens_cnt_list = [record.prompt_tokens_cnt for record in all_req_outputs]
        prompt_tokens_cnt_list = [record.modality_tokens_cnt for record in all_req_outputs]
        decode_tokens_cnt_list = [record.decode_tokens_cnt for record in all_req_outputs]
        
        # Plot line
        xi, yi = get_cdf(prompt_tokens_cnt_list)
        xo, yo = get_cdf(decode_tokens_cnt_list)

        ax0.plot(xi, yi, label=static_iso_inst["label"], color=static_iso_inst["color"], linewidth=4)
        ax1.plot(xo, yo, color=static_iso_inst["color"], linewidth=4)

        # # Plot mean values
        # xi_mean, yi_mean = find_mean_coord(xi, yi)
        # xo_mean, yo_mean = find_mean_coord(xo, yo)
        # # ax0.plot(xi_mean, yi_mean, marker="o", markersize=8, color=meta["color"])
        # # ax1.plot(xo_mean, yo_mean, marker="o", markersize=8, color=meta["color"])

    # Plot median line
    # xmin, xmax = ax0.get_xlim()
    # ax0.hlines(y=0.5, xmin=xmin, xmax=xmax, colors="#999999", ls="--", linewidth=2,
    #        alpha=0.8)

    ax0.set_ylabel("Probability (%)", size=param_dictionary["axis_title_size"])
    ax0.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax0.set_yticklabels(["0 ", "25", "50", "75", "100"])
    ax0.tick_params(axis="y", labelsize=param_dictionary["label_size"])

    ax0.set_xlabel("Input Length (# of tokens)", size=param_dictionary["axis_title_size"])
    ax0.set_xscale("log", base=10, subs=[])
    ax0.set_xticks([1, 10, 100, 1000, 10000, 12000])
    ax0.set_xticklabels(["", "10", r"$10^2$", r"$10^3$", r"$10^4$", ""])
    ax0.tick_params(axis="x", labelsize=param_dictionary["label_size"])

    # xmin, xmax = ax1.get_xlim()
    # ax1.hlines(y=0.5, xmin=xmin, xmax=xmax, colors="#999999", ls="--", linewidth=2,
    #        alpha=0.8)

    ax1.set_yticks([])
    ax1.set_xlabel("Output Length (# of tokens)", size=param_dictionary["axis_title_size"])
    ax1.set_xscale("log", base=10, subs=[])
    ax1.set_xticks([1, 10, 100, 1000, 10000, 12000])
    ax1.set_xticklabels(["", "10", r"$10^2$", r"$10^3$", r"$10^4$", ""])
    ax1.tick_params(axis="x", labelsize=param_dictionary["label_size"])

    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )

    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)

    plt.show()
    return fig

def input_output_token_hist(prompt_tokens_cnts, modality_tokens_cnt, decode_tokens_cnt, bins=40, title="", xlabtit="", ylabtit="", show=True):
    
    """
    Plot histograms for input text tokens, input modality tokens, and output tokens.

    This function creates a figure with three subplots, each displaying a histogram of 
    the token counts for different categories: input text, input modality, and output. 
    The histograms are shown side-by-side in a single row.

    Parameters
    ----------
    prompt_tokens_cnts : list
        A list of token counts for input text prompts.
    modality_tokens_cnt : list
        A list of token counts for input modalities.
    decode_tokens_cnt : list
        A list of token counts for decoded outputs.
    bins : int, optional
        Number of bins to use in the histograms. Default is 40.
    title : str, optional
        The overall title of the figure.
    xlabtit : str, optional
        The x-axis label for the entire figure.
    ylabtit : str, optional
        The y-axis label for each subplot.
    show : bool, optional
        Whether to display the figure. If False, the figure is closed after creation. Default is True.

    Returns
    -------
    fig : Figure
        The matplotlib figure object containing the histograms.
    """

    fig, axs = plt.subplots(nrows=1, ncols=3,
                           figsize=[param_dictionary["figsize_mul"]*6.4, 4.8],
                        #    figsize=(3 * param_dictionary["mul_col_size"], 1 * param_dictionary["mul_row_size"]),
                           constrained_layout=True
                           )
    
    # Flatten axs if multiple subplots exist, otherwise treat it as a single plot
    axs = np.array(axs).flatten() if isinstance(axs, np.ndarray) else [axs]

    # axs[0].hist(prompt_tokens_cnts, color='lightgreen', ec='black', edgecolor = "black", bins=bins)
    axs[0].hist(prompt_tokens_cnts, color='lightgreen', ec="black", bins=bins)
    axs[0].set_title("Input Text Tokens")
    axs[0].set_ylabel(ylabtit, size=param_dictionary["ylabel_size"])
    
    axs[1].hist(modality_tokens_cnt, color='lightgreen', edgecolor = "black", bins=bins)
    axs[1].set_title("Input Modality Tokens")
    
    axs[2].hist(decode_tokens_cnt, color='lightgreen', edgecolor = "black", bins=bins)
    axs[2].set_title("Oututput Tokens")

    fig.suptitle(title, size=param_dictionary["title_size"])
    fig.supxlabel(xlabtit, size=param_dictionary["xlabel_size"])
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def simple_hist(values_init, title="", xlabtit="", ylabtit=""):
    """
    Plot a simple histogram of the given values.

    Parameters
    ----------
    values_init : list or array
        The input values
    title : str, optional
        The title of the plot
    xlabtit : str, optional
        The x-axis label
    ylabtit : str, optional
        The y-axis label. Default is empty str

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    plt.hist(values_init, color='lightgreen', ec='black', edgecolor = "black", bins=20)
    
    plt.title(title, size=param_dictionary["title_size"])
    plt.xlabel(xlabtit, size=param_dictionary["xlabel_size"], x=0.37)
    plt.ylabel(ylabtit, size=param_dictionary["ylabel_size"])
    
    plt.show()
    return fig

def latency_stacked_bar_plot(values_init, ttfts, encoder_times, processor_times):
    """
    Create a stacked bar plot visualizing latency components for different modalities.

    Parameters
    ----------
    values_init : list
        A list of dictionaries, where each dictionary contains the following:
        - "label": str, the name of the category (e.g., Text, Audio, Video, Image)
    ttfts : list
        A list of latency values for the LLM component.
    encoder_times : list
        A list of latency values for the encoder component.
    processor_times : list
        A list of latency values for the preprocess component.

    Returns
    -------
    fig : Figure
        The matplotlib figure object representing the stacked bar plot.
    """

    x = np.arange(len([category["label"] for category in values_init]))  # x locations for each category

    # Width of each bar and offset for each percentile within each category
    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    # Plot each stack
    ax.bar(x, ttfts, param_dictionary["bar_width"], label='LLM', color="#C9DD84")
    ax.bar(x, encoder_times, param_dictionary["bar_width"], bottom=processor_times, label='Encoder', color="#859F3D")
    ax.bar(x, processor_times, param_dictionary["bar_width"], label='Preprocess', color="#31511E")

    # Add labels, title, and legend
    ax.set_ylabel('Latency (s)', size=param_dictionary["ylabel_size"])
    ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])

    ax.set_xticks(x)
    ax.set_xticklabels(["Text", "Audio", "Video", "Image"])
    ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])

    # Custom legend to avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=param_dictionary["legend_font_size"], loc='upper left')

    plt.show()
    return fig

def kv_cache_subworkloads(aokvqa_workloads):
    """
    Plot the CDF of the KV Cache footprint (#tokens) for each experiment.
    The x-axis is log-scaled and the y-axis is the probability of having a certain KV Cache footprint (#tokens).
    The legend is placed at the best position.

    Parameters
    ----------
    aokvqa_workloads : dict
        A dictionary with the names of the AOK-VQA workloads as keys and dictionaries
        containing the paths to the JSON Lines files of the requests as values.

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    figsize_mul = 1.14
    ylabel_size = 29
    xlabel_size = 29
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    legend_font_size = 14
    line_width = 4
    
    fig = plt.figure(figsize=[figsize_mul*6.4, figsize_mul*4.8], constrained_layout=True)

    for idx, workload_chosen in enumerate(aokvqa_workloads):
        
        request_stat = []
        with open(aokvqa_workloads[workload_chosen]["stats_file"], "r") as f:
            # collect all the stat values about the stat file
            for line in f:
                whole_req = json.loads(line)
                print("It isn't used anywhere")
                # request_stat.append(whole_req["decode_tokens_cnt"] + whole_req["prompt_tokens_cnt"])
            f.close()
        
        xi, yi = get_cdf(request_stat)
        plt.plot(xi, yi, label=workload_chosen, color=plot_colors_by_num[idx], linewidth=line_width)
    
    plt.ylabel("Probability (%)", size=ylabel_size)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"])
    plt.tick_params(axis="y", labelsize=y_params_label_size)

    plt.xlabel("KV Cache footprint (#tokens)", size=xlabel_size, x=0.37)
    plt.xscale("log", base=10, subs=[])
    plt.xticks([10, 100, 1000, 10000, 12000], ["10", r"$10^2$", r"$10^3$", r"$10^4$", ""])
    plt.tick_params(axis="x", labelsize=x_params_label_size)
    # plt.legend(loc='best', fontsize=legend_font_size)
    # plt.legend(loc='lower center', fontsize=legend_font_size)
    # plt.legend(loc='upper center', fontsize=legend_font_size)

    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)

    plt.show()
    return fig

def kv_cache_footprint(stats_list):
    """
    Plot the CDF of the KV Cache footprint (#tokens) for each experiment.
    The x-axis is log-scaled and the y-axis is the probability of having a certain KV Cache footprint (#tokens).
    The legend is placed at the best position.

    Parameters
    ----------
    stats_list : list
        A list of dictionaries, where each dictionary contains the following:
        - "filenam": str, the name of the file
        - "exp_out": ExperimentOutput, the experiment output
        - "color": str, the color for the experiment in the plot
        - "label": str, the label for the experiment in the plot

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    figsize_mul = 1.04
    ylabel_size = 29
    xlabel_size = 29
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    legend_font_size = 23
    line_width = 4

    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)

    print(f"\n\nWATCH OUT IT USES BOTH MODALITY AND PROMPT TOKENS\n\n")
    for static_iso_inst in stats_list:

        all_req_outputs = static_iso_inst["exp_out"].request_outputs
        in_plus_out_list = [record.decode_tokens_cnt + record.prompt_tokens_cnt for record in all_req_outputs]

        xi, yi = get_cdf(in_plus_out_list)
        plt.plot(xi, yi, label=static_iso_inst["label"], color=static_iso_inst["color"], linewidth=line_width)

    
    # plt.ylabel("Cumulative Probability (%)", size=ylabel_size)
    # plt.ylabel(r'Probability $\le x$ ($\%$)', size=ylabel_size)
    plt.ylabel(r'Probability ≤ x (%)', size=ylabel_size)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"])
    plt.tick_params(axis="y", labelsize=y_params_label_size)

    # plt.xlabel("Input + Output Length (# tokens)", size=xlabel_size, x=0.37
    plt.xlabel("KV Cache footprint (#tokens)", size=xlabel_size, x=0.37
            #    , y=0.5
            )
    plt.xscale("log", base=10, subs=[])
    plt.xticks([10, 100, 1000, 10000, 12000], ["10", r"$10^2$", r"$10^3$", r"$10^4$", ""])
    plt.tick_params(axis="x", labelsize=x_params_label_size)
    plt.legend(loc='best', fontsize=legend_font_size)

    plt.show()
    return fig

def simple_value_plot(values, title="", xlabtit="", ylabtit=""):
    """
    Plot the given values as a line plot.

    Parameters
    ----------
    values : list or array
        The input values
    title : str, optional
        The title of the plot
    xlabtit : str, optional
        The x-axis label
    ylabtit : str, optional
        The y-axis label

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)
    
    plt.title(title, size=param_dictionary["title_size"])

    plt.xlabel(xlabtit, size=param_dictionary["xlabel_size"], x=0.37)
    plt.ylabel(ylabtit, size=param_dictionary["ylabel_size"])
    
    plt.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
    plt.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    
    plt.plot(values, linewidth=param_dictionary["line_width"])
    
    plt.show()
    return fig

def simple_cdf_plot(values_init, title="", xlabtit="", ylabtit="Probability"):
    
    """
    Plot a simple CDF plot of the given values.

    Parameters
    ----------
    values_init : list or array
        The input values
    title : str, optional
        The title of the plot
    xlabtit : str, optional
        The x-axis label
    ylabtit : str, optional
        The y-axis label. Default is "Probability"

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """
    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    values, base = np.histogram(values_init, bins=40)
    cumulative = np.cumsum(values) # Cumulative sum of values
    
    plt.title(title, size=param_dictionary["title_size"])
    plt.xlabel(xlabtit, size=param_dictionary["xlabel_size"], x=0.37)
    plt.ylabel(ylabtit, size=param_dictionary["ylabel_size"])
    
    # # This effectively shows the complement of the CDF (also called the survival function or 1 - CDF), not the CDF itself
    # plt.plot(base[:-1], len(values_init)-cumulative, c='green')

    # To correctly plot the true CDF with probability values on the y-axis, you should
    # normalize the cumulative values by dividing by the total number of data points
    plt.plot(base[:-1], cumulative / cumulative[-1], c='green')  # Normalize to get probabilities

    plt.show()
    return fig

def kv_cache_footprint_comparison(lists_of_stats, categories_name):
    """
    Plot a comparison of the CDFs for different categories of KV Cache footprints.

    This function visualizes the cumulative distribution functions (CDF) of the KV Cache
    footprint (in terms of #tokens) for multiple categories. It also calculates and displays
    the percentage difference in the average KV Cache footprint between the first two categories.

    Parameters
    ----------
    lists_of_stats : list of lists
        A list where each element is a list of dictionaries. Each dictionary contains:
        - "modality_tokens_cnt": int, the number of modality tokens
    categories_name : list
        A list of strings representing the names of the categories to be used as labels in the plot.

    Returns
    -------
    None
    """

    figsize_mul = 1.04
    ylabel_size = 29
    xlabel_size = 29
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    line_width = 4

    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)

    mean_vals = []  # for percentage change of average value
    for idx, list_current in enumerate(lists_of_stats):
        
        request_stat = [entry["modality_tokens_cnt"] for entry in list_current]
        
        mean_vals.append(np.mean([req_stat for req_stat in request_stat]))

        xi, yi = get_cdf(request_stat)
        plt.plot(xi, yi, label=categories_name[idx], color=plot_colors_by_num[idx], linewidth=line_width)


    if mean_vals[0] == 0: # Avoid division by zero
        percentage_diff = "N/A"
    else:
        diff = ((mean_vals[1] - mean_vals[0]) / mean_vals[0]) * 100
        percentage_diff = f"{diff:+.1f}%"
    
    plt.text(3000, 0.95, percentage_diff, bbox=dict(facecolor='white', edgecolor='black'))

    plt.ylabel("Probability (%)", size=ylabel_size)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"])
    plt.tick_params(axis="y", labelsize=y_params_label_size)

    plt.xlabel("KV Cache footprint (#tokens)", size=xlabel_size, x=0.37)
    
    # plt.xscale("log", base=10, subs=[])
    # plt.xticks([10, 100, 1000, 10000, 12000], ["10", r"$10^2$", r"$10^3$", r"$10^4$", ""])
    
    plt.tick_params(axis="x", labelsize=x_params_label_size)

    leg = fig.legend(
        loc="lower center", fontsize=param_dictionary["legend_size"], ncols=2,
        bbox_to_anchor=(0.5, 1), columnspacing=1.5,
        handlelength=1.2, handletextpad=0.5
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)

    plt.show()

def plot_accuracy_to_parameter(parameter_to_group_by, keep_workload_keys,
                               y_axis_metric="Accuracy",
                               x_axis_metric="Dimensions"):
    
    """
    Plots the accuracy vs. a given parameter of the workload.
    Will take in various workload names and categorize them by specific parameters and then plot

    Parameters
    ----------
    parameter_to_group_by : str
        The parameter to group the workloads by. Eg. "Resampling Filter" or "Dimensions".
    keep_workload_keys : list
        The list of workload names to keep.
    y_axis_metric : str, default="Accuracy"
        The metric to plot on the y-axis. Eg. "Accuracy" or "decode_tokens_cnt".
    x_axis_metric : str, default="Dimensions"
        The metric to plot on the x-axis. Eg. "Dimensions" or "modality tokens avg reduction".
    """
    paremeter_instance_dict = {}    # dictionary of all workloads and their created accuracies and dimensions
    """
    LANCZOS
        worklods_assosiated    -> ['aokvqa_350_thu_lan_rgb_00', 'aokvqa_350_thu_lan_rgb_01' ...
        accs                   -> [0.7085714285714285, 0.7771428571428571' ...
        dimensions             -> ['(100, 100)', '(200, 200)' ... or avg token reduction
    """
    if parameter_to_group_by == "Resampling Filter":
        param_dict_cur = RESAMPLING_FILTERS
        
        for param in param_dict_cur:
            for workload_keys in keep_workload_keys:
                if param_dict_cur[param] in workload_keys: # is "lan" in aokvqa_350_avg_ner_rgb_00 ?
                    if param not in paremeter_instance_dict:
                        paremeter_instance_dict[param] = {"worklods_assosiated": [workload_keys]}
                    else:
                        paremeter_instance_dict[param]["worklods_assosiated"].append(workload_keys)
    elif parameter_to_group_by == "Dimensions":
        param_dict_cur = DIMENSIONS_DICT
        
        for param in param_dict_cur:
            for workload_keys in keep_workload_keys:
                if param in workload_keys: # is "00" in aokvqa_350_avg_ner_rgb_00 ?
                    if param not in paremeter_instance_dict:
                        # print(param)
                        paremeter_instance_dict[param] = {"worklods_assosiated": [workload_keys]}
                    else:
                        paremeter_instance_dict[param]["worklods_assosiated"].append(workload_keys)

    for parameter_instance in paremeter_instance_dict: # iterate over each parameter instance eg "LANCZOS"
        # print(parameter_instance)
        paremeter_instance_dict[parameter_instance]["metric"] = []  # the list of metrics for each workload
        paremeter_instance_dict[parameter_instance]["dimensions"] = []
        paremeter_instance_dict[parameter_instance]["modality tokens avg reduction"] = []

        for work_nam in paremeter_instance_dict[parameter_instance]["worklods_assosiated"]: # eg aokvqa_350_avg_ner_rgb_00

            # print(work_nam)
            
            paremeter_instance_dict[parameter_instance]["dimensions"].append(get_dimensions_str_from_work_nam(work_nam)) # the list of dimensions
            paremeter_instance_dict[parameter_instance]["modality tokens avg reduction"].append(
                dictionary_with_workloads[work_nam]["modality tokens avg reduction"])
            
            if y_axis_metric == "Accuracy":
                metric_interested, _, _ = count_aokvqa_accuracy_and_get_wrong(response_jsonl=dictionary_with_workloads[work_nam]["responses_new"])
            else:   # we want to get a value from the stats
                all_stats = get_requests_in_list(dictionary_with_workloads[work_nam]["stats_new"])
                metric_interested = np.mean([stat[y_axis_metric] for stat in all_stats])
            
            paremeter_instance_dict[parameter_instance]["metric"].append(metric_interested)
        
        combined = list(zip( # Collect the data into a list of tuples
            paremeter_instance_dict[parameter_instance]["metric"],
            paremeter_instance_dict[parameter_instance]["dimensions"],
            paremeter_instance_dict[parameter_instance]["modality tokens avg reduction"]
        ))

        combined_sorted = sorted(combined, key=lambda x: x[2]) # Sort by the third element (index 2)
        sorted_metrics, sorted_dimensions, sorted_reductions = zip(*combined_sorted) # Unpack if you want separate sorted lists again
        paremeter_instance_dict[parameter_instance]["metric"] = sorted_metrics
        paremeter_instance_dict[parameter_instance]["dimensions"] = sorted_dimensions
        paremeter_instance_dict[parameter_instance]["modality tokens avg reduction"] = sorted_reductions
        
    ############
    nam_first = list(paremeter_instance_dict.keys())[0] # just a name of workload to take the len of list for plotting
    total_dimensions_registered = len(paremeter_instance_dict[nam_first]["dimensions"])
    ############

    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)
    
    for (idx, parameter_instance) in enumerate(paremeter_instance_dict):
        
        plt.plot(paremeter_instance_dict[parameter_instance]["metric"],
                 label=parameter_instance if parameter_to_group_by == "Resampling Filter" else param_dict_cur[parameter_instance],
                 color=plot_colors_by_num[idx], linewidth=param_dictionary["line_width"])

    if y_axis_metric == "Accuracy": # add a line for initial metric
        original_metric, _, _ = count_aokvqa_accuracy_and_get_wrong(response_jsonl=AOKVQA_350_RESPONSES)
    else:
        all_stats = get_requests_in_list(AOKVQA_350_STATISTICS)
        original_metric = np.mean([stat[y_axis_metric] for stat in all_stats])
    
    plt.plot(
        [original_metric for i in range(total_dimensions_registered)],
        label="Original", color=plot_colors_by_num[idx+1], linewidth=param_dictionary["line_width"]
        )

    plt.ylabel(y_axis_metric, size=param_dictionary["ylabel_size"])
    if parameter_to_group_by == "Dimensions":
        plt.xlabel("Filters", size=param_dictionary["xlabel_size"], x=0.37)
    else:
        plt.xlabel(x_axis_metric, size=param_dictionary["xlabel_size"], x=0.37)
    
    if x_axis_metric == "Dimensions":
        plot_list = paremeter_instance_dict[nam_first]["dimensions"]
    else:
        if parameter_to_group_by == "Dimensions":
            plot_list = [re_nam for re_nam in RESAMPLING_FILTERS]
        else:
            plot_list = paremeter_instance_dict[nam_first]["modality tokens avg reduction"]
            plot_list = [f"{float(x):.3f}" for x in plot_list]
    
    plt.xticks(
        np.arange(len(plot_list)),
        plot_list, rotation=45)
    
    plt.title(f"Group by {parameter_to_group_by}")
    plt.legend()
    plt.show()

def plot_stat_comparison_multiple(metrics_lists, categories, labels=("List 1", "List 2"), show_logarithmic=True):
    """
    Plot comparison of multiple lists of statistics.

    This function plots the comparison of multiple lists of statistics with respect to multiple categories.
    The first list of statistics is used as the baseline and the average values are plotted in blue.
    The remaining lists of statistics are plotted in different colors with hatching and the percentage difference
    with respect to the baseline is annotated above each bar.

    Parameters
    ----------
    metrics_lists : list of lists
        A list of lists of dictionaries where each dictionary contains the statistics.
    categories : list
        A list of strings representing the categories to be used as labels in the plot.
    labels : tuple of str
        A tuple of strings representing the labels of the lists of statistics.
    show_logarithmic : bool
        Whether to show the y-axis in logarithmic scale.

    Returns
    -------
    None

    """
    total_resized = len(metrics_lists) - 1 # how many new resized bars I have
    averages1 = [np.mean([item[cat] for item in metrics_lists[0] if cat in item]) for cat in categories]
    
    x = np.arange(len(categories))  # label locations
    width = 0.15  # width of the bars
    if total_resized > 5:
        width = 0.9 * width

    fig, ax = plt.subplots(figsize=(10, 6))
    _ = ax.bar(x - width/total_resized, averages1, width, label=labels[0], color='skyblue')

    for i in range(1, total_resized+1): # now plot remaining

        averages2 = [np.mean([item[cat] for item in metrics_lists[i] if cat in item]) for cat in categories]
        _ = ax.bar(
            # x - width/total_resized + width*i/total_resized,
            # x + width*i/total_resized,
            x + width*i,
                   averages2, width,
                   label=correspond_alias_to_parameter_string(alias_given=labels[i]), color=plot_colors_by_num[i], hatch='\\')

    for i, cat in enumerate(categories):
        val1 = np.mean([item[cat] for item in metrics_lists[0] if cat in item])
        
        for jj in range(1, total_resized+1):
            val2 = np.mean([item[cat] for item in metrics_lists[jj] if cat in item])
            if val1 == 0: # Avoid division by zero
                percentage_diff = "N/A"
            else:
                diff = ((val2 - val1) / val1) * 100
                percentage_diff = f"{diff:+.1f}%"
            
            # Get bar height and position
            height = max(val1, val2)
            x_pos = x[i] + width * jj  # Centered between the two bars
            
            # Annotate above the bars
            ax.text(x_pos, height * 1.1, percentage_diff, ha='center', va='bottom', fontsize=8, rotation=0)
    
    if show_logarithmic:
        ax.set_ylabel('Average Value (Log Scale)')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('Average Value')

    ax.set_title('TTFT Breakdown')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_stat_comparison_separate(metrics_lists, categories, labels=("List 1", "List 2")):
    """
    Plot each metric (category) in a separate bar chart, with percentage change annotations
    relative to the first list ("Original").

    Parameters
    ----------
    metrics_lists : list of lists
        A list of lists of dictionaries where each dictionary contains the statistics.
    categories : list
        A list of strings representing the categories (metrics).
    labels : tuple of str
        A tuple of strings representing the labels of the metric groups.

    Returns
    -------
    None
    """
    total_lists = len(metrics_lists)

    for cat in categories:
        averages = []
        for i in range(total_lists):
            values = [item[cat] for item in metrics_lists[i] if cat in item]
            averages.append(np.mean(values))

        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(total_lists)
        width = 0.4

        bars = ax.bar(x, averages, width, color='skyblue')

        original_value = averages[0]

        ax.set_ylim(top=max(averages) * 1.15) # Set y-axis limit with margin

        for i, bar in enumerate(bars):
            height = bar.get_height()

            # ax.text(bar.get_x() + bar.get_width()/2., # Annotate raw value on top
            #         height * 1.01,
            #         f'{height:.2f}',
            #         ha='center', va='bottom', fontsize=8)

            # Annotate % change compared to original (skip the first bar)
            if i > 0 and original_value != 0:
                diff_percent = ((height - original_value) / original_value) * 100
                ax.text(bar.get_x() + bar.get_width()/2.,
                        # height * 1.12,
                        height * 1.01,
                        f'{diff_percent:+.1f}%',
                        ha='center', va='bottom', fontsize=8, color='green' if diff_percent > 0 else 'red')

        ax.set_title(f"{cat} Comparison")
        ax.set_ylabel("Average Value")
        ax.set_xticks(x)
        ax.set_xticklabels([
            correspond_alias_to_parameter_string(alias_given=label) if label != "Original" else label
            for label in labels
        ], rotation=30)
        plt.tight_layout()
        plt.show()

def plot_simple_cdf(value_list, x_ax_tit=""):
    # 1. Sort the data
    """
    Plot the Cumulative Distribution Function (CDF) of a given list of values.

    Parameters
    ----------
    value_list : list or array
        The list of values to plot the CDF for.
    x_ax_tit : str, optional
        The title for the x-axis. Default is an empty string.

    Returns
    -------
    int
        The number of elements in the input list.

    Notes
    -----
    A Cumulative Distribution Function (CDF) is a function that maps a given value to the
    probability that a random variable will take on a value less than or equal to that value.
    The CDF is a way to describe the distribution of a random variable.
    """
    sorted_frames = np.sort(value_list)
    # 2. Calculate the cumulative probabilities
    # The y-axis values will range from 1/N to N/N (which is 1)
    n = len(sorted_frames)
    y_cdf = np.arange(1, n + 1) / n

    figsize_mul = 1.04
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    # 3. Plot the CDF
    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)
    plt.plot(sorted_frames, y_cdf, marker='.', linestyle='-', markersize=5)

    plt.title('CDF of ' + x_ax_tit, fontsize=28)

    plt.xlabel(x_ax_tit, size=xlabel_size, x=0.39)
    plt.ylabel('Probability (%)', size=ylabel_size)
    plt.grid(True, linestyle='--', alpha=0.7) # Add grid for better readability
    plt.ylim(0, 1) # Set y-axis limits from 0 to 1

    x_min = sorted_frames.min()
    x_max = sorted_frames.max()
    num_ticks = 3
    desired_xticks = np.linspace(x_min, x_max, num_ticks)
    desired_xlabels = [f"{int(x)}" for x in desired_xticks]
    plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"], fontsize=y_params_label_size)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()
    return n

def plot_cdfs_of_workloads(keep_workload_keys, metric_name="ttft", dictionary_workload=dictionary_with_workloads,
                           title_add=" - (Max) Frames = 4"):
    """
    Plot the Cumulative Distribution Functions (CDF) for specified metrics of given workloads.

    Parameters
    ----------
    keep_workload_keys : list
        A list of keys representing the workloads to be included in the plot.
    metric_name : str, optional
        The name of the metric to plot the CDF for. Default is "ttft".
    dictionary_workload : dict, optional
        A dictionary containing workload information, where each key is a workload identifier
        and each value is a dictionary with paths to the statistics files. Default is `dictionary_with_workloads`.

    Returns
    -------
    None
    """

    total_metric_list = []
    for workload_keys in keep_workload_keys:
        
        stats_path = dictionary_workload[workload_keys]["stats_new"]
        init_stat_list = get_requests_in_list(jsonl_request_or_stat_pth=stats_path)
        if metric_name == "ttft with asset time":
            total_metric_list.append([stat["ttft"] + stat["video_asset_time"] for stat in init_stat_list])
            # continue
        else:
            total_metric_list.append([stat[metric_name] for stat in init_stat_list])
        
    plt.figure(figsize=(8, 6))

    for i, data in enumerate(total_metric_list):
        data = np.sort(data)
        cdf = np.arange(1, len(data)+1) / len(data)
        plt.plot(data, cdf, label=alias_to_parameter(keep_workload_keys[i], type="video", keep_only_general_category=True),
        linewidth=4)

    plt.ylabel('Cumulative Probability')
    if metric_name == "ttft":
        x_ax_tit = "TTFT"
    elif metric_name == "modality_tokens_cnt":
        x_ax_tit = "Modality Tokens"
    elif metric_name == "video_asset_time":
        x_ax_tit = "Video Asset Time"
    elif metric_name == "ttft with asset time":
        x_ax_tit = "ttft with Video Asset Time"
    elif metric_name == "video_frames":
        x_ax_tit = "Frames Sampled"
    # plt.title('Cumulative Distribution Function (CDF) of ' + x_ax_tit)
    plt.title('CDF of ' + x_ax_tit + title_add)
    plt.xlabel(x_ax_tit)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_colormap_colors(n, cmap_name="Blues", vmin=0.2, vmax=0.8):
    """
    Generate `n` colors from a truncated colormap that avoids fading to white.
    
    Parameters
    ----------
    n : int
        Number of colors to generate.
    cmap_name : str
        Name of the matplotlib colormap (e.g., "Blues", "Greens", "Reds").
    vmin, vmax : float
        Range within the colormap to sample (0.0-1.0). 
        Lower values = darker, higher values = lighter.
    
    Returns
    -------
    colors : list
        List of RGBA colors.
    """
    # base_cmap = cm.get_cmap(cmap_name)
    base_cmap = plt.colormaps.get_cmap(cmap_name)
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc",
        base_cmap(np.linspace(vmin, vmax, 256))
    )
    return [truncated_cmap(i) for i in np.linspace(0, 1, n)]

def scatter_o_bar_accur_o_pixel_red(keep_workload_keys, metric_name="ttft", plot_type="scatter", y_axis="Accuracy (%)",
                                    dictionary_workload=dictionary_with_workloads, legend_only_percentage=False, keep_legend=True, type_modality="img",
                                    original_responses=AOKVQA_350_RESPONSES):
                                    # original_responses=MC_0_30_350_RESPONSES):
    """
    Plot a scatter plot of accuracy vs. a given metric, with option to also plot a bar chart.

    Parameters
    ----------
    keep_workload_keys : list
        List of workload names to keep.
    metric_name : str, default="ttft"
        The metric to plot on the x-axis.
    plot_type : str, default="scatter"
        The type of plot to create. Options are "scatter" and "bar".
    y_axis : str, default="Accuracy"
        The metric to plot on the y-axis. Options are "Accuracy" and "Pixel Reduction".
    legend_only_percentage : bool, default=False
        Whether to show only the percentage change in the legend. If True then we are in the case of pixel reduction
    keep_legend : bool, default=True
        Whether to keep the legend or not

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    """

    xlabel_size = 29
    ylabel_size = 29
    x_params_label_size = 24.5
    y_params_label_size = 24.5

    acc_list = []
    decode_tokens_list = []
    idx=0

    # print_dictionary_content(dictionary_workload, limit_print=3)

    for workload_keys in keep_workload_keys:

        if y_axis == "Accuracy (%)":
            if type_modality == "img":
                # original_metric, _, _ = count_aokvqa_accuracy_and_get_wrong(response_jsonl=AOKVQA_350_RESPONSES)
                original_metric, _, _ = count_aokvqa_accuracy_and_get_wrong(response_jsonl=original_responses)
                finl_responses_path = dictionary_workload[workload_keys]["responses_new"]
                acc, _, _ = count_aokvqa_accuracy_and_get_wrong(response_jsonl=finl_responses_path)
            elif type_modality == "video":
                # original_metric, _, _ = count_vid_mc_accuracy_and_get_wrong(response_jsonl=MC_0_30_350_RESPONSES)
                original_metric, _, _ = count_vid_mc_accuracy_and_get_wrong(response_jsonl=original_responses)
                finl_responses_path = dictionary_workload[workload_keys]["responses_new"]
                acc, _, _ = count_vid_mc_accuracy_and_get_wrong(response_jsonl=finl_responses_path)
            acc_list.append(acc)
        elif y_axis == "Pixel Reduction (%)":
            pixel_red = dictionary_workload[workload_keys]["avg pixel reduction"]
            acc_list.append(pixel_red)
        elif y_axis == "modality tokens avg reduction":
            acc_list.append(dictionary_workload[workload_keys][y_axis])

        if metric_name == "Pixel Reduction (%)":
            pixel_red = dictionary_workload[workload_keys]["avg pixel reduction"]
            decode_tokens_list.append(pixel_red)
        elif metric_name != "modality tokens avg reduction":
            stats_path = dictionary_workload[workload_keys]["stats_new"]
            init_stat_list = get_requests_in_list(jsonl_request_or_stat_pth=stats_path)
            decode_tokens = [stat[metric_name] for stat in init_stat_list]
            avg_decode_tokens = np.mean(decode_tokens)
            decode_tokens_list.append(avg_decode_tokens)
        else:
            decode_tokens_list.append(dictionary_workload[workload_keys][metric_name])

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)
    ax.set_ylabel(y_axis, fontsize=ylabel_size)

    # dynamical width for plots
    sorted_x = np.sort(np.array(decode_tokens_list))    # Sort x-values
    x_diffs = np.diff(sorted_x)                         # Compute differences between consecutive x-values
    min_diff = np.min(x_diffs[x_diffs > 0]) if np.any(x_diffs > 0) else 1.0 # Get smallest positive difference
    dynamic_width = 0.6 * min_diff                      # Set dynamic width (e.g., 80% of smallest gap)

    if plot_type == "scatter":
        for idx, (x, y) in enumerate(zip(decode_tokens_list, acc_list)):
            color = plot_colors_by_num[idx % len(plot_colors_by_num)]
            label = keep_workload_keys[idx]
            if type_modality == "img":
                ax.scatter(x, y, color=color, label=correspond_alias_to_parameter_string(label))
            elif type_modality == "video":
                ax.scatter(x, y, color=color, label=label)
        ax.set_xlabel(metric_name)

    elif plot_type == "bar":
        if metric_name == "Pixel Reduction (%)": # the workload bars don't have to be on specific x dimensions
            x_zip = range(len(keep_workload_keys) + 1)
            x_zip = [x for x in x_zip]
        else:
            x_zip = decode_tokens_list
        hatch_patterns = ['//', '\\\\', 'xx', 'oo']  # Add more if needed
        
        # for bar plot we also want to add the original metric as a bar
        acc_list.insert(0, original_metric)
        keep_workload_keys = keep_workload_keys.copy()
        keep_workload_keys.insert(0, "00")

        for idx, (x, y) in enumerate(zip(x_zip, acc_list)):
            # color = plot_colors_by_num[idx % len(plot_colors_by_num) + 1]   # + 1 so that it is the same with the KV caches
            color = COLORS_ALL[idx]   # + 1 so that it is the same with the KV caches
            label = keep_workload_keys[idx]
            if type_modality == "img":
                ax.bar(x, y, color=color, label=correspond_alias_to_parameter_string(label, only_percentage_change=legend_only_percentage))
            elif type_modality == "video":
                if x in x_zip[idx+1:]: # there will be a bar later at the same x
                    indices = [i for i, xin in enumerate(x_zip) if xin == x]    # get second idx of x in x_zip
                    color = plot_colors_by_num[indices[-1] % len(plot_colors_by_num) + 1]   # + 1 so that it is the same with the KV caches
                    ax.bar(x, y, color=color,
                        label=alias_to_parameter(label,only_percentage_change=legend_only_percentage, type=type_modality),
                        width=dynamic_width,
                        hatch=hatch_patterns[0],
                    )
                elif x in x_zip[:idx]:  # there was already a bar at the same x
                    indices = [i for i, xin in enumerate(x_zip) if xin == x]
                    ax.bar(x, y, color=color,
                        label=alias_to_parameter(label,only_percentage_change=legend_only_percentage, type=type_modality),
                        width=dynamic_width,
                        hatch=hatch_patterns[0],
                    )
                else:
                    ax.bar(x, y, color=color,
                        label=alias_to_parameter(label,only_percentage_change=legend_only_percentage, type=type_modality),
                        width=dynamic_width,
                    )
                ##############################
                
                def custom_x_label_formatter(x, pos):
                    """
                    Formats the x-axis tick labels:
                    - If the number is a whole number (e.g., 5.0), displays it as an integer (e.g., "5").
                    - Otherwise, rounds to two decimal places and displays as a float (e.g., "3.12").
                    'pos' is the tick position (index), which is often not needed but required by FuncFormatter.
                    """
                    # Check if the number has a fractional part. Comparing against int(x) or checking x % 1 != 0
                    # are common ways. Using np.isclose for robustness with floating point comparisons if necessary.
                    if np.isclose(x % 1, 0): # Check if the remainder when divided by 1 is close to 0
                        return f"{int(x)}" # Format as an integer string
                    else:
                        # Standard rounding to three decimal places and format as float string
                        return f"{x:.3f}"

                # --- Apply the custom formatter to the x-axis ---
                # This tells Matplotlib to use your function for generating tick labels
                ax.xaxis.set_major_formatter(FuncFormatter(custom_x_label_formatter))

                ax.set_xticks(x_zip)
                ax.tick_params(axis='x', labelsize=x_params_label_size, rotation=45)
                plt.setp(ax.get_xticklabels(), ha="center") # 'ha' means horizontal alignment

                ##############################
        
        ax.tick_params(axis='y', labelsize=y_params_label_size)
        if not keep_legend:
            ax.set_xticks(x_zip)
            ax.tick_params(axis='x', labelsize=x_params_label_size)
            ax.set_xticklabels(
                # [correspond_alias_to_parameter_string(label, only_percentage_change=legend_only_percentage)
                [alias_to_parameter(label, only_percentage_change=legend_only_percentage, type=type_modality)
                for label in keep_workload_keys],
                rotation=45,  # Optional: angle labels if they overlap
                # ha='right'    # Optional: align labels for better spacing
            )
        
        # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1)) # Format x-axis as percentages
        # Custom formatter: convert 0.1 → 10, 0.2 → 20, etc.
        formatter = FuncFormatter(lambda x, _: f"{int(x * 100)}")
        ax.yaxis.set_major_formatter(formatter)
        if metric_name == "modality_tokens_cnt":
            ax.set_xlabel("Modality Tokens", fontsize=xlabel_size)
        elif metric_name == "video_asset_time":
            ax.set_xlabel("Video Sampling Time", fontsize=xlabel_size)
        else:
            ax.set_xlabel(metric_name, fontsize=xlabel_size)

    elif plot_type == "bar2":
        indices = np.arange(len(keep_workload_keys))
        bar_colors = [plot_colors_by_num[idx % len(plot_colors_by_num)] for idx in range(len(keep_workload_keys))]
        ax.bar(indices, acc_list, color=bar_colors)

        ax.set_xticks(indices)
        ax.set_xticklabels([correspond_alias_to_parameter_string(key) for key in keep_workload_keys], rotation=45, ha="right")
        ax.set_xlabel("Workloads")
    
    elif plot_type == "bar3":
        indices = np.arange(len(keep_workload_keys))
        bar_colors = [plot_colors_by_num[idx % len(plot_colors_by_num)] for idx in range(len(keep_workload_keys))]
        
        bars = ax.bar(indices, acc_list, color=bar_colors)
        ax.set_xticks(indices)
        ax.set_xticklabels([correspond_alias_to_parameter_string(key) for key in keep_workload_keys], rotation=45, ha="right")
        ax.set_xlabel("Workloads")

        # Annotate each bar with the corresponding decode_tokens_list value
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            metric_value = decode_tokens_list[idx]
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x position
                height,                             # y position (top of the bar)
                f'{metric_value:.4f}',              # text: metric value rounded nicely
                ha='center', va='bottom',           # center horizontally, text starts slightly above the bar
                fontsize=8, color='black')
    else:
        raise ValueError(f"Invalid plot_type '{plot_type}'. Use 'scatter' or 'bar'.")

    if y_axis == "Accuracy (%)":
        ax.axhline(
            y=original_metric,
            color="black",
            linestyle="--",
            linewidth=param_dictionary["line_width"],
            xmin=0, xmax=1  # full width of the axes
        )

    # if plot_type == "scatter": # Avoid duplicate labels in legend (only for scatter)
    if keep_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='best', fontsize=9)

    # plt.title(f"{y_axis} vs {metric_name}")
    plt.show()
    return fig

def custom_round(numbers):
    """
    Rounds a list of numbers to a nice number of decimal places.

    For numbers with more than 3 integer digits, rounds to the nearest integer.
    For numbers with 2 or 3 integer digits, rounds to one decimal place.
    For numbers with fewer than 2 integer digits, rounds to three decimal places.

    Example: [123.456, 12.34, 0.0123] -> [123, 12.3, 0.012]
    """
    result = []
    for num in numbers:
        int_digits = len(str(int(abs(num))))  # count integer digits
        if int_digits > 3:
            # only integer
            rounded = round(num)
        elif int_digits > 2:
            rounded = round(num, 1)
        else:
            rounded = round(num, 3)
        result.append(rounded)
    return result

def group_bar_plots_video_techniques_by_frame(group_list, frame_list, dictionary_workload=dictionary_with_workloads, metric="modality_tokens_cnt", original_responses=None):
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    """
    Create grouped bar plots for video techniques by frame count.

    This function generates bar plots for different video processing techniques 
    grouped by frame count. It calculates average metrics for each technique 
    and plots them using a bar chart.

    Parameters
    ----------
    group_list : list
        A list of lists, where each sublist contains video technique identifiers 
        for a particular group.
    frame_list : list
        A list of frame counts corresponding to each group.
    dictionary_workload : dict, optional
        A dictionary containing workload information with keys as technique 
        identifiers and values as dictionaries with paths to response and 
        statistics files. Default is `dictionary_with_workloads`.
    metric : str, optional
        The metric to plot. Default is "modality_tokens_cnt".
    original_responses : dict, optional
        A dictionary with original responses, if needed for metric calculation.

    Returns
    -------
    fig : Figure
        The matplotlib figure object representing the bar plot.
    """

    xlabel_size = 29
    ylabel_size = 29
    x_params_label_size = 24.5
    y_params_label_size = 24.5

    frame_group_list = [] # list of lists with each inner list being the metrics for the corresponding frame group
    for idx in frame_list:
        frame_group_list.append([])
    # print(frame_group_list) # [[], [], [], [], []]
    
    for idx in range(len(group_list)): # every frame_group idx -> 
        # print(idx)
        # for frame_group in group_list:
        frame_group = group_list[idx]
        # print(frame_group) # ['vid-mc-0-30_350_uni_0', 'vid-mc-0-30_350_scc_2_0', 'vid-mc-0-30_350_shb_1_0']
        
        for idx_technique in range(len(frame_group)):
            # workload_for_current_frame = frame_group[idx]
            workload_for_current_frame = frame_group[idx_technique]
            
            if metric == "accuracy":
                finl_responses_path = dictionary_workload[workload_for_current_frame]["responses_new"]
                avg_metrics, _, _ = count_vid_mc_accuracy_and_get_wrong(response_jsonl=finl_responses_path)
            else:
                stats_path = dictionary_workload[workload_for_current_frame]["stats_new"]
                init_stat_list = get_requests_in_list(jsonl_request_or_stat_pth=stats_path)
                metrics = [stat[metric] for stat in init_stat_list]
                avg_metrics = np.mean(metrics)
            
            # frame_group_list[idx].append(avg_metrics)
            frame_group_list[idx_technique].append(avg_metrics)
    # print(frame_group_list)

    penguin_means = {}
    for idx in range(len(group_list[0])):
    # for idx in range(len(group_list)):
    # for idx in range(len(frame_list)):
        penguin_means[
            alias_to_parameter(group_list[0][idx], type="video", keep_only_general_category=True)
            ] = frame_group_list[idx]

    species = [str(fram) for fram in frame_list] # [4, 8]

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    # fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for idx, (attribute, measurement) in enumerate(penguin_means.items()):
        color = plot_colors_by_num[idx % len(plot_colors_by_num) + 1]
        offset = width * multiplier
        rects = ax.bar(x + offset, custom_round(measurement), width, label=attribute, color=color)
        ax.bar_label(rects, padding=7, rotation=69, fontsize=15, label_type="edge")
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if metric == "modality_tokens_cnt":
        ax.set_ylabel("Modality Tokens", fontsize=ylabel_size)
    else:
        ax.set_ylabel(metric, fontsize=ylabel_size)

    ax.set_xticks(x + width, species)
    # ax.legend(loc='upper left', ncols=1)
    ax.legend(loc='best', ncols=1)

    ax.tick_params(axis='x', labelsize=x_params_label_size, rotation=45)
    ax.tick_params(axis='y', labelsize=y_params_label_size)
    ax.set_xlabel('(Max) Frames', fontsize=xlabel_size)

    plt.show()
    return fig

####################################################################################################################################
####################################################################################################################################
# folders n files

def save_fig_pdf(file_nam, fig, figs_path):
    """
    Saves a matplotlib figure as a PDF file.

    Args:
        file_nam (str): The name of the PDF file to save.
        fig (matplotlib.figure.Figure): The figure to save.
        figs_path (str): The path to the folder where to save the PDF file.

    Returns:
        None
    """
    if not os.path.exists(
        os.path.join(figs_path, file_nam + ".pdf")
    ):
        fig.savefig(os.path.join(figs_path, file_nam+".pdf"), bbox_inches='tight')

def delete_matching_items(folder_path, substrings, files_only=True):
    """
    Deletes files (and optionally folders) in the specified folder if their name contains any of the given substrings.

    :param folder_path: Path to the folder to search in.
    :param substrings: List of substrings to check in file/folder names.
    :param files_only: If True, only deletes files. If False, deletes both files and folders.
    """
    if not os.path.isdir(folder_path):
        print("Invalid folder path")
        return
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        if any(sub in item for sub in substrings):
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item_path}")
                elif not files_only and os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")

def subfolders_n_subfiles(base_path, num_tabs=0):

    """
    Prints the number of subfolders and subfiles in the given base_path
    at the given indentation level.

    Args:
        base_path (str): The path to the folder to be analyzed.
        num_tabs (int): The number of indentation levels to be used when printing the results.

    Note:
        If the base_path folder has less than 25 subfolders, it will call itself recursively
        to print the contents of the subfolders with the same indentation level.
    """
    subfolders = 0
    subfiles = 0

    for subfolder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, subfolder)):
            subfolders += 1
        if os.path.isfile(os.path.join(base_path, subfolder)):
            subfiles += 1

    print("\t"*num_tabs, os.path.basename(base_path))
    print("\t"*num_tabs, subfolders, "folders", subfiles, "files")

    # enter if doesn't have many, many folders
    if subfolders < 25:
        for elements in os.listdir(base_path):

            if os.path.isdir(os.path.join(base_path, elements)):
            
                subfolders_n_subfiles(os.path.join(base_path, elements), num_tabs+1)

def delete_folder_content(folder, avoid_gitkeep=True, avoid_files=[], delete_folder=False):
    """
    Deletes the content of a folder (but not the folder itself).

    Args:
        folder (str): The path to the folder containing the files to be deleted.
        avoid_gitkeep (bool, optional): Avoid deleting .gitkeep files. Defaults to True.
        avoid_files (list, optional): A list of file names to be kept. Defaults to an empty list.

    Note:
        This function deletes the specified files and subfolders from the folder.
    """
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if avoid_gitkeep and filename == ".gitkeep":
                continue
            if filename in avoid_files:
                continue
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    if delete_folder:
        if os.path.exists(folder):
            try:
                os.rmdir(folder)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (folder, e))

def print_dictionary_content(mydict, num_tabs=0, limit_print=10):
    """
    Recursively prints the content of a dictionary in a human-readable format.

    Parameters
    ----------
    mydict : dict
        The dictionary to be printed.
    num_tabs : int, optional
        The number of tabs to print before the dictionary content. Default is 0.
    limit_print : int, optional
        The maximum number of dictionary items to print. If the dictionary has more
        items than this limit, it will stop printing after reaching the limit. Default is 10.

    Notes
    -----
        The function prints each key and its associated value. If the value is a dictionary,
        it calls itself recursively to print the content of the nested dictionary.
    """
    if num_tabs == 0:
        print("Length of data: ", len(mydict))
    for key in mydict.keys():
        if type(mydict[key]) == dict:
            limit_print = limit_print - 1
            if limit_print > 0:
                print(num_tabs*"\t", key)
                print_dictionary_content(mydict[key], num_tabs+1, limit_print)
        else:
            if limit_print > 0:
                print(num_tabs*"\t", key, "\n", (num_tabs+1)*"\t", mydict[key])

def bytes_converter(file_size_bytes, convert_to = "MB"):
    """
    Convert file size from bytes to megabytes (MB) with 3 decimal places.

    We divide by 1024 to convert bytes to KBs. We assume that KBs are 1024 bytes.
    
    Parameters:
        file_size_bytes (int or float): File size in bytes.
        
    Returns:
        float: File size in megabytes (MB) rounded to 3 decimal places.
    """
    if convert_to == "MB":
        megabytes = file_size_bytes / (1024 ** 2) # MBs
    else:
        megabytes = file_size_bytes / (1024 ** 1) # KBs
    # return megabytes
    return round(megabytes, 3)

def file_size(file_path, convert_to = "MB"):
    """
    this function will return the file size in MBs or KBs
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return bytes_converter(file_info.st_size, convert_to = convert_to)

def find_file_in_folder_with_str_occurence(folder_path, str_to_find, model_alias=None):
    """
    Searches for a file in the specified folder that contains a given substring in its name.

    Args:
        folder_path (str): The path to the folder in which to search for the file.
        str_to_find (str): The substring to search for in the file names.
        model_alias (str, optional): An additional substring that must be present in the file name.

    Returns:
        str or None: The full path of the first file found containing the substring, or None
        if no such file is found.

    Note:
        This function only searches at the top level of the specified folder and does not
        search recursively in subdirectories.
    """

    for f in os.listdir(folder_path):
        if str_to_find in f:
            if model_alias:
                if model_alias in f:
                    return os.path.join(folder_path, f)
            else:
                """
                go for the original models if no model alias is given
                """
                for mods in ["text-mistral", "image-mistral", "video-mistral"]:
                    if mods in f:
                        return os.path.join(folder_path, f)
    return None

def display_file_count(folder):
    """
    Prints the number of files in the given folder.

    Args:
        folder (str): The path to the folder whose file count will be displayed.

    Note:
        This function does not count subdirectories.
    """
    print(len(
    [name for name in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, name))]
    ))

def create_folder_if_no_exists(folder_pth):
    """
    Create a folder if it does not exist.

    Parameters
    ----------
    folder_pth : str
        The path of the folder to be created.

    Returns
    -------
    bool
        True if the folder was created, False if it already exists.
    """

    if not os.path.exists(folder_pth):
        os.makedirs(folder_pth)
        print(f"Folder {folder_pth} is created")
        return True
    else:
        print(f"Folder {folder_pth} already exists")
        return False

def plot_frame_sizes(frame_sizes_list: list[tuple[int, int]], title="Frame Sizes LLaVA-Video"):
    """
    Plots the frame sizes (width vs. height) for all video instances.

    Args:
        frame_sizes_list (list of tuple): A list where each tuple is (width, height)
                                           representing the dimensions of a video frame.
    """
    widths = [size[0] for size in frame_sizes_list]
    heights = [size[1] for size in frame_sizes_list]

    figsize_mul = 1.04
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)
    plt.scatter(widths, heights, alpha=0.6, s=20) # s is marker size

    plt.xticks(fontsize=x_params_label_size)
    plt.yticks(fontsize=y_params_label_size)

    x_min = min(widths)
    # x_min = 0
    x_max = max(widths)
    num_ticks = 4
    desired_xticks = np.linspace(x_min, x_max, num_ticks)
    desired_xlabels = [f"{int(x)}" for x in desired_xticks]
    plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)

    plt.xlabel('Width (pixels)', size=xlabel_size, x=0.39)
    plt.ylabel("Height (pixels)", size=ylabel_size)
    plt.title(title, fontsize=28)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Optional: Add axis limits based on data range for better visualization
    max_width = max(widths) if widths else 0
    max_height = max(heights) if heights else 0
    plt.xlim(0, max_width * 1.1)
    plt.ylim(0, max_height * 1.1)

    # Add labels for common resolutions if desired (e.g., 640x480, 1280x720, 1920x1080)
    # This might clutter if there are many unique sizes, so consider carefully.
    common_res = [(640, 480), (1280, 720), (1920, 1080)]
    for w, h in common_res:
        if w <= max_width * 1.1 and h <= max_height * 1.1:
            plt.axvline(w, color='gray', linestyle=':', alpha=0.5, label=f'{w}x{h}')
            plt.axhline(h, color='gray', linestyle=':', alpha=0.5)
    # plt.tight_layout()
    plt.show()

def print_top_n_frame_sizes(frame_sizes_list: list[tuple[int, int]], top_n: int = 5):
    """
    Identifies and prints the top_n most common video frame sizes
    along with their respective counts (instances).

    Args:
        frame_sizes_list (list of tuple): A list where each tuple is (width, height)
                                           representing the dimensions of a video frame.
        top_n (int): The number of top common frame sizes to display. Defaults to 5.
    """
    # Use Counter to count occurrences of each (width, height) tuple
    size_counts = Counter(frame_sizes_list)

    # Get the top_n most common sizes
    most_common_sizes = size_counts.most_common(top_n)

    print(f"\n--- Top {top_n} Most Common Frame Sizes ---")
    if not most_common_sizes:
        print("No common frame sizes found (list might be empty or all unique).")
        return

    # Calculate total instances for percentage calculation
    total_instances = sum(size_counts.values())

    for size, count in most_common_sizes:
        width, height = size
        percentage = (count / total_instances) * 100
        print(f"  {width}x{height}: {count} instances ({percentage:.2f}%)")
    print("-" * 40)

def display_frame_saved(pth, display_all_frames=False, limit_print=5):
    """
    Displays the saved frames of a video.

    Args:
        pth (str): Path to the saved numpy array file.
        display_all_frames (bool): If True, all frames will be displayed.
        limit_print (int): The maximum number of frames to display if display_all_frames is True.
    """
    loaded_array_npy = np.load(pth)
    print(loaded_array_npy.shape)
 
    if not display_all_frames:
        plt.imshow(loaded_array_npy[-1], interpolation='nearest')
        plt.show()
    else:
        for i in range(loaded_array_npy.shape[0]):
            plt.imshow(loaded_array_npy[i], interpolation='nearest')
            plt.show()
            if i == limit_print:
                break

def plot_image_sizes(jsonl_path, dataset_name="A-OKVQA"):
    """
    Plot the distribution of image sizes (width vs. height) from a JSON Lines file.

    Parameters
    ----------
    jsonl_path : str
        The path to the JSON Lines file containing the image size information.
    """
    figsize_mul = 1.04
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    widths = []
    heights = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                size = data.get("request", {}).get("modality_size")
                if isinstance(size, list) and len(size) == 2:
                    widths.append(size[0])
                    heights.append(size[1])
            except json.JSONDecodeError:
                continue  # skip bad lines

    # Plot the image sizes
    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)
    # plt.scatter(widths, heights, alpha=0.6, s=30, edgecolor='k')
    plt.scatter(widths, heights, alpha=0.6, s=20) # s is marker size

    # x_min = 0
    x_min = min(widths)
    x_max = max(widths)
    num_ticks = 4
    desired_xticks = np.linspace(x_min, x_max, num_ticks)
    desired_xlabels = [f"{int(x)}" for x in desired_xticks]

    plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)
    plt.yticks(fontsize=y_params_label_size)
    
    plt.xlabel('Width (pixels)', size=xlabel_size, x=0.39)
    plt.ylabel("Height (pixels)", size=ylabel_size)
    plt.title('Frame Sizes ' + dataset_name, fontsize=28)
    plt.grid(True)
    plt.show()

def plot_image_size_density(jsonl_path):
    """
    Plot a density heatmap of image sizes (width vs. height) from a JSON Lines file.

    Parameters
    ----------
    jsonl_path : str
        The path to the JSON Lines file containing the image size information.
    """
    widths = []
    heights = []

    # Read the file
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                size = data.get("request", {}).get("modality_size")
                if isinstance(size, list) and len(size) == 2:
                    widths.append(size[0])
                    heights.append(size[1])
            except json.JSONDecodeError:
                continue

    # 2D Histogram (Heatmap)
    plt.figure(figsize=(8, 6))
    plt.hist2d(widths, heights, bins=30, cmap='Blues')
    plt.colorbar(label='Number of Images')
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Image Size Distribution (Density Heatmap)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def show_top_image_sizes(jsonl_path, top_n=10):
    """
    Show the top_n most common image sizes in a JSON Lines file.

    Parameters
    ----------
    jsonl_path : str
        The path to the JSON Lines file containing the image size information.
    top_n : int
        The number of most common image sizes to display. Defaults to 10.

    Returns
    -------
    None
    """
    size_counter = Counter()

    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                size = tuple(data.get("request", {}).get("modality_size", []))
                if len(size) == 2:
                    size_counter[size] += 1
            except json.JSONDecodeError:
                continue

    most_common = size_counter.most_common(top_n)
    print(f"Top {top_n} most common image sizes:")
    for size, count in most_common:
        print(f"  {size}: {count} images")

def add_token_pixel_reduction(workloadsdict, init_stats):
    """
    Calculate the average pixel reduction and average reduction in the number of tokens between the initial and new requests for each workload.

    Parameters
    ----------
    workloadsdict : dict
        A dictionary containing the information about the workloads.
    init_stats : str
        The path to the JSON Lines file containing the initial statistics of the workloads.

    Returns
    -------
    None
    """
    init_stats = get_requests_in_list(init_stats)
    init_mod_tokens = [init_stat["modality_tokens_cnt"] for init_stat in init_stats]
    for workloads_created in workloadsdict.keys():

        init_reqs = get_requests_in_list(workloadsdict[workloads_created]["initial_requests_jsonl"])
        new_reqs = get_requests_in_list(workloadsdict[workloads_created]["new_request_jsonl"])
        
        init_mods = [init_reqs["request"]["modality_size"] for init_reqs in init_reqs]
        new_mods = [new_reqs["request"]["modality_size"] for new_reqs in new_reqs]

        new_stats = get_requests_in_list(workloadsdict[workloads_created]["stats_new"])
        new_mod_tokens = [new_stat["modality_tokens_cnt"] for new_stat in new_stats]

        # count average pixel reduction
        pix_reduction_list = []
        for i in range(len(init_mods)):
            init_width = int(init_mods[i][0])
            init_height = int(init_mods[i][1])
            fin_width = int(new_mods[i][0])
            fin_height = int(new_mods[i][1])

            orig_size = (init_width*init_height) # measure the percentage reduction
            new_size = (fin_width*fin_height)
            pix_reduction_list.append(
                ((orig_size-new_size) / orig_size) * 100
            )
        # if np.mean(pix_reduction_list) < 1:
        #     print(workloads_created, "Avereage reduction", np.mean(pix_reduction_list))
        workloadsdict[workloads_created]["avg pixel reduction"] = np.mean(pix_reduction_list)
        workloadsdict[workloads_created]["new modality tokens list"] = new_mod_tokens
        orig_size = np.mean(init_mod_tokens)
        new_size = np.mean(new_mod_tokens)
        workloadsdict[workloads_created]["modality tokens avg reduction"] = ((orig_size - new_size) / orig_size) * 100

def pixel_size_CDF(file_list, labels, title=None):
    """
    Plots the CDF of pixel sizes for multiple workload files.
    Parameters
    ----------
    file_list : list of str
        List of file paths to workload JSONL files.
    labels : list of str
        List of labels corresponding to each workload.
    """

    figsize_mul = 1.04
    x_params_label_size = 29.5
    y_params_label_size = 29.5
    ylabel_size = 29
    xlabel_size = 29

    assert len(file_list) == len(labels), "Each file must have a corresponding label."

    fig = plt.figure(figsize=[figsize_mul*6.4, 4.8], constrained_layout=True)
    for idx, file_path in enumerate(file_list):
        reqs = get_requests_in_list(file_path)
        modality_sizes = [req["request"]["modality_size"] for req in reqs]

        pixel_sizes = [int(size[0]) * int(size[1]) for size in modality_sizes]

        sorted_data = np.sort(pixel_sizes)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label="Original" if idx == 0 else correspond_alias_to_parameter_string(labels[idx]), linewidth=2)

    desired_xticks = [100000, 250000, 400000]
    desired_xlabels = ["100K", "250K", "400K"] # Optional: customize labels for readability
    plt.xticks(desired_xticks, desired_xlabels, fontsize=x_params_label_size)

    plt.xlabel('Pixels = Width * Height', size=xlabel_size, x=0.39)
    plt.ylabel("Probability (%)", size=ylabel_size)
    if title:
        plt.title(title, fontsize=28)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"], fontsize=y_params_label_size)
    # plt.legend()
    plt.grid(True)
    plt.show()
    return fig

def kv_cache_footprint_comparison_multi_or_ttft(lists_of_stats, categories_name, plot_choice="KV Cache footprint (#tokens)"):
    """
    Plot a comparison of the CDFs for different categories of KV Cache footprints.
    Or plot CDF for TTFT

    Parameters
    ----------
    lists_of_stats : list of lists
        A list where each element is a list of dictionaries containing:
        - "modality_tokens_cnt": int
    categories_name : list
        A list of category names corresponding to each list in lists_of_stats.

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)
    ax = fig.gca() # Get the current axes to make explicit plotting calls

    # mean_vals = []
    for idx, list_current in enumerate(lists_of_stats):
        if plot_choice == "TTFT (sec)":
            print(list_current)
            request_stat = [entry["ttft"] for entry in list_current]
        elif plot_choice == "KV Cache footprint (#tokens)":
            request_stat = [entry["modality_tokens_cnt"] for entry in list_current]
            # request_stat = [entry["modality_tokens_cnt"] for entry in list_current]
        # mean_vals.append(np.mean(request_stat))
        xi, yi = get_cdf(request_stat)
        plt.plot(xi, yi, label=categories_name[idx], color=plot_colors_by_num[idx], linewidth=param_dictionary["line_width"])

    plt.ylabel("Probability (%)", size=param_dictionary["ylabel_size"])
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 ", "25", "50", "75", "100"])
    plt.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
    plt.xlabel(plot_choice, size=param_dictionary["xlabel_size"], x=0.37)
    plt.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])

    # n_items = len(categories_name) # Or dynamically set number of columns based on legend length
    leg = fig.legend(
        # loc="lower right",                  # Places the legend at the bottom right of the figure
        loc="lower center",                  # Places the legend at the bottom center of the figure
        fontsize=param_dictionary["legend_size"],  # Controls text size using your own setting
        # ncols=n_items,                       # One item per column = all in a row
        ncols=2,                             # Forces the legend into 2 columns
        bbox_to_anchor=(0.5, 1),             # Positions the legend above the plot (anchor point at top center)
        # bbox_to_anchor=(1, 0),               # Positions the legend's bottom-right corner at figure's bottom-right
        columnspacing=1.5,                   # Space between legend columns
        handlelength=1.2,                    # Length of the legend lines
        handletextpad=0.5                    # Space between line and label text
        # frameon=False                        # Optional: removes the frame around the legend
    )
    
    # # --- Legend Placement for Bottom Right Without Overlap ---
    # # The key is to place the legend relative to the Axes, not the figure directly
    # # and use loc='lower right' which places the legend's bottom-right corner
    # # at the specified bbox_to_anchor coordinates (which are axes coordinates here).

    # leg = ax.legend( # Use ax.legend() instead of fig.legend()
    #     loc="lower right", # Aligns the legend's lower-right point to bbox_to_anchor
    #     fontsize=param_dictionary["legend_size"],
    #     ncols=2, # Or adjust based on how many columns look best for your tags
    #     # You might even try ncols=1 if you have many long labels and want a vertical stack
    #     # ncols=len(categories_name), # Use this if you want a single row
    #     bbox_to_anchor=(0.98, 0.02), # These are coordinates relative to the axes (0 to 1)
    #                                  # 0.98 for X (close to right edge)
    #                                  # 0.02 for Y (just above bottom edge)
    #     columnspacing=1.5,
    #     handlelength=1.2,
    #     handletextpad=0.5,
    #     frameon=False,               # Removes the legend frame for cleaner look
    #     borderaxespad=0.              # Removes padding between axes and legend bbox (important with bbox_to_anchor)
    # )

    for handle in leg.legend_handles:
        handle.set_linewidth(7.5)           # Thickens the legend lines

    plt.show()
    return fig

def kv_cache_footprint_comparison_multi_or_ttft_refactored(lists_of_stats, categories_name, ax, plot_choice="KV Cache footprint (#tokens)"):
    """
    Plots a comparison of CDFs for different categories onto a given Axes object.

    Parameters
    ----------
    lists_of_stats : list of lists
        A list where each element is a list of dictionaries containing:
        - "modality_tokens_cnt": int
        - "ttft": float (if plot_choice is "TTFT (sec)")
    categories_name : list
        A list of category names corresponding to each list in lists_of_stats.
    ax : matplotlib.axes.Axes
        The Axes object to plot onto.
    plot_choice : str, optional
        The metric to plot ("KV Cache footprint (#tokens)" or "TTFT (sec)").
        Defaults to "KV Cache footprint (#tokens)".

    Returns
    -------
    tuple: (handles, labels)
        A tuple containing lists of Line2D objects (handles) and their corresponding
        labels for creating a combined legend.
    """
    print("lists_of_stats", len(lists_of_stats))
    for idx, list_current in enumerate(lists_of_stats):
        # print("list_current", list_current)
        if plot_choice == "TTFT (sec)":
            # request_stat = [entry["ttft"] for entry in list_current]
            # do something 100 times
            print("NEW TTFT ENTRY")
            if "ttft" in list_current[0]:
                request_stat = [entry["ttft"] for entry in list_current]
            else:
                request_stat = [ttft_by_request(request) for request in list_current]
        elif plot_choice == "KV Cache footprint (#tokens)":
            request_stat = [entry["modality_tokens_cnt"] for entry in list_current]
        xi, yi = get_cdf(request_stat)
        ax.plot(xi, yi, label=categories_name[idx], color=COLORS_ALL[idx], linewidth=param_dictionary["line_width"]) # color=plot_colors_by_num[idx]

    ax.set_ylabel("Probability (%)", size=param_dictionary["ylabel_size"])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0 ", "25", "50", "75", "100"])
    ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
    ax.set_xlabel(plot_choice, size=param_dictionary["xlabel_size"]) # Removed x-offset, tight_layout handles this
    ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])

    return ax.get_legend_handles_labels() # Return handles and labels for combined legend

def kv_cache_footprint_comparison_multi_or_ttft_refactored_percentiles(lists_of_stats, categories_name, plot_choice="KV Cache footprint (#tokens)"):
    """
    Plot a bar chart comparing the KV cache footprint or TTFT across different workloads.

    Parameters
    ----------
    lists_of_stats : list
        A list of lists, where each sublist contains dictionaries with statistics
        for a specific workload.
    categories_name : list
        A list of category names corresponding to each list in lists_of_stats.
    plot_choice : str, optional
        The metric to plot ("KV Cache footprint (#tokens)" or "TTFT (sec)").
        Defaults to "KV Cache footprint (#tokens)".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    """
    
    num_cats = len(categories_name)
    # print(categories_name)
    # bar_width = 0.8 / max(1, 2) # 2 percentiles
    bar_width = 0.35  # width per percentile bar
    x = np.arange(num_cats)

    fig, ax_cur = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for idx, list_current in enumerate(lists_of_stats):
        if plot_choice == "TTFT (sec)":
            # request_stat = [entry["ttft"] for entry in list_current]
            request_stat = [entry["first_token_time"] - entry["first_scheduled_time"] for entry in list_current]
            for i in range(2):
                print("NEW TTFT ENTRY HERE")
        else:
            request_stat = [entry["modality_tokens_cnt"] for entry in list_current]
        
        # Compute percentiles
        p95 = np.percentile(request_stat, 95)
        p99 = np.percentile(request_stat, 99)
        
        # p95 bar
        # xpos = x[idx] + 0 * bar_width
        ax_cur.bar(
            x[idx] - bar_width / 2,
            p95,
            bar_width,
            color=COLORS_ALL[idx],
            alpha=0.8,
            label=f"{categories_name[idx]} (p95)" if idx == 0 else None,
        )
        
        # p99 bar
        # xpos = x[idx] + 1 * bar_width
        ax_cur.bar(
            x[idx] + bar_width / 2,
            p99,
            bar_width,
            color=COLORS_ALL[idx],
            alpha=0.4,  # lighter so p99 is distinguishable
            hatch="//",
            edgecolor="black",
            label=f"{categories_name[idx]} (p99)" if idx == 0 else None,
        )

    # Axis formatting
    ax_cur.set_ylabel(plot_choice, size=param_dictionary["ylabel_size"])
    # ax_cur.tick_params(axis="y", labelsize=y_params_label_size)
    ax_cur.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
    ax_cur.set_xlabel("Workload", size=param_dictionary["xlabel_size"])
    ax_cur.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    ax_cur.set_xticks(x)
    ax_cur.set_xticklabels(categories_name, rotation=20, ha="right")

    # Legend
    handles, labels = ax_cur.get_legend_handles_labels()
    ax_cur.legend(handles[:2], labels[:2], fontsize=14, frameon=False, loc="best")

    plt.show()
    return fig

def collect_plots(rets_keys, dict_work, original_responses=AOKVQA_350_RESPONSES, orig_stats=AOKVQA_350_STATISTICS):
    """
    Collects plots comparing the KV Cache footprint and TTFT (in seconds) of different workloads.

    Parameters:
    rets_keys (list): A list of aliases for the workloads.
    dict_work (dict): A dictionary containing the workloads and their corresponding statistics.
    original_responses (str): A JSONL file containing the original responses.
    orig_stats (str): A JSONL file containing the original statistics.

    Returns:
    fig_ret (matplotlib.figure.Figure): The figure containing the combined plots.
    """
    list_of_metrics = []
    for workload_keys in rets_keys:
        list_of_metrics.append(get_requests_in_list(jsonl_request_or_stat_pth=dict_work[workload_keys]["stats_new"]))
    # scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="modality tokens avg reduction", plot_type="scatter", y_axis="Pixel Reduction (%)")
    # scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="Pixel Reduction (%)", plot_type="scatter")
    # scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="modality_tokens_cnt") # ttft
    # _ = scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="modality tokens avg reduction", plot_type="scatter", dictionary_workload=dict_work, original_responses=original_responses)
    
    # fig_ret = scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="Pixel Reduction (%)", plot_type="bar", y_axis="Accuracy (%)", legend_only_percentage=True, keep_legend=False,
    #                                           dictionary_workload=dict_work, original_responses=original_responses)
    print("rets_keys", rets_keys)
    fig_ret = scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="Pixel Reduction (%)", plot_type="bar", y_axis="Accuracy (%)", legend_only_percentage=True, keep_legend=False,
                                              dictionary_workload=dict_work, original_responses=original_responses)
    print("done accuracy")
    # save_fig_pdf(file_nam="acc_vs_pixel_reduction_" + MODEL_ALIAS, fig=fig_ret,
    #          figs_path="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/figures")

    # scatter_o_bar_accur_o_pixel_red(rets_keys, metric_name="ttft", plot_type="bar2")
    
    #################################################################################### KV Cache Footprint
    total_lists = [get_requests_in_list(jsonl_request_or_stat_pth=orig_stats)]
    cat_nams = ["Original"]
    # print(rets_keys)
    # print(total_lists)
    for workload_keys in rets_keys:
        print(dict_work[workload_keys]["stats_new"])
        finl_stat_list = get_requests_in_list(dict_work[workload_keys]["stats_new"])

        total_lists.append(finl_stat_list)
        # cat_nams.append(correspond_alias_to_parameter_string(workload_keys))
        cat_nams.append(correspond_alias_to_parameter_string(workload_keys, only_percentage_change=True))
        # kv_cache_footprint_comparison(lists_of_stats=[init_stat_list, finl_stat_list],
        #                               categories_name=["Original", correspond_alias_to_parameter_string(workload_keys)])
    print(cat_nams)
    # return

    for i in total_lists:
        print(i[0])
    # fig_ret = kv_cache_footprint_comparison_multi_or_ttft(total_lists, cat_nams)
    # # save_fig_pdf(file_nam="kv_cache_footprint", fig=fig_ret,
    #         #  figs_path="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/figures")
    # # kv_cache_footprint_comparison_multi_or_ttft(total_lists[:4], cat_nams[:4])
    # fig_ret = kv_cache_footprint_comparison_multi_or_ttft(total_lists, cat_nams, plot_choice="TTFT (sec)")
    # # save_fig_pdf(file_nam="ttft_footprint", fig=fig_ret,
    # #          figs_path="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/figures")
    # # kv_cache_footprint_comparison_multi_or_ttft(total_lists[:4], cat_nams[:4], plot_choice="TTFT (sec)")
    
    #################################################################################### TTFT Breakdown
    # rets_keys.insert(0, "Original") # In the beginning, add original before resizing
    # init_stat_list = get_requests_in_list(jsonl_request_or_stat_pth=AOKVQA_350_STATISTICS)
    # list_of_metrics.insert(0, init_stat_list)
    # plot_stat_comparison_separate(metrics_lists=list_of_metrics,
    #                             categories=["encoder_time", "processor_time", "ttft"], labels=rets_keys)
    # # plot_stat_comparison_multiple(metrics_lists=list_of_metrics,
    # #                               categories=["encoder_time", "processor_time", "ttft"], labels=rets_keys,
    # #                               show_logarithmic=True)
    ####################################################################################
    # Combine both CDFs
    # 1. Create the main figure and two subplots (axes)
    fig_combined, (ax_footprint, ax_ttft) = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=False)

    # 2. Call the refactored function for each subplot - Pass the specific Axes object to each call
    handles_footprint, labels_footprint = kv_cache_footprint_comparison_multi_or_ttft_refactored(
        total_lists, cat_nams, ax=ax_footprint, plot_choice="KV Cache footprint (#tokens)")
    handles_ttft, labels_ttft = kv_cache_footprint_comparison_multi_or_ttft_refactored(
        total_lists, cat_nams, ax=ax_ttft, plot_choice="TTFT (sec)")
    
    # 3. Set subplot titles (optional, but good for clarity)
    ax_footprint.set_title("KV Cache Footprint CDF", fontsize=32)
    ax_ttft.set_title("TTFT CDF", fontsize=32)

    # 4. Create a single, shared legend for the entire figure
    # We can just take them directly or iterate through one of the sets of handles/labels
    # and manually set the legend items.
    # Given your function's loop, categories_name already serves as the distinct labels.
    # So we can just take the first set of handles/labels (they should be identical for both plots)
    # Assuming labels_footprint and labels_ttft are the same for the categories:
    unique_labels = labels_footprint
    unique_handles = handles_footprint # Or handles_ttft, they should be the same handles
    
    leg = fig_combined.legend(
        unique_handles, unique_labels,
        loc="lower center", # Places the legend at the bottom center of the figure
        fontsize=param_dictionary["legend_size"] * 1.5, # Adjust size for combined legend
        # ncols=len(cat_nams), # Number of categories in your legend
        # ncols=6, # Number of categories in your legend
        ncols=11, # Number of categories in your legend
        # bbox_to_anchor=(0.5, -0.15), # Position below the plots (adjust as needed)
        bbox_to_anchor=(0.5, 0.93),
        columnspacing=1.2,
        handlelength=1.1,
        handletextpad=0.5,
        frameon=False # Optional: remove legend frame
    )

    for handle in leg.legend_handles: # Apply line thickness to the shared legend handles
        handle.set_linewidth(7.5)

    # [left, bottom, right, top] in figure coordinates (0-1)
    # bottom=0.2 means the plotting area starts 20% up from the bottom, leaving space.
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    plt.show()
    # save_fig_pdf(file_nam="cdfs_combined_" + MODEL_ALIAS, fig=fig_combined,
    #          figs_path="/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/artifacts/figures")
    
    _ = kv_cache_footprint_comparison_multi_or_ttft_refactored_percentiles(
        total_lists, cat_nams, plot_choice="TTFT (sec)")

idx_to_reduction = {
    0: "0",
    1: "5",
    2: "10",
    3: "20",
    4: "30",
    5: "40",
    6: "50",
    7: "60",
    8: "70",
    9: "80",
    10: "90"
}

def plot_model_reductions_by_models(results_dict, percentages=[0, 1, 2, 3], metric="ttft",
                                    models_orig_results = {
                                        "qwen-2b-instruct": AOKVQA_350_QWEN_TWO_STATISTICS[0],
                                        "qwen-7b-instruct": AOKVQA_350_QWEN_SEVEN_STATISTICS[0],
                                        "llava-ov-qwen2-0.5b": AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS[0],
                                        "llava-ov-qwen2-7b": AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS[0],
                                        "pixtral_12b": AOKVQA_350_PIXTRAL_12B_STATISTICS[0],
                                        "image-mistral": AOKVQA_350_STATISTICS[0],
                                    }):
    """
    Plot model reductions by models.
    
    Parameters:
    results_dict (dict): results from load_workload_stats
    percentages (list): percentages of reduction to plot
    metric (str): metric to plot
    models_orig_results (dict): map model names to original stats files
    
    Returns:
    None
    """
    models = list(results_dict.keys())
    num_models = len(models)
    num_perc = len(percentages)

    model_metrics = {model: {} for model in models}
    for model in models:

        # original first
        stats_file = models_orig_results[model]
        results = get_requests_in_list(stats_file)
        if metric == "ttft":
            if "ttft" in results[0]:
                metrics = [res["ttft"] for res in results]
            else:
                metrics = [ttft_by_request(request) for request in results]
            print("NEW TTFT ENTRY HERE HERE HERE A")
        else:
            metrics = [res[metric] for res in results]
        model_metrics[model][0] = metrics
        # print(stats_file)
        # print(np.mean(model_metrics[model][0]))

        for idx, works in enumerate(results_dict[model]):
            stats_file = results_dict[model][works]["stats_new"]
            results = get_requests_in_list(stats_file)
            if metric == "ttft":
                if "ttft" in results[0]:
                    metrics = [res["ttft"] for res in results]
                else:
                    metrics = [ttft_by_request(request) for request in results]
                print("NEW TTFT ENTRY HERE HERE HERE gio")
            else:
                metrics = [res[metric] for res in results]
            model_metrics[model][idx+1] = metrics
            # print(stats_file)
            # print(np.mean(model_metrics[model][idx+1]))
    
    bar_width = 0.8 / max(1, num_perc)
    x = np.arange(num_models)  # one x position per model
    # cmap = plt.get_cmap("tab10")
    # colors_all = [cmap(i / num_models) for i in range(num_models)]
    colors_all = COLORS_ALL

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for j, model in enumerate(models):
        for i, p in enumerate(percentages):
            vals = model_metrics[model].get(p, [])
            # convert to numpy and drop NaNs
            arr = np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)
            arr = arr[~np.isnan(arr)]

            height = float(np.mean(arr))   # or np.median(arr) if you prefer
            xpos = x[j] + i * bar_width

            ax.bar(
                xpos,
                height,
                bar_width,
                label=f"{idx_to_reduction[p]}%" if j == 0 else None,  # label each percentage only once
                color=colors_all[i])

            # ax.text( # optional annotation
            #     xpos,
            #     height + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] > 0 else 1),
            #     f"{height:.2f}",
            #     ha="center",
            #     va="bottom",
            #     fontsize=8)

    ax.set_xticks(x + bar_width * (num_perc - 1) / 2.0)
    ax.set_xticklabels([map_model_alias_to_name(model) for model in models], fontsize=12)
    # ax.set_xticklabels(models, fontsize=12)
    ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    ax.set_xlabel("Model", size=param_dictionary["xlabel_size"])
    if metric == "ttft":
        ax.set_ylabel("Avg TTFT", size=param_dictionary["ylabel_size"])
    elif metric == "modality_tokens_cnt":
        ax.set_ylabel("Avg Modality Tokens", size=param_dictionary["ylabel_size"])
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),  # x=0.5 centers it, y=1.02 places it just above the plot
        bbox_transform=ax.transAxes,
        ncol=6,                      # number of columns if you have multiple legend items
        fontsize=16,
        frameon=False                # optional: removes the legend box
    )
    plt.yticks(fontsize=param_dictionary["y_params_label_size"])
    # plt.tight_layout()
    plt.show()

# TO HAVE REACHED THIS FAR, I KNOW THAT THEY ARE ONLY ONE FILE PER GLOBAL
def plot_model_reductions_by_reduction(results_dict, percentages=[0, 1, 2, 3], metric="ttft",
                                    models_orig_results = {
                                        "qwen-2b-instruct": AOKVQA_350_QWEN_TWO_STATISTICS[0],
                                        "qwen-7b-instruct": AOKVQA_350_QWEN_SEVEN_STATISTICS[0],
                                        "llava-ov-qwen2-0.5b": AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS[0],
                                        "llava-ov-qwen2-7b": AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS[0],
                                        "pixtral_12b": AOKVQA_350_PIXTRAL_12B_STATISTICS[0],
                                        "image-mistral": AOKVQA_350_STATISTICS[0],
                                    }):
    """
    Plot the mean metric of different models at different reduction percentages.

    Parameters
    ----------
    results_dict : dict
        A dictionary where the keys are the model names and the values are dictionaries with the keys "stats_new" and "stats_orig".
    percentages : list
        A list of reduction percentages to plot.
    metric : str
        The name of the metric to plot.
    models_orig_results : dict
        A dictionary where the keys are the model names and the values are the paths to the statistics files of the original models.

    Returns
    -------
    None
    """
    models = list(results_dict.keys())
    num_models = len(models)
    num_perc = len(percentages)

    model_metrics = {model: {} for model in models}
    for model in models:

        # original first
        stats_file = models_orig_results[model]
        print(stats_file)
        results = get_requests_in_list(stats_file)
        if metric == "ttft":
            if "ttft" in results[0]:
                metrics = [res["ttft"] for res in results]
            else:
                metrics = [ttft_by_request(request) for request in results]
            for i in range(2):
                print("NEW TTFT ENTRY HERE HERE")
        else:
            metrics = [res[metric] for res in results]
        model_metrics[model][0] = metrics

        for idx, works in enumerate(results_dict[model]):
            stats_file = results_dict[model][works]["stats_new"]
            results = get_requests_in_list(stats_file)
            if metric == "ttft":
                if "ttft" in results[0]:
                    metrics = [res["ttft"] for res in results]
                else:
                    metrics = [ttft_by_request(request) for request in results]
                print("NEW TTFT ENTRY HERE HERE HERE")
            else:
                metrics = [res[metric] for res in results]
            model_metrics[model][idx+1] = metrics
            # print(np.mean(model_metrics[model][idx+1]))
    
    bar_width = 0.8 / max(1, num_models)
    x = np.arange(num_perc)  # one x position per percentage slot
    # cmap = plt.get_cmap("tab10")
    # colors_all = [cmap(i / num_models) for i in range(num_models)]
    colors_all = COLORS_ALL

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    for j, p in enumerate(percentages):
        for i, model in enumerate(models):
            vals = model_metrics[model].get(p, [])
            # convert to numpy and drop NaNs
            arr = np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)
            arr = arr[~np.isnan(arr)]

            height = float(np.mean(arr))   # or np.median(arr) if you prefer
            xpos = x[j] + i * bar_width

            ax.bar(
                xpos,
                height,
                bar_width,
                label=model if j == 0 else None,  # add legend label only once
                color=colors_all[i])

            ax.text( # optional annotation
                xpos,
                height + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] > 0 else 1),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8)

    ax.set_xticks(x + bar_width * (num_models - 1) / 2.0)
    ax.set_xticklabels([f"{idx_to_reduction[p]}%" for p in percentages], fontsize=12)
    ax.set_xlabel("Reduction (%)")
    ax.set_ylabel(metric, size=param_dictionary["ylabel_size"])
    ax.legend(loc="best", fontsize=10)
    # plt.tight_layout()
    plt.show()

def transform_list_of_single_entry_dicts_to_dict(input_list: list[dict]) -> dict:
    """
    Transforms a list of dictionaries (where each dict has a single key-value pair)
    into a single dictionary.

    Args:
        input_list: A list of dictionaries, e.g.,
                    `[{'outer_key1': {'inner_dict1'}}, {'outer_key2': {'inner_dict2'}}]`.

    Returns:
        A single dictionary, e.g.,
        `{'outer_key1': {'inner_dict1'}, 'outer_key2': {'inner_dict2'}}`.

    Raises:
        TypeError: If an element in the input_list is not a dictionary,
                   or if the value of an inner dictionary is not a dictionary.
        ValueError: If an inner dictionary does not contain exactly one key-value pair.
    """
    result_dict = {}
    for item_dict in input_list:
        if not isinstance(item_dict, dict):
            raise TypeError(
                f"Expected each element in the list to be a dictionary, "
                f"but found type: {type(item_dict)} with value: {item_dict}"
            )

        if len(item_dict) != 1:
            raise ValueError(
                f"Each dictionary in the list must contain exactly one key-value pair, "
                f"but found {len(item_dict)} in {item_dict}"
            )

        # Efficiently get the single key and value from the dictionary
        # dict.items() returns a view, next(iter(...)) gets the first (and only) item
        key, value = next(iter(item_dict.items()))

        if not isinstance(value, dict):
            raise TypeError(
                f"The value for key '{key}' was expected to be a dictionary, "
                f"but found type: {type(value)} with value: {value}"
            )

        result_dict[key] = value

    return result_dict

def map_model_alias_to_name(model_alias):
    dic_ret = {
        "llava-ov-qwen2-7b": "LLaVA-7b",
        "qwen-7b-instruct": "Qwen-7b",
        "qwen-2b-instruct": "Qwen-2b",
        "llava-ov-qwen2-0.5b": "LLaVA-0.5b",
        "pixtral_12b": "Pixtral"
    }
    return dic_ret[model_alias]

def plot_latency_breakdowns(results_dict, models_orig_results = {
                                        "qwen-2b-instruct": AOKVQA_350_QWEN_TWO_STATISTICS[0],
                                        "qwen-7b-instruct": AOKVQA_350_QWEN_SEVEN_STATISTICS[0],
                                        "llava-ov-qwen2-0.5b": AOKVQA_350_LLAVA_OV_QWEN2_0_5_STATISTICS[0],
                                        "llava-ov-qwen2-7b": AOKVQA_350_LLAVA_OV_QWEN2_7_STATISTICS[0],
                                        "pixtral_12b": AOKVQA_350_PIXTRAL_12B_STATISTICS[0],
                                        "image-mistral": AOKVQA_350_STATISTICS[0],
                                    },
                                    tit="Latency Breakdown for Images"):
    models = list(results_dict.keys())
    num_models = len(models)

    # Lists to store latency components
    ttft_list = []
    processor_list = []
    encoder_list = []

    processor_list_std = []
    ttft_list_std = []
    encoder_list_std = []
    
    for model in models:
        print(model)
        # original first
        stats_file = models_orig_results[model]
        results = get_requests_in_list(stats_file)
        
        # if "ttft" in results[0]:
        #     ttft = [res["ttft"] for res in results]
        #     ttft_times_std = [np.std([res["ttft"] - res["processor_time"] - res["encoder_time"] for res in results])]
        # else:
        print(results)
        ttft = [ttft_by_request(res) for res in results]
        ttft_times_std = [np.std([ttft_by_request(res) - processor_time_by_request(res) - encoder_time_by_request(res) for res in results])]
        for i in range(2):
            print("CORRECT HERE")
        
        ttft_list_std.append(ttft_times_std[0])
        ttft_avg = np.average([0 if v is None else v for v in ttft])
        ttft_list.append(ttft_avg)
        
        if "processor_time" in results[0]:
            processor = [res["processor_time"] for res in results]
            processor_times_std = [np.std([res["processor_time"] for res in results])]
        else:
            processor = [processor_time_by_request(res) for res in results]
            processor_times_std = [np.std([processor_time_by_request(res) for res in results])]
            for i in range(2):
                print("CORRECT HERE AS WELL")
        processor_list_std.append(processor_times_std[0])
        processor_avg = np.average([0 if v is None else v for v in processor])
        processor_list.append(processor_avg)
        print("Processor:", processor_avg)

        if "encoder_time" in results[0]:
            encoder = [res["encoder_time"] for res in results]
            encoder_times_std = [np.std([res["encoder_time"] for res in results])]
        else:
            encoder = [encoder_time_by_request(res) for res in results]
            encoder_times_std = [np.std([encoder_time_by_request(res) for res in results])]
            for i in range(2):
                print("CORRECT HERE AS WELL 222")
        
        encoder_list_std.append(encoder_times_std[0])
        encoder_avg = np.average([0 if v is None else v for v in encoder])
        encoder_list.append(encoder_avg)
        print("Encoder:", encoder_avg)

    # Convert to numpy arrays for stacking
    ttft_array = np.array(ttft_list)
    processor_array = np.array(processor_list)
    encoder_array = np.array(encoder_list)

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, param_dictionary["figsize_mul"]*4.8], constrained_layout=True)
    x = np.arange(num_models)

    ax.bar(x, processor_array, label='Processor', color="#31511E")
    ax.bar(x, encoder_array, bottom=processor_array, label='Encoder', color="#859F3D")
    ax.bar(x, ttft_array, bottom=processor_array + encoder_array, label='LLM', color='#C9DD84')

    # for i in range(len(x)):
    #     # Processor whiskers
    #     ax.errorbar(
    #         x[i] - 0.2,
    #         processor_list[i],
    #         yerr=processor_list_std[i],
    #         ecolor="#8B0000",
    #         color="#31511E",
    #         capsize=2,
    #         elinewidth=1.8,
    #     )
    #     # Encoder whiskers (stacked on processor)
    #     encoder_y = processor_list[i] + encoder_list[i]
    #     ax.errorbar(
    #         x[i],
    #         encoder_y,
    #         yerr=encoder_list_std[i],
    #         ecolor="#D1495B",
    #         color="#859F3D",
    #         capsize=2,
    #         elinewidth=1.8,
    #     )
    #     # LLM whiskers (stacked on encoder + processor)
    #     llm_y = processor_list[i] + encoder_list[i] + ttft_list[i]
    #     ax.errorbar(
    #         x[i] + 0.2,
    #         llm_y,
    #         yerr=ttft_list_std[i],
    #         ecolor="#F4A261",
    #         color="#C9DD84",
    #         capsize=2,
    #         elinewidth=1.8,
    #     )
    
    # Custom legend to avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    preprocess_legend = Line2D([0], [0],  linewidth=8, color='#31511E', marker="$I$", markerfacecolor='#8B0000', markeredgecolor='#8B0000', markersize=10)
    encoder_legend = Line2D([0], [0],  linewidth=8, color='#859F3D', marker="$I$", markerfacecolor='#D1495B', markeredgecolor='#D1495B', markersize=10)
    llm_legend = Line2D([0], [0],  linewidth=8, color='#C9DD84', marker="$I$", markerfacecolor='#F4A261', markeredgecolor='#F4A261', markersize=10)

    # Combine all
    legend_items = [preprocess_legend, encoder_legend, llm_legend]
    legend_labels = ["Preprocess", "Encoder", "LLM"]

    ax.legend(
        legend_items,
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=param_dictionary["legend_font_size"],
        ncols=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.23)
    )

    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    
    ax.set_xticklabels([map_model_alias_to_name(model) for model in models], rotation=45, ha='right')
    ax.set_ylabel('Latency (seconds)', size=param_dictionary["ylabel_size"])

    ax.tick_params(axis="y", labelsize=param_dictionary["y_params_label_size"])
    
    ax.set_title(tit, size=param_dictionary["title_size"])
    plt.show()

def processor_time_by_request(request) -> float:
    return request["input_processed_time"] - request["arrival_time"]

def ttft_by_request(request) -> float:
    # print(request)
    return processor_time_by_request(request) + request["time_in_queue"] + (request["first_token_time"] - request["first_scheduled_time"])

def encoder_time_by_request(request) -> float:
    return request["model_encoder_time"]




# def plot_model_video_technique_by_models(results_dict, techniques_list = ['scene_change', 'sharpness', 'motion_based'], metric="ttft",
def plot_model_video_technique_by_models(results_dict, techniques_list = ['scene_change', 'sharpness_based', 'motion_based'], metric="ttft",
                                    models_orig_results = {
                                        "qwen-2b-instruct": MC_0_30_350_STATISTICS_QWEN_TWO,
                                        "qwen-7b-instruct": MC_0_30_350_STATISTICS_QWEN_SEVEN,
                                        "llava-ov-qwen2-0.5b": MC_0_30_350_STATISTICS_LLAVA_OV_0_5,
                                        "llava-ov-qwen2-7b": MC_0_30_350_STATISTICS_LLAVA_OV_7,
                                        "pixtral_12b": MC_0_30_350_STATISTICS_PIXTRAL,
                                    }):
    models = list(results_dict.keys())
    num_models = len(models)
    num_perc_tecn = len(techniques_list)

    model_metrics = {model: {} for model in models}
    for model in models:

        print(model)

        # original first
        stats_file = models_orig_results[model]
        # print("stats_file", stats_file)

        #
        # ΣΟΣ
        #
        #
        
        results = get_requests_in_list(stats_file)
        if metric == "ttft":
            if "ttft" in results[0]:
                metrics = [res["ttft"] for res in results]
            else:
                metrics = [ttft_by_request(request) for request in results]
            # print("NEW TTFT ENTRY HERE HERE HERE A")
        else:
            metrics = [res[metric] for res in results]
        #
        # ΣΟΣ
        #
        #
        
        model_metrics[model][0] = metrics
        # print(stats_file)
        # print(np.mean(model_metrics[model][0]))
        # original first

        print("before enter: ", model_metrics)

        # for idx, works in enumerate(results_dict[model]):
        for idx, works in enumerate(techniques_list):

            print("works", works)
            
            print("results_dict[model][works]", results_dict[model][works])
            
    #         if "vid-mc-0-30_350_uni_4" in results_dict[model][works].keys():
            if works == "scene_change":
                stats_file = results_dict[model][works]["vid-mc-0-30_350_scc_2_4"]["stats_new"]
            elif works == "sharpness_based":
                stats_file = results_dict[model][works]["vid-mc-0-30_350_shb_1_4"]["stats_new"]
            elif works == "motion_based":
                stats_file = results_dict[model][works]["vid-mc-0-30_350_mbd_1_4"]["stats_new"]
            else:
                raise ValueError(f"Unknown technique: {works}")
            print("stats_file sh", stats_file)
            
            # stats_file = results_dict[model][works]["stats_new"]
            results = get_requests_in_list(stats_file)
            if metric == "ttft":
                if "ttft" in results[0]:
                    metrics = [res["ttft"] for res in results]
                else:
                    metrics = [ttft_by_request(request) for request in results]
                print("NEW TTFT ENTRY HERE HERE HERE gio")
            else:
                metrics = [res[metric] for res in results]
            print(len(metrics), metrics)
            model_metrics[model][idx+1] = metrics
            # print(stats_file)
            # print(np.mean(model_metrics[model][idx+1]))
        print()
    
    print_dictionary_content(model_metrics, limit_print=5)

    techniques_list_new = techniques_list.copy()
    techniques_list_new.insert(0, "unifrom")
    num_perc_techns_new = len(techniques_list_new)

    # bar_width = 0.8 / max(1, num_perc_tecn)
    bar_width = 0.8 / max(1, num_perc_techns_new)
    
    # x = np.arange(num_models)  # one x position per model
    group_spacing = 1.1  # try 1.5 or 2.0 for wider spacing
    x = np.arange(num_models) * group_spacing
    
    colors_all = COLORS_ALL

    fig, ax = plt.subplots(figsize=[param_dictionary["figsize_mul"]*6.4, 4.8], constrained_layout=True)

    # print()

    for j, model in enumerate(models):
        for i_techn, p_techn in enumerate(techniques_list_new):
            
            vals = model_metrics[model][i_techn]
            
            # convert to numpy and drop NaNs
            arr = np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)
            arr = arr[~np.isnan(arr)]

            height = float(np.mean(arr))   # or np.median(arr) if you prefer
            # xpos = x[j] + i * bar_width
            xpos = x[j] + i_techn * bar_width

            ax.bar(
                xpos,
                height,
                bar_width,
                # label=f"{idx_to_reduction[p]}%" if j == 0 else None,  # label each percentage only once
                label=f"{p_techn}" if j == 0 else None,  # label each percentage only once
                # color=colors_all[i])
                color=colors_all[i_techn])

    # ax.set_xticks(x + bar_width * (num_perc_tecn - 1) / 2.0)
    ax.set_xticks(x + bar_width * (num_perc_techns_new - 1) / 2.0)
    # ax.set_xticklabels(models, fontsize=12)
    ax.set_xticklabels([map_model_alias_to_name(model) for model in models], fontsize=12)
    # ax.set_xticklabels([map_model_alias_to_name(model) for model in models], fontsize=12, rotation=45)
    ax.tick_params(axis="x", labelsize=param_dictionary["x_params_label_size"])
    
    # ax.set_xlabel("Model", size=param_dictionary["xlabel_size"])
    
    if metric == "ttft":
        ax.set_ylabel("Avg TTFT", size=param_dictionary["ylabel_size"])
    elif metric == "modality_tokens_cnt":
        ax.set_ylabel("Avg Modality Tokens", size=param_dictionary["ylabel_size"])
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),  # x=0.5 centers it, y=1.02 places it just above the plot
        bbox_transform=ax.transAxes,
        ncol=6,                      # number of columns if you have multiple legend items
        fontsize=16,
        frameon=False                # optional: removes the legend box
    )
    plt.yticks(fontsize=param_dictionary["y_params_label_size"])
    # plt.tight_layout()
    plt.show()