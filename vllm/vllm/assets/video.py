# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Literal

import cv2
import numpy as np
import numpy.typing as npt
from huggingface_hub import hf_hub_download
from PIL import Image

from vllm.multimodal.video import sample_frames_from_video

from .base import get_cache_dir

from scenedetect import open_video
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from typing import Dict

import av # Required for keyframe extraction. Install with: pip install av
from typing import Set, Tuple
from decord import VideoReader, cpu
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
import torch
import math
import os

@lru_cache
def download_video_asset(filename: str) -> str:
    """
    Download and open an image from huggingface
    repo: raushan-testing-hf/videos-test
    """
    video_directory = get_cache_dir() / "video-example-data"
    video_directory.mkdir(parents=True, exist_ok=True)

    video_path = video_directory / filename
    video_path_str = str(video_path)
    if not video_path.exists():
        video_path_str = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename=filename,
            repo_type="dataset",
            cache_dir=video_directory,
        )
    return video_path_str

def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    """
    Reads a video file and returns a NumPy array of frames.

    Args:
        path (str): The path to the video file.
        num_frames (int, optional): The number of frames to return. If `-1`, all frames are returned.

    Returns:
        npt.NDArray: A NumPy array of shape `(num_frames, height, width, channels)` containing the frames.

    Raises:
        ValueError: If the video file cannot be opened or if no frames are read from the video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    frames = np.stack(frames)
    frames = sample_frames_from_video(frames, num_frames)
    if len(frames) < num_frames:
        raise ValueError(f"Could not read enough frames from video file {path}"
                         f" (expected {num_frames} frames, got {len(frames)})")
    return frames

def _read_and_uniformly_sample_frames(path: str, max_frames: int = -1) -> npt.NDArray:
    """
    Reads all frames from a video and then uniformly samples a specified number of frames.
    If max_frames is -1 or greater than total frames, all frames are returned.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    if not all_frames:
        raise ValueError(f"No frames read from video file {path}.")
    
    frames_np = np.stack(all_frames)
    total_frames = frames_np.shape[0]

    if max_frames == -1 or max_frames >= total_frames:
        return frames_np
    
    # Generate integer indices evenly spaced from 0 to total_frames - 1
    # np.linspace(start, stop, num) generates num evenly spaced samples over the interval [start, stop]
    frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    # frames[frame_indices, :, :, :] is equivalent to frames[frame_indices, ...]
    sampled_frames = frames_np[frame_indices, ...]
    
    # A sanity check, though linspace with dtype=int should usually produce max_frames elements
    if len(sampled_frames) < max_frames:
        print(f"Warning: Expected {max_frames} frames but got {len(sampled_frames)} after uniform sampling. Video might be too short.")

    return sampled_frames

############################################################################################################################################
############################################################################################################################################

def sample_frames_by_scene_change_open_cv_version(
    video_path: str,
    *,
    max_frames: int = 32,
    threshold: float = 27.0,
) -> np.ndarray:
    """
    The difference from the sample_frames_by_scene_change_decord_version is that it doesn't count
    timestamps and duration. It takes shorter amount of time. Since the extra two measurements are
    not useful in this project for the vllm statistics, we prefer this function.

    Sample `max_frames` frames from `video_path` by first detecting scene cuts and
    then selecting (1) the first, (2) middle, and (3) last frame of every scene.
    Falls back to uniform sampling when no scenes are found.

    Scenes are detected using the ContentDetector from scenedetect, which uses a threshold
    of 27.0 by default. The first frame of each scene is selected. If the scene is long
    enough, the middle frame is also selected. The last frame of each scene is selected
    if the scene is long enough

    Concept: Identify moments where the visual content changes significantly (scene cuts).
    Sample one or more frames from each distinct scene

    Parameters
    ----------
    video_path : str
        Path to the input video.
    max_frames : int, default 32
        Hard cap on the number of frames returned.
    threshold : float, default 27.0
        Sensitivity of the cut detector.  ↓ threshold  ⇒ more cuts detected.

    Returns
    -------
    np.ndarray
        Array of RGB frames with shape (N, H, W, 3), where N ≤ `max_frames` containing the sampled frames
    """
    # ------------------------------------------------------------------
    # 1. Scene-detect quickly on down-scaled frames
    # ------------------------------------------------------------------
    video = open_video(video_path)
    scene_manager = SceneManager()
    # ContentDetector: detects fast cuts using weighted average of HSV change. The ContentDetector
    # works by comparing successive frames of a video. If difference <= threshold, the frames
    # are considered part of the same scene. Higher threshold: Fewer scene changes will be
    # detected. The detector will be less sensitive to minor changes and will only mark very distinct cuts
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()  # (start_tc, end_tc) tuples
    # ------------------------------------------------------------------
    # 2. Pick candidate indices: first / mid / last of each scene
    # ------------------------------------------------------------------
    idx: Set[int] = set()
    for start_tc, end_tc in scene_list:
        start, end = start_tc.get_frames(), end_tc.get_frames()          # end is exclusive
        idx.add(start)                                                   # first
        if end - start > 3:                                              # middle
            idx.add((start + end) // 2)
        if end - start >= 1:                                             # last in scene
            idx.add(end - 1)
    # ------------------------------------------------------------------
    # 3. Thin if necessary – uniformly across the *already* selected idx
    # ------------------------------------------------------------------
    if len(idx) > max_frames:
        sorted_idx = sorted(idx)
        keep = np.linspace(0, len(sorted_idx) - 1, max_frames, dtype=int)
        idx = {sorted_idx[i] for i in keep}
    # ------------------------------------------------------------------
    # 4. Fallback: no scenes (or empty idx) → uniform sampling
    # ------------------------------------------------------------------
    # Maybe can be improved
    if not idx:
        # total = int(video.props.num_frames)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = set(np.linspace(0, total - 1, max_frames, dtype=int))

    sorted_idx: List[int] = sorted(idx)
    # ------------------------------------------------------------------
    # 5. Grab frames efficiently (single forward scan rather than random seeks)
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    next_target = 0
    target_count = len(sorted_idx)

    for frame_no in range(sorted_idx[-1] + 1):          # iterate once
        ok, frame = cap.read()
        if not ok:
            break                                       # EOF / corruption
        if frame_no == sorted_idx[next_target]:
            frames.append(frame[..., ::-1])             # BGR→RGB if needed
            next_target += 1
            if next_target == target_count:             # collected all
                break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return np.stack(frames)

def sample_frames_by_scene_change_decord_version(
    video_path: str,
    *,
    max_frames: int = 32,
    threshold: float = 27.0,
) -> Tuple[np.ndarray, List[str], float]:
    """
    The difference from sample_frames_by_scene_change_open_cv_version is that it
    returns some extra measurements, which are useful in VLMEvalKit and not in vllm
    statistics.

    Sample frames from a video based on scene changes using a content detector.

    This function performs scene detection on a given video file to identify
    distinct scenes by analyzing changes in content. It then selects key frames 
    from each identified scene and returns them. If no scenes are detected or 
    the number of selected frames exceeds the specified limit, frames are
    uniformly sampled from the video.

    Args:
        video_path (str): The path to the video file.
        max_frames (int, optional): The maximum number of frames to return. 
            Defaults to 32.
        threshold (float, optional): The threshold for the ContentDetector. 
            Higher values mean less sensitivity to changes, resulting in 
            fewer detected scene changes. Defaults to 27.0.

    Returns:
        Tuple[np.ndarray, List[str], float]: A tuple containing:
            - A numpy array of shape (N, H, W, 3) with the sampled frames.
            - A list of strings representing the timestamp of each frame in seconds.
            - The total duration of the video in seconds.
    """
    # -----------------------------
    # Step 1: Scene Detection
    # -----------------------------
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()  # List[Tuple[Timecode, Timecode]]

    idx: Set[int] = set()
    for start_tc, end_tc in scene_list:
        start, end = start_tc.get_frames(), end_tc.get_frames()
        idx.add(start)
        if end - start > 3:
            idx.add((start + end) // 2)
        if end - start >= 1:
            idx.add(end - 1)

    # -----------------------------
    # Step 2: Fallback or Downsample
    # -----------------------------
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps_val = vr.get_avg_fps()
    video_duration = total_frames / fps_val

    if not idx:
        idx = set(np.linspace(0, total_frames - 1, max_frames, dtype=int))

    if len(idx) > max_frames:
        sorted_idx = sorted(idx)
        keep = np.linspace(0, len(sorted_idx) - 1, max_frames, dtype=int)
        idx = {sorted_idx[i] for i in keep}

    sorted_idx = sorted(idx)

    # -----------------------------
    # Step 3: Extract Frames and Timestamps
    # -----------------------------
    frames_np = vr.get_batch(sorted_idx).asnumpy()  # (N, H, W, 3), RGB
    frame_times = [f"{i / fps_val:.2f}s" for i in sorted_idx]

    return frames_np, frame_times, video_duration

############################################################################################################################################
############################################################################################################################################

def video_to_ndarrays_all_frames(path: str) -> npt.NDArray:
    """
    Reads all frames from a video file (used as fallback or when target_fps is very high) and returns them as a NumPy array.

    Args:
        path (str): The path to the video file.

    Returns:
        npt.NDArray: A NumPy array containing all frames from the video
                     of shape (num_frames, height, width, channels).

    Raises:
        ValueError: If the video file cannot be opened or if no frames
                    are read from the video.
    """

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames read from video file {path}.")
        
    return np.stack(frames)

def video_to_ndarrays_fixed_rate_sampling(path: str, target_fps: float) -> npt.NDArray:
    """
    Reads frames from a video file, sampling them at a specified target FPS.
    Specify a target sampling rate (e.g., 1 frame per second). This ensures
    consistent temporal density regardless of video length

    Args:
        path (str): The path to the video file.
        target_fps (float): The desired frames per second to sample from the video.
                            If the video's original FPS is lower than target_fps,
                            all frames will be included.

    Returns:
        npt.NDArray: A NumPy array containing the sampled video frames
                     of shape (num_sampled_frames, height, width, channels).

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames are: {total_frames}")

    if original_fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError(f"Could not get valid FPS or total frames from video file {path}")

    if target_fps <= 0: # Determine which frames to sample based on target_fps
        cap.release()
        raise ValueError("Target FPS must be positive.")

    original_duration_seconds = total_frames / original_fps # original duration in seconds
    # print(f"Original duration in seconds: {original_duration_seconds}")
    num_frames_to_sample = int(np.ceil(original_duration_seconds * target_fps)) # how many frames we'd get at the target_fps

    # If the target_fps implies sampling more frames than available or if target_fps is very high, just take all original frames
    if num_frames_to_sample >= total_frames:
        print(f"  Warning: Target FPS ({target_fps:.2f}) results in sampling >= total frames ({total_frames})."
              " Returning all frames.")
        return video_to_ndarrays_all_frames(path)   # Fallback to reading all frames if sampling too aggressively
    
    # Calculate the indices of the frames to be sampled. We use linspace to get evenly distributed
    # indices across the *time* of the video from the first frame (index 0) to the last frame (total_frames - 1).
    frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

    sampled_frames = []
    
    # Efficiently read only the required frames by setting the position
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx) # Set the read pointer to the specific frame index
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:   # Handle cases where a frame might not be readable even if index is valid
            print(f"Warning: Could not read frame at index {idx} from {path}. Skipping.")
            
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were sampled from video file {path}. Check video integrity or sampling parameters.")

    return np.stack(sampled_frames)

def _thin_out_frames(selected_indices: list[int], max_frames: int) -> list[int]:
    """
    If a selection strategy yields too many frames, this function thins them out
    uniformly to a maximum number of frames. Ensures unique and sorted indices.

    Args:
        selected_indices (list[int]): A list of frame indices identified by a strategy.
        max_frames (int): The maximum number of frames desired in the output.

    Returns:
        list[int]: A new list of unique, sorted, and thinned frame indices.
    """
    unique_sorted_indices = sorted(list(set(selected_indices)))

    if len(unique_sorted_indices) <= max_frames:
        return unique_sorted_indices

    # Use linspace to select evenly spaced indices from the already selected ones
    # np.round is used to ensure we get integer indices from the float array
    thinning_indices_float = np.linspace(0, len(unique_sorted_indices) - 1, max_frames)
    thinning_indices = np.round(thinning_indices_float).astype(int)
    
    return [unique_sorted_indices[i] for i in thinning_indices]

def sample_frames_by_motion(video_path: str, max_frames: int = 32, motion_threshold: float = 1.0) -> npt.NDArray:
    """
    Concept: Prioritize frames where significant motion or relevant activity occurs, assuming these
    frames carry more information about actions.

    How: Calculate motion vectors (e.g., optical flow) between frames. Frames with high optical flow
    magnitude could indicate important action.

    Samples frames based on the amount of motion detected between consecutive frames using
    dense optical flow (Farneback method). Frames with motion exceeding a `motion_threshold`
    are prioritized. If too many frames are selected, they are uniformly thinned out.

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to return.
        motion_threshold (float): Average optical flow magnitude threshold. A frame is
                                  considered "active" if the mean magnitude of motion
                                  vectors between it and the previous frame exceeds this value.
                                  Typical values range from 0.1 (very sensitive) to 5.0 (only
                                  major movements). This parameter often requires tuning.

    Returns:
        npt.NDArray: A NumPy array of sampled frames.
    
    Raises:
        ValueError: If video cannot be opened or first frame cannot be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    ret, prev_frame_bgr = cap.read()    # Read the first frame
    if not ret:
        cap.release()
        raise ValueError(f"Could not read first frame from {video_path}.")
    
    prev_frame_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    selected_indices = [0] # Always include the first frame as a starting point
    
    current_frame_idx = 1
    while True:
        ret, next_frame_bgr = cap.read()
        if not ret:
            break # End of video
        
        next_frame_gray = cv2.cvtColor(next_frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate Farneback optical flow (dense flow)
        # Parameters for Farneback (tuned for general use, can be adjusted):
        # pyr_scale=0.5: pyramid scale, reduces image size by half at each level
        # levels=3: number of pyramid layers
        # winsize=15: averaging window size; larger -> smoother motion, less noise, more computation
        # iterations=3: number of iterations at each pyramid level
        # poly_n=5, poly_sigma=1.2: polynomial expansion size and standard deviation for Gaussian
        # flags=0: typically 0 for standard optical flow (or cv2.OPTFLOW_FARNEBACK_GAUSSIAN for Gaussian window)
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, next_frame_gray, 
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate the magnitude (length) of the flow vectors
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate the average motion magnitude across the entire frame
        avg_motion = np.mean(magnitude)
        
        # Select frame if average motion exceeds the threshold
        if avg_motion > motion_threshold:
            selected_indices.append(current_frame_idx)
            
        prev_frame_gray = next_frame_gray # Update previous frame for next iteration
        current_frame_idx += 1
        
    cap.release()

    if len(selected_indices) <= 1: # Only the initial frame was selected, or no significant motion
        print(f"  Warning: Few or no frames selected by motion for {video_path}. Falling back to uniform sampling.")
        # Fallback to uniform if no motion detected or threshold is too high
        return _read_and_uniformly_sample_frames(video_path, max_frames=max_frames)

    # print(f"Selected {len(selected_indices)} frames based on motion. Thinning to {max_frames} if necessary.")
    # Thin out selected frames if too many
    final_indices = _thin_out_frames(selected_indices, max_frames)
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen cap for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"  Warning: Could not read frame at index {idx} from {video_path}. Skipping.")
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using motion detection.")

    return np.stack(sampled_frames)

def sample_frames_by_sharpness(video_path: str, max_frames: int = 32, sharpness_threshold: float = 100.0) -> npt.NDArray:
    """
    Concept: Discard blurry or very low-quality frames, even if they are uniformly sampled. Prioritize
    sharper frames.

    Samples frames based on their sharpness, prioritizing sharper frames. Sharpness is measured
    using the variance of the Laplacian operator, which is sensitive to edges and high-frequency
    content. If too many sharp frames are found, they are uniformly thinned out

    Args:
        video_path (str): Path to the video file.
        max_frames (int): Maximum number of frames to return.
        sharpness_threshold (float): Minimum Laplacian variance for a frame to be
                                     considered "sharp enough" to be selected.
                                     Higher values mean only very sharp frames are selected.
                                     Typical values vary widely depending on resolution
                                     and content (e.g., 50 for blurry, 500+ for very sharp).
                                     This parameter often requires tuning.
    Returns:
        npt.NDArray: A NumPy array of sampled frames.
    
    Raises:
        ValueError: If video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    selected_indices = []
    current_frame_idx = 0
    # print(f"  Analyzing sharpness for {video_path} (threshold={sharpness_threshold})...")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        # Convert to grayscale, as Laplacian works on single-channel images
        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian operator
        # cv2.CV_64F is used as the depth of the output image to avoid overflow
        # when computing variance, as Laplacian can produce negative values.
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        
        # The variance of the Laplacian indicates the amount of edges and thus sharpness.
        sharpness_score = laplacian.var()
        
        # Select frame if its sharpness score exceeds the threshold
        if sharpness_score > sharpness_threshold:
            selected_indices.append(current_frame_idx)
            
        current_frame_idx += 1
        
    cap.release()

    if not selected_indices:
        print(f"  Warning: No frames selected by sharpness for {video_path}. Falling back to uniform sampling.")
        # Fallback to uniform sampling if no sharp frames are found or threshold is too high
        return _read_and_uniformly_sample_frames(video_path, max_frames=max_frames)

    # print(f"Selected {len(selected_indices)} frames based on sharpness. Thinning to {max_frames} if necessary.")
    final_indices = _thin_out_frames(selected_indices, max_frames)  # Thin out selected frames if too many
    
    # Read only the selected frames using OpenCV for efficiency
    sampled_frames = []
    cap = cv2.VideoCapture(video_path) # Reopen for specific frame access
    if not cap.isOpened():
        raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

    for idx in final_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)
        else:
            print(f"  Warning: Could not read frame at index {idx} from {video_path}. Skipping.")
    cap.release()

    if not sampled_frames:
        raise ValueError(f"No frames were successfully sampled from {video_path} using sharpness detection.")

    return np.stack(sampled_frames)

def video_to_ndarrays_clever_sample(path: str, strategy: str = "uniform", param_dict: Dict[str, float] = {}) -> npt.NDArray:
    """
    Samples frames from a video file using various 'clever' strategies. Choosing the "clever" method depends
    on what kind of information is most critical for your LLaVA-Next model (e.g., overall context, specific actions,
    object appearances). Scene change detection is often a good balance between information density and frame reduction.

    Args:
        path (str): The path to the video file.
        strategy (str): The sampling strategy to use.
                         Options: "uniform", "fixed_rate", "keyframes",
                                  "motion_based", "sharpness_based".
        param_dict (Dict[str, float]): Additional parameters specific to the chosen strategy:
            - "uniform": num_frames (int, default: 32) - Number of frames to sample uniformly.
            - "fixed_rate": target_fps (float, default: 1.0) - Desired frames per second.
            - "keyframes": max_frames (int, default: 32) - Max keyframes to select.
            - "motion_based": max_frames (int, default: 32), motion_threshold (float, default: 1.0)
            - "sharpness_based": max_frames (int, default: 32), sharpness_threshold (float, default: 100.0)

    Returns:
        npt.NDArray: A NumPy array containing the sampled video frames.
    
    Raises:
        ValueError: If an unknown strategy is provided or video cannot be processed.
    """

    save_nam = ""
    
    vid_fold_cur = path.split(".mp4")[0].split("videos")[-1] # '/academic_source/NextQA/1202/4295889026'
    vid_fold_cur = vid_fold_cur[1:] # 'academic_source/NextQA/1202/4295889026'
    PATH_TO_SAVE_SAMPLED_FRAMES = "/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/save_here_sampled_video_frames/"
    path_split_mp4 = os.path.join(PATH_TO_SAVE_SAMPLED_FRAMES, vid_fold_cur) # '/home/ioannis.dalianis/code/las_konpap/mllm-inference-workload-eval/exploration/save_here_sampled_video_frames/academic_source/NextQA/1202/4295889026'
    
    os.makedirs(path_split_mp4, exist_ok=True)
    video_nam_no_mp4 = path.split("/")[-1].split(".mp4")[0] # '4295889026'
    save_nam += f"{video_nam_no_mp4}_{strategy}"
    for i in param_dict:
        num = str(param_dict[i]).replace(".", "_")
        save_nam += f"_{i}_{num}"
    save_nam += ".npy"

    # print(f"\n- Sampling '{path.split('/academic_source/')[-1]}' strategy -> '{strategy}'\tparams -> {param_dict} -")
    print(f"- Sampling '{path.split('/academic_source/')[-1]}' strategy -> '{strategy}'\tparams -> {param_dict} -")
    # print(f"{path_split_mp4}/{save_nam}")
    if os.path.exists(f"{path_split_mp4}/{save_nam}"):
        import logging
        logging.info(f"  Found cached sampled frames at {path_split_mp4}/{save_nam}, loading...")
        return np.load(f"{path_split_mp4}/{save_nam}")

    try:
        if strategy == "uniform":
            max_frames = param_dict.get('max_frames', 32)
            # return _read_and_uniformly_sample_frames(path, max_frames=max_frames)
            ret = _read_and_uniformly_sample_frames(path, max_frames=max_frames)
        elif strategy == "fixed_rate":
            target_fps = param_dict.get('target_fps', 1.0)
            # return video_to_ndarrays_fixed_rate_sampling(path, target_fps=target_fps)
            ret = video_to_ndarrays_fixed_rate_sampling(path, target_fps=target_fps)
        # elif strategy == "keyframes":
        #     max_frames = kwargs.get('max_frames', 32)
        #     return sample_frames_by_keyframes(path, max_frames=max_frames)
        elif strategy == "motion_based":
            max_frames = param_dict.get('max_frames', 32)
            motion_threshold = param_dict.get('motion_threshold', 1.0)
            # return sample_frames_by_motion(path, max_frames=max_frames, motion_threshold=motion_threshold)
            ret = sample_frames_by_motion(path, max_frames=max_frames, motion_threshold=motion_threshold)
        elif strategy == "sharpness_based":
            max_frames = param_dict.get('max_frames', 32)
            sharpness_threshold = param_dict.get('sharpness_threshold', 100.0)
            # return sample_frames_by_sharpness(path, max_frames=max_frames, sharpness_threshold=sharpness_threshold)
            ret = sample_frames_by_sharpness(path, max_frames=max_frames, sharpness_threshold=sharpness_threshold)
        elif strategy == "scene_change":
            max_frames = param_dict.get('max_frames', 32)
            content_threshold = param_dict.get('content_threshold', 27.0)
            # downscale_factor = kwargs.get('downscale_factor', 2)
            ###################################################################################################
            # return sample_frames_by_scene_change_open_cv_version(path, max_frames=max_frames, threshold=content_threshold)
            ret = sample_frames_by_scene_change_open_cv_version(path, max_frames=max_frames, threshold=content_threshold)
            # return sample_frames_by_scene_change_decord_version(path, max_frames=max_frames, threshold=content_threshold)[0]
            ###################################################################################################
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        np.save(f"{path_split_mp4}/{save_nam}", ret)
        return ret
    except ValueError as e:
        print(f"Error during video processing for strategy '{strategy}': {e}")
        return np.array([]) # Return an empty array to indicate failure, or re-raise if you prefer
    except Exception as e:
        print(f"An unexpected error occurred during strategy '{strategy}': {e}")
        return np.array([]) # Return an empty array or re-raise

def video_to_pil_images_list(path: str,
                             num_frames: int = -1) -> List[Image.Image]:
    frames = video_to_ndarrays(path, num_frames)
    return [
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for frame in frames
    ]


@dataclass(frozen=True)
class VideoAsset:
    name: Literal["sample_demo_1.mp4"]
    num_frames: int = -1

    @property
    def pil_images(self) -> List[Image.Image]:
        video_path = download_video_asset(self.name)
        ret = video_to_pil_images_list(video_path, self.num_frames)
        return ret

    @property
    def np_ndarrays(self) -> npt.NDArray:
        video_path = download_video_asset(self.name)
        ret = video_to_ndarrays(video_path, self.num_frames)
        return ret

@dataclass(frozen=True)
class VideoAsset_CleverSample:
    param_dict: Dict[str, float]
    name: Literal["sample_demo_1.mp4"]
    strategy: str = "uniform"

    @property
    def np_ndarrays(self) -> npt.NDArray:
        video_path = download_video_asset(self.name)
        ret = video_to_ndarrays_clever_sample(video_path, strategy=self.strategy, param_dict=self.param_dict)
        # ret.shape: (4, 480, 640, 3)
        return ret

@dataclass(frozen=True)
class VideoAsset_for_Qwen_plus_CleverSample:
    # class theat takes into account potential resizes of video frames that need to happen
    param_dict: Dict[str, float]
    name: Literal["sample_demo_1.mp4"]
    strategy: str = "None"
    num_frames: int = -1

    @property
    def np_ndarrays(self) -> npt.NDArray:
        video_path = download_video_asset(self.name)
        if self.strategy != "None":
            # clever sampling
            ret = video_to_ndarrays_clever_sample(video_path, strategy=self.strategy, param_dict=self.param_dict)
            nframes, height, width, _ = ret.shape

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )

            processed_frames = []
            for frame in ret:  # iterate over T frames
                # Convert (H, W, C) numpy → torch (C, H, W)
                tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float()
                # tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

                resized = transforms.functional.resize(
                    tensor_frame,
                    [resized_height, resized_width],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ).float()
                # processed_frames.append(resized.permute(1, 2, 0).numpy()) # Back to numpy (H, W, C)
                processed_frames.append(resized.permute(1, 2, 0).byte().numpy()) # Back to numpy uint8

            # Stack back into (T, H, W, C)
            ret = np.stack(processed_frames)

        else:
            # according to /home/ioannis.dalianis/code/VLMEvalKit/.env_pixtral_vllm/lib/python3.11/site-packages/qwen_vl_utils/vision_process.py
            nframes = round_by_factor(self.num_frames, FRAME_FACTOR)
            
            video, audio, info = io.read_video(
                video_path,
                pts_unit="sec",
                output_format="TCHW",
            )
            total_frames = video.size(0)
            idx = torch.linspace(0, total_frames - 1, nframes).round().long()
            video = video[idx] # len(video) = 567 and idx will have 64 indices etc

            nframes, _, height, width = video.shape

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            
            ret = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
            ret = ret.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return ret # ret.shape: (64, 336, 448, 3) ret.max() = 255.0 ret.min() = 0.0

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# def sample_frames_by_keyframes(video_path: str, max_frames: int = 32) -> npt.NDArray:
#     """
#     Concept: Many videos are compressed using keyframes (I-frames) which are full images,
#     and then predicted frames (P-frames, B-frames) that just store changes. Keyframes often
#     represent significant scene changes or important moments
#     Extracts keyframes (I-frames) from a video file using PyAV. Keyframes often
#     represent significant scene changes or are strategically placed by the encoder.
#     If the number of keyframes found exceeds `max_frames`, they are uniformly
#     thinned out to meet the limit.

#     Args:
#         video_path (str): Path to the video file.
#         max_frames (int): Maximum number of keyframes to return. If more are found,
#                           they are uniformly sampled from the set of keyframes.

#     Returns:
#         npt.NDArray: A NumPy array of sampled keyframes.
    
#     Raises:
#         ValueError: If the video cannot be opened with PyAV or no keyframes are found.
#     """
#     keyframe_indices = []
    
#     print(f"  Detecting keyframes for {video_path}...")
#     try:
#         container = av.open(video_path)
#         stream = container.streams.video[0]

#         # from collections import Counter
#         # counter = Counter()
#         # for frame in stream.decode():
#         #     counter[frame.pict_type] += 1
#         # print(counter)
        
#         # # Iterate through all frames using stream.decode to access frame properties
#         # # for frame_idx, frame in enumerate(stream.decode(video=0)):
#         # for frame_idx, frame in enumerate(stream.decode()):
#         #     print(f"Frame {frame_idx}: keyframe={frame.is_keyframe}, pict_type={frame.pict_type}")

#         #     if frame.is_keyframe:
#         #         keyframe_indices.append(frame_idx)
#         keyframe_indices = []
#         frame_idx = 0
#         container = av.open(video_path)
#         video_stream = container.streams.video[0]
#         for packet in container.demux(video_stream):
#             for frame in packet.decode():
#                 if frame.pict_type == "I":  # more reliable than is_keyframe
#                     keyframe_indices.append(frame_idx)
#                 frame_idx += 1
#                 # print(f"Frame {frame_idx}: pict_type={frame.pict_type}")

#         container.close()
        
#     # except AVError as e:
#     #     raise ValueError(f"Could not open or process video with PyAV: {video_path}. Error: {e}")
#     except Exception as e:
#         raise ValueError(f"Could not open or process video with PyAV: {video_path}. Error: {e}")

#     if not keyframe_indices:
#         print(f"  Warning: No keyframes found in {video_path}. Falling back to uniform sampling.")
#         # Fallback to uniform sampling if no keyframes are detected (e.g., in very short videos,
#         # or videos with unusual encoding where PyAV can't find I-frames).
#         # return _read_and_uniformly_sample_frames(video_path, max_frames=max_frames)
#         return video_to_ndarrays(video_path, num_frames=max_frames)

#     print(f"  Found {len(keyframe_indices)} keyframes. Thinning to {max_frames} if necessary.")
#     # Thin out keyframes if too many were found
#     final_indices = _thin_out_frames(keyframe_indices, max_frames)
    
#     # Read only the selected keyframes using OpenCV for efficiency (faster random access)
#     sampled_frames = []
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Could not re-open video file {video_path} for reading sampled frames.")

#     for idx in final_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, idx) # Seek to the specific frame index
#         ret, frame = cap.read()
#         if ret:
#             sampled_frames.append(frame)
#         else:
#             print(f"  Warning: Could not read keyframe at index {idx} from {video_path}. Skipping.")
#     cap.release()

#     if not sampled_frames:
#         raise ValueError(f"No keyframes were successfully sampled from {video_path}.")

#     return np.stack(sampled_frames)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 150800
FRAME_FACTOR = 2

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar