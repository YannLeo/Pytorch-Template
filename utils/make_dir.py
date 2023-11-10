import os
from pathlib import Path
import time
from typing import Optional, Any
import shutil


def make_dir(info: dict[str, Any], config_file: str, resume: Optional[str]) -> Path:
    """
    Make a directory structure for saving training results.

    Args:
        info (dict[str, Any]): Information about the directory.
        config_file (str): Path to the configuration file.
        resume (Optional[str]): Path to the resume file, if provided.

    Returns:
        Path: The path to the created directory.
    """

    save_dir = Path("saved")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir / info["name"]):
        os.mkdir(save_dir / info["name"])
    time_list = list(time.localtime(time.time()))
    time_string = "-".join(map(lambda x: str(x).zfill(2), time_list[:6]))
    if not os.path.exists(save_dir / info["name"] / time_string):
        os.mkdir(save_dir / info["name"] / time_string)
        os.mkdir(save_dir / info["name"] / time_string / "log")
        os.mkdir(save_dir / info["name"] / time_string / "model")
        os.mkdir(save_dir / info["name"] / time_string / "confusion_matrix")

    # copy config file to save_dir
    shutil.copy(config_file, save_dir / info["name"] / time_string / "log")

    if resume:
        info["resume"] = resume
        print("--- Overwriting resume path from terminal  ---")

    if not info.get("resume"):
        info["resume"] = ""
    
    return save_dir / info["name"] / time_string
