# @.@ coding  : utf-8 ^_^
# @Author     : Leon Rein
# @Time       : 24/01/28 ~ 21:34:25
# @File       : clean_at_exit.py
# @Note       : This file is used to clean up the log files when the program is terminated by `Ctrl+C`.

from typing import Callable, NoReturn, Any

import sys
import signal
import shutil
from pathlib import Path


def signal_handler(path: str | Path, extra_function: Callable[[], Any]) -> Callable[..., NoReturn]:
    kill6_counter = 0

    def signal_handler_impl(*args, **kwargs) -> NoReturn:
        print("\n---\n" "You have pressed `Ctrl+C`!")
        nonlocal kill6_counter
        kill6_counter += 1  # prevent infinite loop caused by `Ctrl+C` (kill -6)

        # First time: cathing `Ctrl+C` (kill -6)
        if kill6_counter == 1 and input("Do you want to delete the log files for this run? [y/N] ").strip().lower() in ["y", "yes"]:
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path} due to {e}.")

            extra_function()
            print(f"Cleaned up.")

        print("Exiting...")
        sys.exit(1)

    return signal_handler_impl


def cleaner(path: str | Path, extra_function: Callable[[], Any] = lambda: None):
    signal.signal(signal.SIGINT, signal_handler(path, extra_function))
