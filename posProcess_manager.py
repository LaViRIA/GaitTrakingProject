from tkinter import filedialog
import glob
import subprocess

# from multiprocessing import Process
from pathlib import Path
import time
import argparse
import os
import psutil
import sys

if sys.platform == "linux":
    split_char = "/"
else:
    split_char = "\\"


def get_subject_n_capture(str_path):
    split = str_path.split(split_char)
    num = split[-1].split(".")[0][-1]
    return (split[-2], num)


def get_vid_path(main_folder, capture):
    return f"{main_folder}{split_char}{capture[0]}{split_char}vid-{capture[1]}.mp4"


def get_VID_path(main_folder, capture):
    return f"{main_folder}{split_char}{capture[0]}{split_char}VID-{capture[1]}.mp4"


def get_mcd_path(main_folder, capture):
    return f"{main_folder}{split_char}{capture[0]}{split_char}mcd-{capture[1]}.csv"


def run_script(mcd, vid, cores):
    if sys.platform == "linux":
        python_command = "python3"
    else:
        python_command = "python"
    process = subprocess.Popen([python_command, "posProcess.py", mcd, vid])
    pid = process.pid
    p = psutil.Process(pid)
    p.cpu_affinity([i for i in range(cores[0], cores[1] + 1)])
    return process


if __name__ == "__main__":
    # seleccion del folder del dataset y obtencion de las capturas basadas en los archivos mcd
    main_folder = filedialog.askdirectory(
        initialdir="/media/moybadajoz/Nuevo vol/Dataset_25"
    )
    print(main_folder)
    files = sorted(glob.glob(f"{main_folder}{split_char}*{split_char}mcd-*.csv"))
    # print(main_folder, files)
    captures = [get_subject_n_capture(s) for s in files]

    # parametros del script
    parser = argparse.ArgumentParser(description="idk")
    parser.add_argument("--process", type=int)
    parser.add_argument(
        "--core_min",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--core_max", type=int, default=os.cpu_count() - 1, required=False
    )
    args = parser.parse_args()

    num_process = args.process
    cores = [max(0, args.core_min), min(os.cpu_count() - 1, args.core_max)]

    i = 0
    process_list = [None] * num_process
    process_running = [0] * num_process
    process_file = [None] * num_process

    running_i = 0
    running_chars = ["|", "/", "-", "\\"]
    t_init = time.time()
    print_eraser = ""
    while i < len(captures) or sum(process_running) > 0:
        if sum(process_running) < num_process and i < len(captures):
            vid_file = get_vid_path(main_folder, captures[i])
            mcd_file = get_mcd_path(main_folder, captures[i])
            if not Path(vid_file).exists():
                vid_file = get_VID_path(main_folder, captures[i])
                if not Path(vid_file).exists():
                    print(f"<<{vid_file}>> no found")
                    i += 1
                    continue
            if not Path(mcd_file).exists():
                print(f"<<{mcd_file}>> no found")
                i += 1
                continue
            idx = process_running.index(0)
            process_list[idx] = run_script(mcd_file, vid_file, cores)
            process_running[idx] = 1
            process_file[idx] = captures[i]
            i += 1

        print_str = []
        for idx, r in enumerate(process_running):
            if r == 1:
                print_str.append(
                    f"{process_file[idx]}>" + f"{idx} {running_chars[running_i]} | "
                )
            else:
                print_str.append(f"{idx} Stopped | ")
        print(print_eraser, end="\r")
        print("".join(print_str), end="\r")
        print_eraser = " " * len("".join(print_str))
        running_i = 0 if running_i >= len(running_chars) - 1 else running_i + 1

        for idx, p in enumerate(process_running):
            if not p:
                continue
            if process_list[idx].poll() is None:
                continue
            print(print_eraser, end="\r")
            print(f"{process_file[idx]} Finished")
            process_file[idx] = None
            process_list[idx] = None
            process_running[idx] = 0
        time.sleep(0.25)

    print(time.time() - t_init)
