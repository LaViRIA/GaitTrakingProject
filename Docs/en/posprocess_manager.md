# Parallel Processing Script Documentation

This Python script automates the parallel execution of the `posProcess.py` script (previously documented) on multiple pairs of video files (vid) and inertial sensor data (mcd). It uses the `subprocess` library to execute `posProcess.py` as separate processes, and `psutil` to control the CPU affinity of each process, thus allowing parallel processing and taking advantage of multiple CPU cores.

## Dependencies

*   **tkinter.filedialog:** To open a directory selection dialog.
*   **glob:** To find files matching a pattern (used to find MCD files).
*   **subprocess:** To execute system commands (in this case, to execute `posProcess.py`).
*   **pathlib.Path:** To check if files exist efficiently and compatibly across operating systems.
*   **time:** To measure total execution time and introduce pauses.
*   **argparse:** To parse command-line arguments.
*   **os:** To get the number of available CPU cores (`os.cpu_count()`) and for OS-related operations.
*    **psutil:** To manage processes.
*    **sys:** To get the platform on which it is running.

## Functions

### `get_subject_n_capture(str_path)`

*   **Purpose:** Extracts the subject name and capture number from an MCD file path.
*   **Arguments:**
    *   `str_path` (`str`): The full path to the MCD file.
*   **Return:**
    *   A tuple `(subject, capture_num)`:
        *   `subject` (`str`): The subject name (the name of the directory containing the files).
        *   `capture_num` (`str`): The capture number (the last digit of the MCD filename, before the extension).
*   **Operation:**
    1.  Uses `str_path.split(split_char)` to split the path into its components, using the correct directory separator (OS dependent).
    2.  Extracts the subject name: `split[-2]`.
    3.  Extracts the relevant part of the filename: `split[-1].split('.')[0][-1]`.
    4.  Returns the tuple `(subject, capture_num)`.

### `get_vid_path(main_folder, capture)`

*   **Purpose:** Constructs the full path to a video file (vid) from the main dataset folder and capture information.
*   **Arguments:**
    *   `main_folder` (`str`): The path to the main dataset folder.
    *   `capture` (`tuple`): A tuple containing the subject name and capture number (obtained from `get_subject_n_capture`).
*   **Return:**
    *   `str`: The full path to the video file.
* **Operation:**
    1.  Uses f-strings to construct the path to the video file, combining the main folder, subject name, capture number, and video file basename (`vid-{capture[1]}.mp4`).

### `get_mcd_path(main_folder, capture)`

*   **Purpose:** Constructs the full path to an MCD file from the main dataset folder and capture information.
*   **Arguments:**
    *   `main_folder` (`str`): The path to the main dataset folder.
    *   `capture` (`tuple`): A tuple containing the subject name and capture number (obtained from `get_subject_n_capture`).
*   **Return:**
    *   `str`: The full path to the MCD file.
* **Operation:**
    1.  Uses f-strings to construct the path, similar to `get_vid_path`, but using the MCD file basename (`mcd-{capture[1]}.csv`).

### `run_script(mcd, vid, cores)`

*   **Purpose:** Executes the `posProcess.py` script in a new process with specific CPU affinity.
*   **Arguments:**
    *   `mcd` (`str`): The path to the MCD file.
    *   `vid` (`str`): The path to the video file.
    *   `cores` (`list`): A list of two integers specifying the range of CPU cores to assign to the process (inclusive lower and upper bounds).
*   **Return:**
    *   A `subprocess.Popen` object representing the running process.
* **Operation:**
    1. Determines the command to execute based on the platform.
    2.  Uses `subprocess.Popen` to start a new process executing `posProcess.py` with `mcd` and `vid` arguments. `Popen` allows executing commands in separate subprocesses, non-blocking.
    3.  Gets the PID (Process ID) of the created process.
    4.  Uses `psutil.Process(pid)` to get a `psutil.Process` object representing the newly created process.
    5.  Uses the `p.cpu_affinity()` method to set the CPU affinity of the process. `p.cpu_affinity([i for i in range(cores[0], cores[1]+1)])` assigns the process to the CPU cores specified in the `cores` list. This ensures the process will run only on the specified cores.
    6.  Returns the `subprocess.Popen` object.

### `main`

*   **Purpose:** Main function of the script. Coordinates file selection, process configuration, parallel execution, and process state management.
*   **Command Line Arguments:**
    *   `process` (`int`): The number of processes to run in parallel.
    *   `core_min` (`int`): The index of the first CPU core to use (lower bound).
    *   `core_max` (`int`): The index of the last CPU core to use (upper bound).
*   **Variables:**
    *   `main_folder`: Dataset folder.
    *   `files`: List with all mcd files.
    *   `captures`: List of tuples, where each tuple contains the subject name and capture number.
    *   `num_process`: Number of parallel processes.
    *   `cores`: List containing the indices of cores to use.
    *   `process_list`: List to store `subprocess.Popen` objects.
    *   `process_running`: List of flags (0 or 1) to track the state of each process (0: stopped, 1: running).
    *   `process_file`: List saving the tuple (subject, capture) currently running.
    *   `running_i`: Index for the progress bar animation.
    *   `running_chars`: Characters for the progress bar animation.
    *   `t_init`: Script start time.
    * `print_eraser`: String to clear the console line.

*   **Main Flow:**
    1.  **Directory Selection:** Uses `filedialog.askdirectory()` to allow the user to select the main dataset directory (`main_folder`).
    2.  **Find MCD Files:** Uses `glob.glob()` to find all MCD files within the selected directory and its subdirectories. The search is recursive within dataset folders (`f'{main_folder}{split_char}*{split_char}mcd-*.csv'`). Sorts them.
    3.  **Capture Information Extraction:** Uses list comprehension and the `get_subject_n_capture` function to create the `captures` list, extracting the subject and capture number from each found MCD file.
    4.  **Command Line Arguments:** Uses `argparse` to process command line arguments: `process`, `core_min`, and `core_max`.
    5.  **Process Configuration:**
        *   Sets `num_process` from the `process` argument.
        *   Validates and sets `cores` from `core_min` and `core_max` arguments. Ensures core_min is >= 0 and core_max is less than the computer's core count.
        *   Initializes `process_list`, `process_running`, and `process_file` with initial values (`None` or 0).
    6.  **Main Loop (`while` loop):**
        *   The loop continues until all captures have been processed (`i < len(captures)`) *or* until all processes have finished (`sum(process_running) > 0`).
        *   **Process Creation:**
            *   If fewer processes are running than the maximum allowed (`sum(process_running) < num_process`) and captures remain to be processed (`i < len(captures)`):
                *   Gets paths to video and MCD files for the next capture using `get_vid_path` and `get_mcd_path`.
                *   **File Existence Check:** Checks if video and MCD files exist using `Path(vid_file).exists()` and `Path(mcd_file).exists()`. If any file does not exist, prints an error message and skips to the next capture.
                *   Finds an available index in `process_running` (an index where the value is 0, indicating no process running in that position).
                *   Starts a new process using `run_script()`, passing file paths and CPU cores to use.
                *   Updates `process_list`, `process_running`, and `process_file` to reflect the new running process.
                *   Increments `i` to move to the next capture.
        *   **Show Status:**
           * Clears the previous line.
            *   Creates a text string (`print_str`) showing the status of each process (running or stopped) along with a simple animation (`running_chars`).
            *   Prints the status string to the console, using `end='\r'` to overwrite the previous line and create the animation.
            *  Updates `running_i` for the next iteration.
        *   **Check Process Completion:**
            *   Iterates over `process_running`.
            *   If a process is marked as running (`p` is non-zero):
                *   Uses `process_list[idx].poll()` to check if the process has finished. `poll()` returns `None` if the process is still running, or the exit code if it has finished.
                *   If the process has finished (`process_list[idx].poll() is not None`):
                    *   Clears the console line.
                    *   Prints a message indicating the process has finished.
                    *   Clears `process_list`, `process_running`, and `process_file` lists for that index.
        * Pause of 0.25 seconds.
    7.  **Print Total Time:** Once the main loop finishes, prints the total execution time.

### Execution
Running this script requires 3 parameters as indicated in the previous section, so the command would be as follows:
`python posProcess_manager.py <number of processes> <lower core limit> <upper core limit>`
Example:
`py posProcess_manager.py 12 0 24`

> **Note:** The core range must be between 0 and the maximum number of cores, although if it is less than 0 or greater than the maximum number, it will be limited to that range.

### Input File System
The file system requires that the `mcd-*.csv` file and the corresponding `vid-*.mp4` file be located in the same path.
Example:
```Bash
DataSet
├───ID0
│   ├───vid-0.mp4
│   ├───mcd-0.csv
│   ├───vid-1.mp4
│   ├───mcd-1.csv
│   ├───vid-2.mp4
│   └───mcd-2.csv
├───ID1
│   ├───vid-0.mp4
│   ├───mcd-0.csv
│   ├───vid-1.mp4
│   ├───mcd-1.csv
│   ├───vid-2.mp4
│   └───mcd-2.csv
└───ID2
    ├───vid-0.mp4
    ├───mcd-0.csv
    ├───vid-1.mp4
    ├───mcd-1.csv
    ├───vid-2.mp4
    └───mcd-2.csv
```
In summary, this script orchestrates the parallel execution of `posProcess.py` on a set of video and MCD files. It allows the user to specify the number of parallel processes and CPU cores to use. The script manages process creation, CPU core assignment, process status verification, and overall execution synchronization. Using `subprocess.Popen` instead of `multiprocessing.Process` allows greater control over subprocess execution, particularly regarding CPU affinity, and avoids issues related to the Python GIL (Global Interpreter Lock) in CPU-intensive scenarios. The interface is simple, using a file selector and command-line arguments.
