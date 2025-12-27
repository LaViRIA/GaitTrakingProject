# Capture Application Documentation

This application is designed to interact with sensors via a serial connection, capture data from them, visualize it in real-time, and save it to CSV files. It uses `customtkinter`, `matplotlib`, `numpy`, `serial` libraries, and a custom API (`McdAPI`).

## Interface
![Capture.py Interface](../img/Capture.png)

## Main Class: `App`

The `App` class inherits from `customtkinter.CTk` and represents the main window of the application.

### `__init__`

The constructor initializes the graphical user interface (GUI), configuring the title, size, and layout of elements. The window is divided into three main frames:

*   **`frame_left` (Left):** Contains controls to select the data save directory, enter an ID, and specify anthropometric measurements (height, age, femur length, tibia length, hip width).
*   **`frame_center` (Center):** Contains controls for serial connection management (port selection, open/close), data reading control (start/stop), sensor reset and synchronization, and options to save/load sensor configuration.
*   **`frame_right` (Right):** A scrollable frame (`CTkScrollableFrame`) that dynamically displays information from connected sensors, including real-time graphs of acceleration on the X, Y, and Z axes, and battery level.

#### GUI Elements and Associated Variables:

*   **Left Menu (`frame_left`):**
    *   `btn_select_dir`: Button to select the save directory (calls `select_dir`).
    *   `label_id`, `input_id`: Label and input field for the dataset ID.
    *   `label_height`, `input_height`, `var_height`: Label, input field, and variable (`DoubleVar`) for height.
    *   `label_age`, `input_age`, `var_age`: Label, input field, and variable (`DoubleVar`) for age.
    *   `label_femur`, `input_femur`, `var_femur`: Label, input field, and variable (`DoubleVar`) for femur length.
    *   `label_tibia`, `input_tibia`, `var_tibia`: Label, input field, and variable (`DoubleVar`) for tibia length.
    *   `label_hip`, `input_hip`, `var_hip`: Label, input field, and variable (`DoubleVar`) for hip width.

*   **Center Menu (`frame_center`):**
    *   `optionMenu_serial_port`: Dropdown menu to select the serial port (updated with `btn_update_ports`).
    *   `btn_update_ports`: Button to reload the list of available serial ports.
    *   `mcd`: Instance of the `McdAPI` class for serial communication.
    *   `btn_port`: Button to open/close the serial port (calls `open_port` and `close_port`).
    *   `btn_read`: Button to start/stop data reading (calls `start_read` and `stop_read`). Enabled/disabled based on connection state.
    *   `btn_reset`: Button to reset sensors (calls `mcd.reset_sensor`). Enabled/disabled based on connection state.
    *   `btn_capture`: Button to start/stop data capture (calls `start_capture` and `stop_capture`). Enabled/disabled based on reading state.
    *   `btn_sync`: Button to activate sensor synchronization mode (calls `mcd.sync_mode`). Enabled/disabled based on connection state.
    *   `label_sensors_config`, `btn_save_config`, `btn_load_config`: Label and buttons to save and load sensor configuration (call `save_config` and `load_config`).

*   **Right Menu (`frame_right`):**
    *   Contains dynamically generated frames for each detected sensor (see `add_sensor` and `del_sensor`).

*   **State Variables and Queues:**
    *   `dict_sensor`: Dictionary storing information and widgets for each sensor. The key is the sensor address.
    *   `config`: Dictionary saving sensor configuration (which bone each sensor represents).
    *   `queue_data`: Queue (`queue.Queue`) to store data received from sensors.
    *   `queue_battery`: Queue (`queue.Queue`) to store battery information from sensors.
    *   `capture_flag`: Queue (`queue.Queue`) to control capture state (started/stopped).
    *   `process`: Thread (`threading.Thread`) executing the `read_capture_process` function for background data reading and processing.
    *    `after_update_sensors_id`, `after_graphs_id`: Identifiers for tasks scheduled with `after` that update sensors and graphs.



### Main Methods:

*   **`select_dir(self)`:** Opens a dialog to select the save directory and updates `self.dataset_path`.

*   **`open_port(self)`:** Sets the serial port in the `McdAPI` instance, attempts to connect, and updates the GUI accordingly (enables/disables buttons, starts the reading thread, etc.). Initializes data queues and the `process` thread. Calls `update_sensors`.

*   **`close_port(self)`:** Stops reading, closes the serial port, updates the GUI (buttons), and removes sensor information from `frame_right`. Cancels `after` calls.

*   **`start_read(self)`:** Starts data reading by calling `mcd.read_start()` and updates the GUI. Starts graph updates with `update_graphs`.

*   **`stop_read(self)`:** Stops data reading by calling `mcd.read_stop()` and updates the GUI. Cancels the `after` call.

*   **`start_capture(self)`:** Starts data capture. Prepares a comment with anthropometric data, save path, and filename. Saves sensor configuration and sends a signal via `capture_flag` to start writing to the CSV file.

*   **`stop_capture(self)`:** Stops data capture by sending a signal via `capture_flag`.

*   **`add_sensor(self, addr, battery)`:** Adds a new sensor to the interface (in `frame_right`). Creates a frame (`CTkFrame`) for the sensor, including:
    *   Label with the sensor address (`address`).
    *   Label with battery level (`battery`).
    *   Dropdown menu to select the bone associated with the sensor (`select_bone`, `var_bone`).
    *   Buttons to identify, reset, and calibrate the sensor (`btn_identify`, `btn_reset`, `btn_calibrate`).
    *   A sub-frame (`frame_graph`) containing a matplotlib plot (`fig`, `ax`) to visualize acceleration on X, Y, and Z axes.

*   **`del_sensor(self, addr)`:** Removes a sensor from the interface (from `frame_right`) and from the `dict_sensor` dictionary. Closes the matplotlib figure.

*   **`update_sensors(self)`:** Executed periodically (every 1000 ms) using `self.after`. Gets battery data from `queue_battery`. Adds new sensors (if they appear in received data), updates the battery level of existing sensors, and removes sensors that are no longer detected. Also assigns each sensor the corresponding skeleton part using the loaded configuration or "Select" if no configuration exists.

*   **`update_graphs(self)`:** Executed periodically (every 16 ms) using `self.after`. Gets data from `queue_data`. Updates acceleration graphs for each sensor with new data.

*   **`save_config(self)`:** Saves the current sensor configuration (which bone each sensor represents) to a CSV file. Asks the user for the filename via a dialog.

*   **`load_config(self)`:** Loads a sensor configuration from a CSV file. Asks the user for the file via a dialog. Saves the configuration in the `self.config` dictionary.

*   **`on_closing(self)`:** Window close event handler. Stops capture, reading, closes serial port, and destroys the application window. Cancels any scheduled `after` calls.

## Function: `read_capture_process`

This function runs in a separate thread and handles continuous data reading from the serial port, data packet processing, updating `queue_data` and `queue_battery` queues, and writing captured data to a CSV file.

### Arguments:

*   `mcd`: Instance of the `McdAPI` class.
*   `capture_flag`: Queue (`queue.Queue`) to control capture state.
*   `queue_data`: Queue for sensor data.
*   `queue_battery`: Queue for battery data.

### Operation:

1.  **Main Loop:** The `while True` loop runs indefinitely until the serial port is closed.

2.  **Capture Control:** Checks if there are messages in `capture_flag`. If there is a message, capture is started or stopped, depending on the message content. If capture starts, a new CSV file (with a unique name) is created and a CSV writer (`csv_writer`) is initialized. Sensor configuration (`config`) is inverted (from bone to address).

3.  **Data Reading:** Reads data from the serial port using `mcd.read()` and accumulates it in a buffer (`buffer`).

4.  **Packet Processing:** Searches for the frame start byte ('S'). Depending on the message type (data or battery), extracts packet length and waits to receive all necessary bytes.
    *   **Data Messages (type 1):** Unpacks data using `unpack_data` and `FORMAT_DATA`. Adds data to `queue_data`. If capture is active, writes a row to the CSV file with timestamp data and acceleration, gyroscope, and quaternion values for each sensor, according to the loaded configuration.
    *   **Battery Messages (type 2):** Unpacks data using `unpack_bat` and `FORMAT_BAT`. Adds data to `queue_battery`.

5.  **Error Handling:** Includes a `try...except` block to handle `KeyboardInterrupt` (to exit cleanly) and other exceptions.

## Function: `init_csv_file`

This auxiliary function initializes a CSV file for data capture.

### Arguments:

*   `path`: CSV file path.
*   `fields`: List of field names for the CSV header.
*   `delimiter`: Delimiter character.
*   `comments`: Optional comments to add to the beginning of the file.

### Return:

*   A tuple containing the open file object and the `csv.DictWriter` object.

## Main Execution (`if __name__ == "__main__":`)

Creates an instance of the `App` class, configures the window close event handler (`on_closing`), and executes the main application loop (`mainloop`). This starts the graphical interface and the reading/capture process.

### Dataset File System
When performing a series of captures for different users, these must be organized in some way. This program organizes files in the path selected with the `Select dir` button. Inside this, folders are created with names assigned to the `ID` space. Inside this folder, files are created with the name `mcd-*.csv`, where `*` is the capture number. It starts searching for name availability with `mcd-0.csv`; if not available, it continues with `mcd-1.csv`, then `mcd-2.csv`, until an unused number is found.
Example:
```Bash
DataSet
├───ID0
│   ├───mcd-0.csv
│   ├───mcd-1.csv
│   └───mcd-2.csv
├───ID1
│   ├───mcd-0.csv
│   ├───mcd-1.csv
│   └───mcd-2.csv
└───ID2
    ├───mcd-0.csv
    ├───mcd-1.csv
    └───mcd-2.csv        
```
