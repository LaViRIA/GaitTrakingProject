"""
Author: Moises Badajoz Martinez <m.badajozmartinez@ugto.mx>

University of Guanajuato (2025)
"""

import customtkinter as ctk
from tkinter import filedialog, font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
from pathlib import Path
from serial.tools import list_ports
from api import McdAPI, unpack_bat, unpack_data
import struct
import queue
import threading
from pprint import pprint
import csv
from pathlib import Path


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Capture")
        self.geometry("1280x720")
        self.font = ctk.CTkFont(family="Roboto", size=17)

        # layout 2 filas 3 columnas
        self.grid_rowconfigure(0, weight=3)
        self.grid_columnconfigure([0, 1], weight=1)
        self.grid_columnconfigure(2, weight=10)

        self.frame_left = ctk.CTkFrame(self, corner_radius=5)
        self.frame_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_center = ctk.CTkFrame(self, corner_radius=5)
        self.frame_center.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.frame_right = ctk.CTkScrollableFrame(self, corner_radius=5)
        self.frame_right.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        # Menu (left)
        # select data set dir
        self.dataset_path = "DataSet"
        self.btn_select_dir = ctk.CTkButton(
            self.frame_left, text="Select dir", command=self.select_dir, font=self.font
        )
        self.label_id = ctk.CTkLabel(self.frame_left, text="ID", font=self.font)
        self.input_id = ctk.CTkEntry(
            self.frame_left, text_color="#FFFFFF", placeholder_text="ID", font=self.font
        )
        self.input_id.insert(0, string="-1")
        # measures
        # height
        self.var_height = ctk.DoubleVar()
        self.label_height = ctk.CTkLabel(
            self.frame_left, text="Height (m)", font=self.font
        )
        self.input_height = ctk.CTkEntry(
            self.frame_left,
            text_color="#FFFFFF",
            placeholder_text="Height (m)",
            textvariable=self.var_height,
            justify="center",
            font=self.font,
        )
        # age
        self.var_age = ctk.DoubleVar()
        self.label_age = ctk.CTkLabel(self.frame_left, text="Age", font=self.font)
        self.input_age = ctk.CTkEntry(
            self.frame_left,
            text_color="#FFFFFF",
            placeholder_text="Age",
            textvariable=self.var_age,
            justify="center",
            font=self.font,
        )
        # femur
        self.var_femur = ctk.DoubleVar()
        self.label_femur = ctk.CTkLabel(
            self.frame_left, text="Femur length (m)", font=self.font
        )
        self.input_femur = ctk.CTkEntry(
            self.frame_left,
            text_color="#FFFFFF",
            placeholder_text="Femur length (m)",
            textvariable=self.var_femur,
            justify="center",
            font=self.font,
        )
        # tibia
        self.var_tibia = ctk.DoubleVar()
        self.label_tibia = ctk.CTkLabel(
            self.frame_left, text="Tibia length (m)", font=self.font
        )
        self.input_tibia = ctk.CTkEntry(
            self.frame_left,
            text_color="#FFFFFF",
            placeholder_text="Tibia length (m)",
            textvariable=self.var_tibia,
            justify="center",
            font=self.font,
        )
        # hip
        self.var_hip = ctk.DoubleVar()
        self.label_hip = ctk.CTkLabel(
            self.frame_left, text="Hip width (m)", font=self.font
        )
        self.input_hip = ctk.CTkEntry(
            self.frame_left,
            text_color="#FFFFFF",
            placeholder_text="Hip width (m)",
            textvariable=self.var_hip,
            justify="center",
            font=self.font,
        )

        # packing left
        self.btn_select_dir.pack(padx=5, pady=(10, 5))
        self.label_id.pack(padx=5, pady=(0, 0))
        self.input_id.pack(padx=5, pady=(0, 5))
        self.label_height.pack(padx=5, pady=(0, 0))
        self.input_height.pack(padx=5, pady=(0, 5))
        self.label_age.pack(padx=5, pady=(0, 0))
        self.input_age.pack(padx=5, pady=(0, 5))
        self.label_femur.pack(padx=5, pady=(0, 0))
        self.input_femur.pack(padx=5, pady=(0, 5))
        self.label_tibia.pack(padx=5, pady=(0, 0))
        self.input_tibia.pack(padx=5, pady=(0, 5))
        self.label_hip.pack(padx=5, pady=(0, 0))
        self.input_hip.pack(padx=5, pady=(0, 5))

        # serial port
        self.optionMenu_serial_port = ctk.CTkOptionMenu(
            self.frame_center,
            values=[x.device for x in sorted(list_ports.comports())],
            font=self.font,
        )
        self.btn_update_ports = ctk.CTkButton(
            self.frame_center,
            text="Reload Ports",
            command=lambda: self.optionMenu_serial_port.configure(
                values=[x.device for x in sorted(list_ports.comports())]
            ),
            font=self.font,
        )
        self.mcd = McdAPI("", 250000)
        # open/close port
        self.btn_port = ctk.CTkButton(
            self.frame_center,
            text="Open Port",
            command=self.open_port,
            font=self.font,
        )
        # start/stop read
        self.btn_read = ctk.CTkButton(
            self.frame_center,
            text="Start Read",
            command=self.start_read,
            state="disabled",
            font=self.font,
        )
        # Reset all sensors
        self.btn_reset = ctk.CTkButton(
            self.frame_center,
            text="Reset Sensors",
            command=self.mcd.reset_sensor,
            state="disabled",
            font=self.font,
        )
        # start/stop capture
        self.btn_capture = ctk.CTkButton(
            self.frame_center,
            text="Start Capture",
            command=self.start_capture,
            state="disabled",
            font=self.font,
        )
        # Sync mode
        self.btn_sync = ctk.CTkButton(
            self.frame_center,
            text="Sync Mode",
            command=self.mcd.sync_mode,
            state="disabled",
            font=self.font,
        )

        self.label_sensors_config = ctk.CTkLabel(
            self.frame_center, text="Save Sensors Config", font=self.font
        )
        self.btn_save_config = ctk.CTkButton(
            self.frame_center, text="Save", command=self.save_config, font=self.font
        )
        self.btn_load_config = ctk.CTkButton(
            self.frame_center, text="Load", command=self.load_config, font=self.font
        )

        # packing center
        self.optionMenu_serial_port.pack(padx=5, pady=(10, 5))
        self.btn_update_ports.pack(padx=5, pady=(0, 5))
        self.btn_port.pack(padx=5, pady=(0, 5))
        self.btn_sync.pack(padx=5, pady=(0, 5))
        self.btn_reset.pack(padx=5, pady=(0, 5))
        self.btn_read.pack(padx=5, pady=(0, 5))
        self.btn_capture.pack(padx=5, pady=(0, 5))
        self.label_sensors_config.pack(padx=5, pady=(20, 0))
        self.btn_save_config.pack(padx=5, pady=(0, 5))
        self.btn_load_config.pack(padx=5, pady=(0, 5))

        self.dict_sensor = {}
        self.config = {}

        self.after_update_sensors_id = None
        self.after_graphs_id = None

    def save_config(self):
        file = filedialog.asksaveasfile()
        # if file == None:
        if isinstance(file, type(None)):
            return

        csv_writer = csv.DictWriter(file, fieldnames=["addr", "bone"], delimiter=",")
        csv_writer.writeheader()

        for addr, data in self.dict_sensor.items():
            row = {"addr": addr, "bone": data["var_bone"].get()}
            csv_writer.writerow(row)
        file.close()

    def load_config(self):
        file = filedialog.askopenfile()
        if isinstance(file, type(None)):
            return
        for r in csv.DictReader(file):
            print(r)
            self.config[r["addr"]] = r["bone"]

    def update_sensors(self):
        if not self.queue_battery.qsize():
            self.after_update_sensors_id = self.after(1000, self.update_sensors)
            return
        battery = self.queue_battery.get_nowait()
        for addr, bat in battery["data"].items():
            if addr not in self.dict_sensor.keys():
                self.add_sensor(addr, bat)
            self.dict_sensor[addr]["battery"].configure(text=f"B: {bat}%")
            if self.dict_sensor[addr]["var_bone"].get() == "Select":
                self.dict_sensor[addr]["select_bone"].set(
                    self.config.get(addr, "Select")
                )
        sensors = list(self.dict_sensor.keys())
        for addr in sensors:
            if addr not in battery["data"].keys():
                self.del_sensor(addr)
        self.after_update_sensors_id = self.after(1000, self.update_sensors)

    def update_graphs(self):
        if not self.queue_data.qsize():
            self.after_graphs_id = self.after(16, self.update_graphs)
            return
        data = self.queue_data.get_nowait()
        for addr, sensor in self.dict_sensor.items():
            if addr not in data["data"].keys():
                continue
            for idx, axis in enumerate(["x", "y", "z"]):
                sensor["graph_data"][axis] = np.append(
                    sensor["graph_data"][axis][1:],
                    data["data"][addr]["accel"][idx] * 9.81 / 16_384,
                )
                sensor["ax"][axis].set_data(
                    np.linspace(0, 1, 100), sensor["graph_data"][axis]
                )
            sensor["canvas"].draw_idle()
        self.after_graphs_id = self.after(16, self.update_graphs)

    def select_dir(self):
        self.dataset_path = filedialog.askdirectory()

    def open_port(self):
        port = self.optionMenu_serial_port.get()
        self.mcd.set_port(port)
        if self.mcd.connect():
            self.btn_port.configure(text="Close Port", command=self.close_port)
            self.btn_read.configure(state="normal")
            self.btn_reset.configure(state="normal")
            self.btn_sync.configure(state="normal")
            self.queue_data = queue.Queue()
            self.queue_battery = queue.Queue()
            self.capture_flag = queue.Queue()
            self.process = threading.Thread(
                target=read_capture_process,
                name="process",
                args=(self.mcd, self.capture_flag, self.queue_data, self.queue_battery),
            )
            self.process.start()
            self.after_update_sensors_id = self.after(1000, self.update_sensors)

    def close_port(self):
        self.mcd.read_stop()
        time.sleep(0.1)
        if self.mcd.close():
            self.btn_port.configure(text="Open Port", command=self.open_port)
            self.btn_reset.configure(state="disabled")
            self.btn_sync.configure(state="disabled")
            self.btn_read.configure(
                text="Start Read", command=self.start_read, state="disabled"
            )
            self.btn_capture.configure(
                text="Start Capture", command=self.start_capture, state="disabled"
            )
            self.after_cancel(self.after_update_sensors_id)
            sensors = list(self.dict_sensor.keys())
            for s in sensors:
                self.del_sensor(s)

    def start_read(self):
        self.btn_read.configure(text="Stop Read", command=self.stop_read)
        self.btn_capture.configure(state="normal")
        self.mcd.read_start()
        self.after_graphs_id = self.after(33, self.update_graphs)

    def stop_read(self):
        self.btn_read.configure(text="Start Read", command=self.start_read)
        self.btn_capture.configure(text="Start Capture", state="disabled")
        self.mcd.read_stop()
        self.after_cancel(self.after_graphs_id)

    def start_capture(self):
        self.btn_capture.configure(text="Stop Capture", command=self.stop_capture)
        comment = (
            f";height: {self.var_height.get()};"
            + f"age: {self.var_age.get()};"
            + f"femur_length: {self.var_femur.get()};"
            + f"tibia_length: {self.var_tibia.get()};"
            + f"hip_width: {self.var_hip.get()};"
            + "direction: R\n"
        )
        save_path = self.dataset_path
        save_name = self.input_id.get()

        for addr, data in self.dict_sensor.items():
            self.config[addr] = data["var_bone"].get()

        self.capture_flag.put(
            {
                "capture": True,
                "comment": comment,
                "path": f"{save_path}/{save_name}",
                "config": self.config,
            }
        )

    def stop_capture(self):
        self.btn_capture.configure(text="Start Capture", command=self.start_capture)
        self.capture_flag.put({"capture": False})

    def add_sensor(self, addr, battery):
        if self.dict_sensor.get(addr, False):
            return
        frame = ctk.CTkFrame(self.frame_right, corner_radius=10)
        frame_graph = ctk.CTkFrame(frame)
        fig, ax = plt.subplots(figsize=(1, 1), sharex=True, sharey=True)
        ax.set_xlim((0, 1))
        ax.set_ylim((-20, 20))
        graph_canvas = FigureCanvasTkAgg(fig, master=frame_graph)
        fig.subplots_adjust(left=0.06, right=0.98)
        var_bone = ctk.StringVar(value="Select")
        self.dict_sensor[addr] = {
            "frame": frame,
            "address": ctk.CTkLabel(frame, text=addr),
            "battery": ctk.CTkLabel(frame, text=f"B: {battery}%"),
            "var_bone": var_bone,
            "select_bone": ctk.CTkOptionMenu(
                frame,
                width=100,
                variable=var_bone,
                values=["Select", "R Tibia", "L Tibia", "R Femur", "L Femur"],
            ),
            "btn_identify": ctk.CTkButton(
                frame,
                text="Identify",
                width=100,
                command=lambda: self.mcd.identify_sensor(addr),
            ),
            "btn_reset": ctk.CTkButton(
                frame,
                text="Reset",
                width=100,
                command=lambda: self.mcd.reset_sensor(addr),
            ),
            "btn_calibrate": ctk.CTkButton(
                frame,
                text="Calibrate",
                width=100,
                command=lambda: self.mcd.calibrate_sensor(addr),
            ),
            "frame_graph": frame_graph,
            "fig": fig,
            "ax": {
                "x": ax.plot([], [], "-r")[0],
                "y": ax.plot([], [], "-g")[0],
                "z": ax.plot([], [], "-b")[0],
            },
            "graph_data": {
                "x": np.zeros((100), dtype=float),
                "y": np.zeros((100), dtype=float),
                "z": np.zeros((100), dtype=float),
            },
            "canvas": graph_canvas,
        }

        self.dict_sensor[addr]["frame"].pack(padx=5, pady=5, fill="x", expand=True)
        self.dict_sensor[addr]["frame"].grid_columnconfigure([0, 1], weight=1)
        self.dict_sensor[addr]["frame"].grid_columnconfigure(2, weight=15)
        self.dict_sensor[addr]["frame"].grid_rowconfigure([0, 1, 2], weight=1)
        self.dict_sensor[addr]["address"].grid(
            column=0, row=0, sticky="w", padx=15, pady=5
        )
        self.dict_sensor[addr]["battery"].grid(
            column=0, row=1, sticky="w", padx=15, pady=5
        )
        self.dict_sensor[addr]["select_bone"].grid(
            column=0, row=2, sticky="w", padx=10, pady=2
        )
        self.dict_sensor[addr]["btn_identify"].grid(
            column=1, row=0, sticky="e", padx=10, pady=2
        )
        self.dict_sensor[addr]["btn_reset"].grid(
            column=1, row=1, sticky="e", padx=10, pady=2
        )
        self.dict_sensor[addr]["btn_calibrate"].grid(
            column=1, row=2, sticky="e", padx=10, pady=2
        )
        self.dict_sensor[addr]["frame_graph"].grid(
            column=2, row=0, rowspan=3, sticky="nsew", padx=(10, 0), pady=0
        )
        self.dict_sensor[addr]["canvas"].draw()
        self.dict_sensor[addr]["canvas"].get_tk_widget().pack(
            side="right", expand=True, fill="both"
        )

    def del_sensor(self, addr):
        plt.close(self.dict_sensor[addr]["fig"])
        self.dict_sensor[addr]["frame"].destroy()
        self.dict_sensor.pop(addr)

    def on_closing(self):
        if self.mcd.is_open():
            self.stop_capture()
            time.sleep(0.2)
            self.mcd.read_stop()
            self.mcd.close()
        sensors = list(self.dict_sensor.keys())
        for s in sensors:
            self.del_sensor(s)

        try:
            if not isinstance(self.after_update_sensors_id, type(None)):
                self.after_cancel(self.after_update_sensors_id)
        except Exception as e:
            print(e)
            pass
        try:
            if not isinstance(self.after_update_sensors_id, type(None)):
                self.after_cancel(self.after_graphs_id)
        except Exception as e:
            print(e)
            pass
        time.sleep(0.1)
        self.destroy()


def read_capture_process(
    mcd: McdAPI,
    capture_flag: queue.Queue,
    queue_data: queue.Queue,
    queue_battery: queue.Queue,
):
    buffer: bytearray = b""
    capturing = False
    reference_time = None
    csv_file = None
    csv_writer = None
    config = {}
    fields = [
        "time",
        "quaternion_RF",
        "acceleration_RF",
        "gyro_RF",
        "quaternion_RT",
        "acceleration_RT",
        "gyro_RT",
        "quaternion_LF",
        "acceleration_LF",
        "gyro_LF",
        "quaternion_LT",
        "acceleration_LT",
        "gyro_LT",
    ]
    while True:
        if capture_flag.qsize() > 0:
            capture_msg = capture_flag.get()
            if capturing := capture_msg["capture"]:
                reference_time = None
                i = 0
                _path = Path(f"{capture_msg['path']}/mcd-{i}.csv")
                if not _path.parent.parent.exists():
                    _path.parent.parent.mkdir()
                if not _path.parent.exists():
                    _path.parent.mkdir()
                while _path.exists():
                    i += 1
                    _path = Path(f"{capture_msg['path']}/mcd-{i}.csv")

                csv_file, csv_writer = init_csv_file(
                    path=_path,
                    fields=fields,
                    delimiter=";",
                    comments=capture_msg["comment"],
                )
                config = {v: k for k, v in capture_msg["config"].items()}
            else:
                try:
                    csv_file.close()
                except Exception as e:
                    print(e)
                    pass
                capturing = False
        try:
            if not mcd.is_open():
                break
            if mcd.in_waiting() > 0:
                buffer += mcd.read()

            if (s_idx := buffer.find(b"S")) == -1:
                continue
            buffer = buffer[s_idx:]
            if len(buffer) < 3:
                continue
            if buffer[1] == 1:
                len_data = buffer[2]
                FORMAT_DATA = "<BBQ" + (len_data * "6B4f3h3h")
                EXPECTED_BYTES_DATA = struct.calcsize(FORMAT_DATA)
                while len(buffer) < EXPECTED_BYTES_DATA + 1:
                    buffer += mcd.read()
                data_pack = unpack_data(buffer[1:], FORMAT_DATA)
                buffer = buffer[EXPECTED_BYTES_DATA + 1 :]
                # pprint(data_pack)
                if queue_data.qsize() == 0:
                    queue_data.put_nowait(data_pack)
                if capturing:
                    row = {}
                    if isinstance(reference_time, type(None)):
                        reference_time = data_pack["timestamp"]

                    row["time"] = data_pack["timestamp"] - reference_time
                    for r, c in zip(
                        ["RF", "RT", "LF", "LT"],
                        ["R Femur", "R Tibia", "L Femur", "L Tibia"],
                    ):
                        if not config.get(c, False):
                            continue
                        data = data_pack["data"][config[c]]
                        row[f"quaternion_{r}"] = ",".join(map(str, data["quat"]))
                        row[f"acceleration_{r}"] = ",".join(map(str, data["accel"]))
                        row[f"gyro_{r}"] = ",".join(map(str, data["gyro"]))
                    csv_writer.writerow(row)

            elif buffer[1] == 2:
                len_data = buffer[2]
                FORMAT_BAT = "<BB" + (len_data * "6B1B")
                EXPECTED_BYTES_BAT = struct.calcsize(FORMAT_BAT)
                while len(buffer) < EXPECTED_BYTES_BAT + 1:
                    buffer += mcd.read()
                battery_pack = unpack_bat(buffer[1:], FORMAT_BAT)
                buffer = buffer[EXPECTED_BYTES_BAT + 1 :]
                # print(battery_pack)
                if queue_battery.qsize() == 0:
                    queue_battery.put_nowait(battery_pack)

            else:
                print(buffer[:10])
                buffer = buffer[1:]
        except KeyboardInterrupt:
            mcd.read_stop()
            break
        except Exception as e:
            e.with_traceback(None)
            pass


def init_csv_file(path: str, fields: list[str], delimiter: str, comments: str = None):
    """
    init a csv file

    Args:
        path (str): the path for the cvs file
        fields (list[str]): fields for the csv header
        delimiter (str): csv delimiter caracter

    Returns:
        tuple: Tuple with the file and the csv dict writer
    """
    file = open(path, "w", newline="")
    # if comments != None:
    if not isinstance(comments, type(None)):
        file.write(f"# {comments}")
    csv_writer = csv.DictWriter(file, fieldnames=fields, delimiter=delimiter)
    csv_writer.writeheader()
    return (file, csv_writer)


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
