"""
Author: Moises Badajoz Martinez <m.badajozmartinez@ugto.mx>

University of Guanajuato (2025)
"""

import customtkinter as ctk
from tkinter import filedialog
from ahrs.filters import Madgwick, EKF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import time
from serial.tools import list_ports
from api import McdAPI, unpack_bat, unpack_data
import struct
import queue
import threading
from pprint import pprint
import csv
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, lfilter, detrend, find_peaks


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def conjugado(q):
    """
    Calcula el conjugado de un cuaternión.

    Args:
        q (np.array): Cuaternión de entrada en formato [w, x, y, z].

    Returns:
        np.array: Cuaternión conjugado.
    """
    return q * np.array([1, -1, -1, -1])


def multiplicacion(q1, q2):
    """
    Multiplica dos cuaterniones (q1 * q2).

    Args:
        q1 (np.array): Primer cuaternión [w1, x1, y1, z1].
        q2 (np.array): Segundo cuaternión [w2, x2, y2, z2].

    Returns:
        np.array: Cuaternión resultante.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def rotacion_relativa(q1, q2):
    """
    Calcula la rotación relativa para ir de la orientación q1 a la q2.

    La fórmula es: q_rel = conjugado(q1) * q2

    Args:
        q1 (np.array): Cuaternión de la orientación inicial [w, x, y, z].
        q2 (np.array): Cuaternión de la orientación final [w, x, y, z].

    Returns:
        np.array: Cuaternión que representa la rotación relativa.
    """
    # Asegurarse de que los cuaterniones estén normalizados (opcional pero recomendado)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calcular el conjugado de la orientación inicial
    q1_conj = conjugado(q1)

    # Multiplicar el conjugado de q1 por q2
    q_rel = multiplicacion(q1_conj, q2)

    return q_rel


def rotation_matrix_from_vectors(vec_sensor, vec_world=[0, 0, 1]):
    """Calcula la matriz de rotación para alinear vec_sensor con vec_world."""
    vec_sensor = vec_sensor / np.linalg.norm(vec_sensor)
    vec_world = vec_world / np.linalg.norm(vec_world)

    v = np.cross(vec_sensor, vec_world)
    c = np.dot(vec_sensor, vec_world)
    s = np.linalg.norm(v)

    # Matriz de rotación (Fórmula de Rodrigues)
    if s < 1e-6:  # Vectores paralelos
        return np.eye(3)
    else:
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def apply_quaternion_rotation(quaternion, points):
    """Aplica la rotación a los puntos usando el cuaternión actual."""
    try:
        if np.sum(quaternion) == 0:
            quaternion = np.array([1, 0, 0, 0])
        if quaternion[0] > 1:
            quaternion = quaternion * (1 / np.sqrt(quaternion @ quaternion))
        rot = R.from_quat(
            quaternion, scalar_first=True
        )  # Crear la rotación a partir del cuaternión

        # Aplicar la rotación a los puntos (con transposición para operaciones)
        rotated_points = rot.apply(points.T).T
        return rotated_points
    except Exception as e:
        print(quaternion, e)
        return points


def slerp_quaternion_(q1, q2):
    """Interpolate between two quaternions using Slerp."""
    rots = R.from_quat([q1, q2], scalar_first=True)
    slerp = Slerp([0, 1], rots)
    return slerp(0.5).as_quat(scalar_first=True)


def slerp_quaternion(q1, q2):
    """Interpolate between two quaternions using Slerp."""
    rots = R.concatenate([q1, q2])
    slerp = Slerp([0, 1], rots)
    return slerp(0.5)


def transformation_matrix_(position, quaternion) -> np.ndarray:
    """
    Construye una matriz de transformación 4x4 a partir de una posición 3D y un cuaternión.

    Args:
        position (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa la posición.
        quaternion (list or numpy.ndarray): Lista o arreglo numpy de 4 elementos [x, y, z, w] que representa el cuaternión.

    Returns:
        numpy.ndarray: Matriz de transformación 4x4.
    """

    # 1. Matriz de Rotación
    rotation_matrix = R.from_quat(quaternion, scalar_first=True).as_matrix()

    # 2. Matriz de Traslación
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # 3. Matriz de Transformación Completa
    t_matrix = np.eye(4)
    t_matrix[:3, :3] = rotation_matrix
    t_matrix[:3, 3] = position

    return t_matrix


def transformation_matrix(position, rotation) -> np.ndarray:
    """
    Construye una matriz de transformación 4x4 a partir de una posición 3D y un cuaternión.

    Args:
        position (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa la posición.
        rotation: --
    Returns:
        numpy.ndarray: Matriz de transformación 4x4.
    """

    # 1. Matriz de Rotación
    rotation_matrix = rotation.as_matrix()

    # 2. Matriz de Traslación
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = position

    # 3. Matriz de Transformación Completa
    t_matrix = np.eye(4)
    t_matrix[:3, :3] = rotation_matrix
    t_matrix[:3, 3] = position

    return t_matrix


def apply_transformation(t_matrix, point):
    """
    Aplica una matriz de transformación a un punto.

    Args:
        t_matrix (numpy.ndarray): Matriz de transformación 4x4.
        point (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa el punto.

    Returns:
        numpy.ndarray: Arreglo numpy de 3 elementos [x', y', z'] que representa la nueva posición del punto.
    """

    # 1. Representación del punto en coordenadas homogéneas
    homogeneous_point = np.append(point, 1)  # add 1 to the end of the array
    # change the array to be a 4x1 array.
    homogeneous_point = homogeneous_point.reshape((4, 1))

    # 2. Multiplicación de matrices
    transformed_point = np.dot(t_matrix, homogeneous_point)

    # 3. Obtención de la nueva posición
    # take the first 3 elements, and flatten the array.
    new_point = transformed_point[:3].flatten()

    return new_point


def get_pose(quats: dict, points: dict):
    """
    Obtiene la pose de los puntos del esqueleto a partir de los cuaterniones.

    Args:
        quats (dict): Diccionario con los cuaterniones de cada hueso.
        points (dict): Diccionario con los puntos del esqueleto.

    Returns:
        dict: Diccionario con los puntos del esqueleto rotados.
    """
    pose = {}
    # matriz de transformacion para la cadera
    T_h = transformation_matrix([0, 0, 0], quats["hip"])
    # puntos de cadera
    pose["hip_r"] = apply_transformation(T_h, points["hip_r"])
    pose["hip_l"] = apply_transformation(T_h, points["hip_l"])
    # Matriz de tranformacion de las rodillas (respecto a la cadera)
    # formadas por la traslacion (posicion) de la cadera y la rotacion del femur
    T_kr = transformation_matrix(pose["hip_r"], quats["rf"])
    T_kl = transformation_matrix(pose["hip_l"], quats["lf"])
    # puntos de rodilla
    pose["knee_r"] = apply_transformation(T_kr, points["knee_r"])
    pose["knee_l"] = apply_transformation(T_kl, points["knee_l"])
    # Matriz de transformacion de los tobillos (respecto a las rodillas)
    # formadas por la traslacion (posicion) de la rodilla y la rotacion de la tibia
    T_ar = transformation_matrix(pose["knee_r"], quats["rt"])
    T_al = transformation_matrix(pose["knee_l"], quats["lt"])
    pose["ankle_r"] = apply_transformation(T_ar, points["ankle_r"])
    pose["ankle_l"] = apply_transformation(T_al, points["ankle_l"])

    return pose


def rot_3dcross(quat, points, traslation=np.array([0, 0, 0])):
    rot_matrix = transformation_matrix(traslation, quat)
    rot_points = np.empty((0, 3))
    for k in points:
        rot_points = np.append(
            rot_points, [apply_transformation(rot_matrix, k)], axis=0
        )
    return rot_points


class RealTimeLowPass:
    def __init__(self, cutoff_freq, fs, order=4):
        """
        Inicializa el filtro pasabajas en tiempo real.

        Args:
            cutoff_freq (float): Frecuencia de corte en Hz.
            fs (float): Frecuencia de muestreo en Hz.
            order (int): Orden del filtro.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff_freq / nyq
        self.b, self.a = butter(order, normal_cutoff, btype="low", analog=False)
        self.zi = np.zeros(max(len(self.b), len(self.a)) - 1)  # Estado inicial

    def process_sample(self, sample):
        """
        Procesa una única muestra de datos.

        Args:
            sample (float): La muestra de datos de entrada.

        Returns:
            float: La muestra filtrada.
        """
        filtered_sample, self.zi = lfilter(self.b, self.a, [sample], zi=self.zi)
        return filtered_sample[0]


# kalman
class EKF_IMU:
    def __init__(self, sample_rate=100.0):
        """
        Inicializa el filtro EKF para un IMU.

        Args:
            sample_rate (float): La frecuencia de muestreo en Hz.
        """
        # Inicializamos el filtro EKF de la biblioteca ahrs.
        # Le pasamos la frecuencia, y podemos ajustar los ruidos.
        # Al no pasarle datos de magnetómetro (mag), operará como un filtro de 6 ejes.
        self.ekf = EKF(
            frequency=sample_rate, fraame="NED", noise_gyro=0.05, noise_acc=0.05
        )

        # Guardamos el cuaternión de estado inicial (orientación neutra)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro_rad, acc_g):
        """
        Actualiza el filtro con una nueva muestra de datos.

        Args:
            gyro_rad (np.array): Datos del giroscopio en radianes/segundo [gx, gy, gz].
            acc_g (np.array): Datos del acelerómetro en g's [ax, ay, az].

        Returns:
            np.array: El cuaternión de orientación estimado [qw, qx, qy, qz].
        """
        # El método update del filtro toma las lecturas y el cuaternión anterior
        self.q = self.ekf.update(self.q, gyro_rad, acc_g)
        return self.q

    def reset(self):
        """Resetea el filtro EKF a su estado inicial."""
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ekf = EKF(
            frequency=self.ekf.frequency, fraame="NED", noise_gyro=0.1, noise_acc=0.1
        )


class SensorFilter:
    def __init__(self, cutoff_freq, fs, order=4):
        self.cutoff_freq = cutoff_freq
        self.fs = fs
        self.order = order
        self.ekf = EKF_IMU(sample_rate=fs)
        # self.rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self.reset_filters()

    def reset_filters(self):
        self.accel_filter = [
            RealTimeLowPass(self.cutoff_freq, self.fs, self.order) for _ in range(3)
        ]
        self.gyro_filter = [
            RealTimeLowPass(self.cutoff_freq, self.fs, self.order) for _ in range(3)
        ]
        self.Q = np.array([1.0, 0.0, 0.0, 0.0])  # Cuaternión inicial
        self.ekf.reset()

    def filter_accel(self, accel):
        """Filtra los datos de aceleración."""
        return np.array(
            [filt.process_sample(a) for filt, a in zip(self.accel_filter, accel)]
        )

    def filter_gyro(self, gyro):
        """Filtra los datos de giroscopio."""
        return np.array(
            [filt.process_sample(g) for filt, g in zip(self.gyro_filter, gyro)]
        )

    def ekf_update(self, accel, gyro):
        """Aplica el filtro EKF."""
        # self.rot = rotation_matrix_from_vectors([1, 0, 0], [0, 0, 1])
        # accel = self.rot @ self.filter_accel(accel)
        # gyro = self.rot @ self.filter_gyro(gyro)
        self.Q = self.ekf.update(gyro, accel)
        return self.Q

    def madgwick_update(self, accel, gyro, dt=0.01):
        """Aplica el filtro Madgwick."""
        madgwick_filter = Madgwick(Dt=dt)
        self.rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        # self.rot = np.eye(3)
        accel = self.rot @ self.filter_accel(accel)
        gyro = self.rot @ self.filter_gyro(gyro)
        self.Q = madgwick_filter.updateIMU(gyr=gyro, acc=accel, q=self.Q)
        return self.Q

    def reorient_calibration(self, accel):
        rest = np.mean(accel, axis=0)
        rest /= np.linalg.norm(rest)
        self.rot = rotation_matrix_from_vectors(rest, [0, 0, 1])


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Demo")
        self.geometry("1280x720")

        # layout 2 filas 3 columnas
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=5)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=20)

        self.frame_left = ctk.CTkFrame(self, corner_radius=5)
        self.frame_left.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.frame_right = ctk.CTkFrame(self, corner_radius=5)
        self.frame_right.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)
        self.frame_left_down = ctk.CTkScrollableFrame(self, corner_radius=5)
        self.frame_left_down.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # serial port
        self.optionMenu_serial_port = ctk.CTkOptionMenu(
            self.frame_left, values=[x.device for x in sorted(list_ports.comports())]
        )
        self.btn_update_ports = ctk.CTkButton(
            self.frame_left,
            text="Reload Ports",
            command=lambda: self.optionMenu_serial_port.configure(
                values=[x.device for x in sorted(list_ports.comports())]
            ),
        )
        self.mcd = McdAPI("", 250000)
        # open/close port
        self.btn_port = ctk.CTkButton(
            self.frame_left, text="Open Port", command=self.open_port
        )
        # start/stop read
        self.btn_read = ctk.CTkButton(
            self.frame_left,
            text="Start Read",
            command=self.start_read,
            state="disabled",
        )
        # Sync mode
        self.btn_sync = ctk.CTkButton(
            self.frame_left,
            text="Sync Mode",
            command=self.mcd.sync_mode,
            state="disabled",
        )
        # reset
        self.btn_reset = ctk.CTkButton(
            self.frame_left, text="Reset All", command=self.reset, state="disabled"
        )
        # reset
        self.btn_calibration = ctk.CTkButton(
            self.frame_left,
            text="Soft Calibration",
            command=lambda: self.queue_msg.put_nowait(0x01),
            state="disabled",
        )

        self.label_sensors_config = ctk.CTkLabel(
            self.frame_left, text="Save Sensors Config"
        )
        self.btn_save_config = ctk.CTkButton(
            self.frame_left, text="Save", command=self.save_config
        )
        self.btn_load_config = ctk.CTkButton(
            self.frame_left, text="Load", command=self.load_config
        )

        # packing center
        self.optionMenu_serial_port.pack(padx=5, pady=(10, 5))
        self.btn_update_ports.pack(padx=5, pady=(0, 5))
        self.btn_port.pack(padx=5, pady=(0, 5))
        self.btn_read.pack(padx=5, pady=(0, 5))
        self.btn_sync.pack(padx=5, pady=(0, 5))
        self.btn_reset.pack(padx=5, pady=(0, 5))
        # self.btn_calibration.pack(padx=5, pady=(0, 5))
        self.label_sensors_config.pack(padx=5, pady=(20, 0))
        self.btn_save_config.pack(padx=5, pady=(0, 5))
        self.btn_load_config.pack(padx=5, pady=(0, 5))

        self.dict_sensor = {}
        self.config = {}

        self.filters = {
            "rf": SensorFilter(7, 1000 / 10),
            "lf": SensorFilter(7, 1000 / 10),
            "rt": SensorFilter(7, 1000 / 10),
            "lt": SensorFilter(7, 1000 / 10),
        }

        self.create_graphs()

    def reset(self):
        self.mcd.reset_sensor()
        self.queue_msg.put_nowait(0x00)

    def create_graphs(self):
        self.fig = plt.figure()
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        ax = self.fig.add_subplot(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-1, 0)
        self.ax_graph = ax
        self.ax = {
            "hip": ax.plot([], [], [], "-o", color="gray", alpha=1, linewidth=5)[0],
            "rf": ax.plot([], [], [], "-o", color="r", alpha=1, linewidth=5)[0],
            "lf": ax.plot([], [], [], "-o", color="b", alpha=1, linewidth=5)[0],
            "rt": ax.plot([], [], [], "-o", color="r", alpha=1, linewidth=5)[0],
            "lt": ax.plot([], [], [], "-o", color="b", alpha=1, linewidth=5)[0],
        }

        self.skeleton_points = {
            "hip_r": np.array([0, 0.26 / 2, 0]),
            "knee_r": np.array([0, 0, -0.45]),
            "ankle_r": np.array([0, 0, -0.54]),
            "hip_l": np.array([0, -0.26 / 2, 0]),
            "knee_l": np.array([0, 0, -0.45]),
            "ankle_l": np.array([0, 0, -0.54]),
        }

        self.last_T_al = np.eye(4)
        self.last_T_ar = np.eye(4)
        self.z_compare = 1
        self.cross3d = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def save_config(self):
        file = filedialog.asksaveasfile()
        if file is None:
            return

        csv_writer = csv.DictWriter(file, fieldnames=["addr", "bone"], delimiter=",")
        csv_writer.writeheader()

        for addr, data in self.dict_sensor.items():
            row = {"addr": addr, "bone": data["var_bone"].get()}
            csv_writer.writerow(row)
        file.close()

    def load_config(self):
        file = filedialog.askopenfile()
        if file is None:
            return
        for r in csv.DictReader(file):
            # print(r)
            self.config[r["addr"]] = r["bone"]
            if self.dict_sensor.get(r["addr"], False):
                self.dict_sensor[r["addr"]]["var_bone"].set(r["bone"])

    def update_sensors(self):
        if not self.queue_battery.qsize():
            self.after_update_sensors_id = self.after(1000, self.update_sensors)
            return
        packet = self.queue_battery.get_nowait()
        for addr, data in packet.items():
            if addr not in self.dict_sensor.keys():
                self.add_sensor(addr, data.battery)
            self.dict_sensor[addr]["battery"].configure(text=f"B: {data.battery}%")
            if self.dict_sensor[addr]["var_bone"].get() == "Select":
                self.dict_sensor[addr]["select_bone"].set(
                    self.config.get(addr, "Select")
                )
        sensors = list(self.dict_sensor.keys())
        for addr in sensors:
            if addr not in packet.keys():
                self.del_sensor(addr)
        self.after_update_sensors_id = self.after(1000, self.update_sensors)

    def update_graphs(self):
        # t0 = time.time()
        # si no hay datos en la cola, esperar 33ms y volver a llamar
        if not self.queue_data.qsize():
            self.after_graphs_id = self.after(1, self.update_graphs)
            return
        for addr, data in self.dict_sensor.items():
            self.config[addr] = data["var_bone"].get()

        # config = {v: k for k, v in self.config.items()}
        config_key_equal = {
            "R Tibia": "rt",
            "L Tibia": "lt",
            "R Femur": "rf",
            "L Femur": "lf",
        }
        packet = self.queue_data.get()
        # pprint(self.config)
        quats = {
            "hip": R.from_quat([1, 0, 0, 0], scalar_first=True),
            "rt": R.from_quat([1, 0, 0, 0], scalar_first=True),
            "lt": R.from_quat([1, 0, 0, 0], scalar_first=True),
            "rf": R.from_quat([1, 0, 0, 0], scalar_first=True),
            "lf": R.from_quat([1, 0, 0, 0], scalar_first=True),
        }
        for addr, sensor_data in self.dict_sensor.items():
            if addr not in self.config.keys() or self.config[addr] == "Select":
                continue
            if not packet.get(addr, False):
                continue
            # accel = packet[addr].accel_data
            # gyro = packet[addr].gyro_data

            quats[config_key_equal[self.config[addr]]] = packet[addr].quat_data

        quats["hip"] = slerp_quaternion(quats["rf"], quats["lf"])
        # print(quats["rf"].as_euler("zxy", degrees=True))
        pose = get_pose(quats=quats, points=self.skeleton_points)

        self.ax["hip"].set_data_3d(
            [pose["hip_r"][0], pose["hip_l"][0]],
            [pose["hip_r"][1], pose["hip_l"][1]],
            [pose["hip_r"][2], pose["hip_l"][2]],
        )
        self.ax["rf"].set_data_3d(
            [pose["hip_r"][0], pose["knee_r"][0]],
            [pose["hip_r"][1], pose["knee_r"][1]],
            [pose["hip_r"][2], pose["knee_r"][2]],
        )
        self.ax["lf"].set_data_3d(
            [pose["knee_l"][0], pose["hip_l"][0]],
            [pose["knee_l"][1], pose["hip_l"][1]],
            [pose["knee_l"][2], pose["hip_l"][2]],
        )
        self.ax["rt"].set_data_3d(
            [pose["knee_r"][0], pose["ankle_r"][0]],
            [pose["knee_r"][1], pose["ankle_r"][1]],
            [pose["knee_r"][2], pose["ankle_r"][2]],
        )
        self.ax["lt"].set_data_3d(
            [pose["knee_l"][0], pose["ankle_l"][0]],
            [pose["knee_l"][1], pose["ankle_l"][1]],
            [pose["knee_l"][2], pose["ankle_l"][2]],
        )

        self.canvas.draw_idle()
        # t = time.time() - t0
        self.after_graphs_id = self.after(33, self.update_graphs)

    def open_port(self):
        port = self.optionMenu_serial_port.get()
        self.mcd.set_port(port)
        if self.mcd.connect():
            self.btn_port.configure(text="Close Port", command=self.close_port)
            self.btn_read.configure(state="normal")
            self.btn_sync.configure(state="normal")
            self.btn_reset.configure(state="normal")
            self.btn_calibration.configure(state="normal")
            self.queue_data = queue.Queue()
            self.queue_battery = queue.Queue()
            self.queue_msg = queue.Queue()
            self.process = threading.Thread(
                target=read_process,
                name="process",
                args=(self.mcd, self.queue_data, self.queue_battery, self.queue_msg),
            )
            self.process.start()
            self.after_update_sensors_id = self.after(1000, self.update_sensors)

    def close_port(self):
        self.mcd.read_stop()
        time.sleep(0.1)
        if self.mcd.close():
            self.btn_port.configure(text="Open Port", command=self.open_port)
            self.btn_sync.configure(state="disabled")
            self.btn_reset.configure(state="disabled")
            self.btn_calibration.configure(state="disabled")
            self.btn_read.configure(
                text="Start Read", command=self.start_read, state="disabled"
            )
            self.after_cancel(self.after_update_sensors_id)
            sensors = list(self.dict_sensor.keys())
            for s in sensors:
                self.del_sensor(s)

    def start_read(self):
        self.btn_read.configure(text="Stop Read", command=self.stop_read)
        self.mcd.read_start()
        self.after_graphs_id = self.after(33, self.update_graphs)

    def stop_read(self):
        self.btn_read.configure(text="Start Read", command=self.start_read)
        self.mcd.read_stop()
        self.after_cancel(self.after_graphs_id)

    def add_sensor(self, addr, battery):
        if self.dict_sensor.get(addr, False):
            return
        frame = ctk.CTkFrame(self.frame_left_down, corner_radius=10)
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
            "accel_data": np.zeros(3),
            "gyro_data": np.zeros(3),
            "quat_data": [],
        }

        self.dict_sensor[addr]["frame"].pack(padx=5, pady=5, fill="x")
        self.dict_sensor[addr]["frame"].grid_columnconfigure([0, 1], weight=1)
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
        # self.dict_sensor[addr]['canvas'].draw()
        # self.dict_sensor[addr]['canvas'].get_tk_widget().pack(
        #     side='right', expand=True, fill='both')

    def del_sensor(self, addr):
        self.dict_sensor[addr]["frame"].destroy()
        self.dict_sensor.pop(addr)

    def on_closing(self):
        plt.close(self.fig)
        if self.mcd.is_open():
            time.sleep(0.2)
            self.mcd.read_stop()
            self.mcd.close()
        sensors = list(self.dict_sensor.keys())
        for s in sensors:
            self.del_sensor(s)
        time.sleep(0.1)
        self.destroy()


class SensorData:
    def __init__(self, addr: str):
        self.addr = addr
        self.accel_data = np.zeros(3)
        self.accel_data_array = np.zeros((100, 3))
        self.gyro_data = np.zeros(3)
        self.quat_data = R.from_quat([1.0, 0.0, 0.0, 0.0], scalar_first=True)
        self.battery = 0
        self.filters = SensorFilter(cutoff_freq=7, fs=100)

    def update_data(self, accel, gyro):
        self.accel_data = accel
        self.gyro_data = gyro
        self.accel_data_array = np.append(self.accel_data_array[1:, :], [accel], axis=0)
        self.quat_data = R.from_quat(
            self.filters.madgwick_update(accel, gyro, dt=0.01), scalar_first=True
        )

    def update_data2(self, accel, gyro):
        self.accel_data = accel
        self.gyro_data = gyro
        self.accel_data_array = np.append(self.accel_data_array[1:, :], [accel], axis=0)
        self.quat_data = R.from_quat(
            self.filters.ekf_update(accel, gyro), scalar_first=True
        )

    def software_calibration(self):
        self.filters.reorient_calibration(self.accel_data_array)

    def reset_filter(self):
        self.filters.reset_filters()
        self.quat_data = R.from_quat([1.0, 0.0, 0.0, 0.0], scalar_first=True)


def read_process(
    mcd: McdAPI,
    queue_data: queue.Queue,
    queue_battery: queue.Queue,
    queue_msg: queue.Queue = None,
):
    buffer: bytearray = b""
    sensors = {}
    # t = time.time()
    while True:
        # if (t1 := (time.time()-t)*1000) > 1:
        #     print(t1)
        # t = time.time()
        try:
            if not mcd.is_open():
                break
            if mcd.in_waiting() > 0:
                buffer += mcd.read()
            if queue_msg and queue_msg.qsize() > 0:
                msg = queue_msg.get_nowait()
                if msg == 0x00:  # reset filters
                    for s in sensors.values():
                        s.reset_filter()
                if msg == 0x01:  # software recalibration
                    for s in sensors.values():
                        s.software_calibration()

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
                # print(data_pack['data'])
                # t__ = time.time()
                for addr, values in data_pack["data"].items():
                    if addr not in sensors.keys():
                        sensors[addr] = SensorData(addr)
                    sensors[addr].update_data(
                        (np.array(values["accel"]) / 16_384) * 9.81,
                        (np.array(values["gyro"]) / 16.4) * (np.pi / 180),
                    )
                    # accel = (np.array(values["accel"]) / 16_384) * 9.81
                    # print(f'{accel[0]:+06.2f}, {accel[1]:+06.2f}, {accel[2]:+06.2f}', end='\r')
                    # pprint(values['accel'])
                    # print(sensors[addr].quat_data)
                    # print(values['quat'])
                    # print()

                    # print(sensors[addr].quat_data)
                for addr in list(sensors.keys()):
                    if addr not in data_pack["data"].keys():
                        sensors.pop(addr)
                # print((time.time()-t__)*1000)
                if queue_data.qsize() == 0:
                    queue_data.put_nowait(sensors)

            elif buffer[1] == 2:
                len_data = buffer[2]
                FORMAT_BAT = "<BB" + (len_data * "6B1B")
                EXPECTED_BYTES_BAT = struct.calcsize(FORMAT_BAT)
                while len(buffer) < EXPECTED_BYTES_BAT + 1:
                    buffer += mcd.read()
                battery_pack = unpack_bat(buffer[1:], FORMAT_BAT)
                buffer = buffer[EXPECTED_BYTES_BAT + 1 :]
                for addr, bat in battery_pack["data"].items():
                    if addr not in sensors.keys():
                        sensors[addr] = SensorData(addr)
                    if addr in sensors.keys():
                        sensors[addr].battery = bat
                for addr in list(sensors.keys()):
                    if addr not in battery_pack["data"].keys():
                        sensors.pop(addr)
                if queue_battery.qsize() == 0:
                    queue_battery.put_nowait(sensors)

            else:
                print(buffer[:10])
                buffer = buffer[1:]
        except KeyboardInterrupt:
            mcd.read_stop()
            break
        except Exception as e:
            # print(e)
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
    if comments is not None:
        file.write(f"# {comments}")
    csv_writer = csv.DictWriter(file, fieldnames=fields, delimiter=delimiter)
    csv_writer.writeheader()
    return (file, csv_writer)


if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
