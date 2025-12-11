from ahrs.filters import Madgwick, EKF
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation as R
import numpy as np
from pprint import pprint
import csv
import glob
import sys
from scipy.signal import butter, filtfilt, lfilter, detrend, find_peaks
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.linear_model import LinearRegression
import time
import os

cache_dir = matplotlib.get_cachedir()
font_cache_dir = os.path.join(cache_dir, "fontlist-v310.json")
if os.path.exists(font_cache_dir):
    os.remove(font_cache_dir)
    print("Font cache removed.")
else:
    print("Font cache not found.")
figsize = (7, 6)
title = ""

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 12
if sys.platform == "linux":
    split_char = "/"
else:
    split_char = "\\"


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

    Args:figsize = (7, 6)

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


def low_pass_filter(data, cutoff_freq, fs):
    """
    Applies a low-pass filter to the given data.

    Args:
        time (array-like): Array of timestamps.
        data (array-like): Array of values corresponding to the timestamps.
        cutoff_freq (float): Cutoff frequency of the low-pass filter in Hz.
        fs (float): Sampling frequency.

    Returns:
        numpy.ndarray: Filtered data.
    """
    # Calculate the Nyquist frequency
    nyq = 0.5 * fs
    # Normalize the cutoff frequency
    normal_cutoff = cutoff_freq / nyq
    # Design the Butterworth filter
    order = 4  # Filter order (can be adjusted)
    b, a = butter(order, normal_cutoff, btype="lowpass", analog=False)
    # Apply the filter using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica un filtro pasa-bandas Butterworth a una señal.

    Parámetros:
    data (array_like): La señal de entrada a filtrar.
    lowcut (float): La frecuencia de corte inferior del filtro (en Hz).
    highcut (float): La frecuencia de corte superior del filtro (en Hz).
    fs (float): La frecuencia de muestreo de la señal (en Hz).
    order (int, opcional): El orden del filtro Butterworth. Por defecto es 5.

    Retorna:
    array_like: La señal filtrada.
    """
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def csv_comment(comment):
    """
    Parse a comment line from the CSV file and return a dictionary of measures.
    The comment line is expected to start with a '#' character, followed by a space,
    """
    line = comment.replace("# ", "")
    comment = line.strip()
    measure = comment.split(";")[1:]
    measures = {
        m.split(":")[0]: n if (n := m.split(":")[1].strip()).isalpha() else float(n)
        for m in measure
    }
    return measures


def get_dataset() -> list[tuple[str, str]]:
    """
    Get the dataset folder and return a list of tuples containing the paths to the mcd and arUco files.
    """

    dataset_path = filedialog.askdirectory(
        title="Select the dataset folder",
        initialdir="/media/moybadajoz/Nuevo vol/",
    )
    print(dataset_path)
    if len(dataset_path) == 0 or dataset_path == "":
        print("No folder selected")
        return []
    subjects = glob.glob("*", root_dir=dataset_path)
    files = []
    for s in subjects:
        for i in range(4):
            mcd_file = f"{dataset_path}{split_char}{s}{split_char}mcd-{i}.csv"
            aruco_file = f"{dataset_path}{split_char}{s}{split_char}arUcos-{i}.csv"
            if not Path(mcd_file).exists() or not Path(aruco_file).exists():
                continue
            files.append((mcd_file, aruco_file))
    return files


def read_mcd_file(path: str) -> dict[str, np.ndarray]:
    """
    Read the mcd file and return a dictionary with the data.
    """
    # abre el archivo y lo desglosa en diccionarios
    with open(path, "r") as file:
        measures = csv_comment(file.readline())
        mcd_file = [row for row in csv.DictReader(file, delimiter=";")]
    # mcd_file = mcd_file[100:]
    # convierte los datos en un diccionario de numpy arrays tipo float
    mcd_data = {
        key: np.array(
            [np.array(data[key].split(","), dtype=float) for data in mcd_file]
        )
        for key in mcd_file[0].keys()
    }
    # 'aplana' el arrat de tiempo para que sea de una unica dimension
    mcd_data["time"] = mcd_data["time"].flatten()
    # agrega el diccionario de medidas
    mcd_data["measures"] = measures
    # obtiene la frecuencia de muestreo
    fs = 1000 / np.mean(np.diff(mcd_data["time"]))
    # fs=100
    # convierte los datos a unidades (rad/s y m/s^2) y aplica un filtro pasabajas
    for i in ["RF", "RT", "LF", "LT"]:
        mcd_data[f"gyro_{i}"] = np.array(
            [
                low_pass_filter(
                    ((mcd_data[f"gyro_{i}"] / 16.4) * (np.pi / 180)).T[n], 7, fs
                )
                for n in range(3)
            ]
            # (mcd_data[f"gyro_{i}"] / 16.4)
            # * (np.pi / 180)
        ).T
        # mcd_data[f"gyro_{i}"] = (mcd_data[f"gyro_{i}"] / 16.4) * (np.pi / 180)
        # mcd_data[f"acceleration_{i}"] = (mcd_data[f"acceleration_{i}"] / 16_384) * 9.81
        mcd_data[f"acceleration_{i}"] = np.array(
            [
                low_pass_filter(
                    ((mcd_data[f"acceleration_{i}"] / 16_384) * 9.81).T[n], 7, fs
                )
                for n in range(3)
            ]
            # (mcd_data[f"acceleration_{i}"] / 16_384)
            # * 9.81
        ).T
        # mcd_data[f'gyro_{i}'] = np.array([
        #     bandpass_filter(
        #         ((mcd_data[f'gyro_{i}']/16.4)*(np.pi/180)).T[n], 0.01, 7, fs, order=4)
        #     for n in range(3)]).T
        # mcd_data[f'acceleration_{i}'] = np.array([
        #     bandpass_filter(
        #         ((mcd_data[f'acceleration_{i}']/16_384)*9.81).T[n], 0.01, 7, fs, order=4)
        #     for n in range(3)]).T
    # print(f'Frecuencia de muestreo: {fs} Hz')
    # analizar_y_graficar_fft(mcd_data['gyro_RF'], fs)

    return mcd_data


def read_aruco_file(path: str) -> dict[str, np.ndarray]:
    """
    Read the arUco file and return a dictionary with the data.
    """
    # abre el archivo y lo desglosa en diccionarios
    with open(path, "r") as file:
        aruco_file = [row for row in csv.DictReader(file, delimiter=";")]
    # extrae las llaves
    aruco_keys = aruco_file[0].keys()
    # agrupa los datos de cada llave en una lista
    aruco_data_no_processed = {
        key: [data[key] for data in aruco_file] for key in aruco_keys
    }
    # extrae todas las marcas de tiempo en un array de enteros
    aruco_time = np.array([row["time"] for row in aruco_file], dtype=int)
    # liga cada dato con su respectiva marca de tiempo
    aruco_data = {
        key: {
            time: np.array(data.split(","), dtype=float)
            for time, data in zip(aruco_time, aruco_data_no_processed[key])
            if data != "" and time >= 0
        }
        for key in aruco_keys
        if key != "time"
    }
    # divide las marcas de tiempo y los datos en dos arrays, siguen ligadas por el indice
    for key, values in aruco_data.items():
        aruco_data[key] = {
            "time": np.array(list(values.keys()), dtype=int),
            "data": np.array([values[time] for time in values.keys()]),
        }

    return aruco_data


def adjust_aruco_data(
    aruco_data: dict[str, np.ndarray], direction
) -> dict[str, np.ndarray]:
    x_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 0])
    )
    y_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 1])
    )
    z_init = np.mean(
        np.array(aruco_data[f"{direction}_ankle_position"]["data"][:10, 2])
    )
    init = np.array([x_init, y_init, z_init])

    aruco_data["hip_position"]["data"] -= init
    aruco_data["R_knee_position"]["data"] -= init
    aruco_data["L_knee_position"]["data"] -= init
    aruco_data["R_ankle_position"]["data"] -= init
    aruco_data["L_ankle_position"]["data"] -= init

    return aruco_data


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
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix


def apply_transformation(transformation_matrix, point):
    """
    Aplica una matriz de transformación a un punto.

    Args:
        transformation_matrix (numpy.ndarray): Matriz de transformación 4x4.
        point (list or numpy.ndarray): Lista o arreglo numpy de 3 elementos [x, y, z] que representa el punto.

    Returns:
        numpy.ndarray: Arreglo numpy de 3 elementos [x', y', z'] que representa la nueva posición del punto.
    """

    # 1. Representación del punto en coordenadas homogéneas
    homogeneous_point = np.append(point, 1)  # add 1 to the end of the array
    # change the array to be a 4x1 array.
    homogeneous_point = homogeneous_point.reshape((4, 1))

    # 2. Multiplicación de matrices
    transformed_point = np.dot(transformation_matrix, homogeneous_point)

    # 3. Obtención de la nueva posición
    # take the first 3 elements, and flatten the array.
    new_point = transformed_point[:3].flatten()

    return new_point


def get_pose(quats, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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


def get_all_poses(quats, points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pose = {
        "hip_r": np.empty((0, 3)),
        "hip_l": np.empty((0, 3)),
        "knee_r": np.empty((0, 3)),
        "knee_l": np.empty((0, 3)),
        "ankle_r": np.empty((0, 3)),
        "ankle_l": np.empty((0, 3)),
    }
    for i in range(len(quats["rf"])):
        current_quats = {
            "rf": quats["rf"][i],
            "lf": quats["lf"][i],
            "rt": quats["rt"][i],
            "lt": quats["lt"][i],
            "hip": quats["hip"][i],
        }
        current_pose = get_pose(current_quats, points)
        pose = {
            "hip_r": np.append(pose["hip_r"], [current_pose["hip_r"]], axis=0),
            "hip_l": np.append(pose["hip_l"], [current_pose["hip_l"]], axis=0),
            "knee_r": np.append(pose["knee_r"], [current_pose["knee_r"]], axis=0),
            "knee_l": np.append(pose["knee_l"], [current_pose["knee_l"]], axis=0),
            "ankle_r": np.append(pose["ankle_r"], [current_pose["ankle_r"]], axis=0),
            "ankle_l": np.append(pose["ankle_l"], [current_pose["ankle_l"]], axis=0),
        }
    return pose


def angulo_gravedad(accel: np.ndarray) -> tuple[float, float]:
    """
    Calcula los ángulos de Roll y Pitch a partir del vector de aceleración
    asumiendo que el sensor está en estado estático.

    Args:
        accel (np.ndarray): Un arreglo de NumPy con las 3 componentes de la
                           aceleración [ax, ay, az].

    Returns:
        tuple[float, float]: Una tupla conteniendo (roll, pitch) en grados.
    """
    # Es una buena práctica normalizar el vector de aceleración primero.
    # Esto hace que el cálculo sea independiente de si la unidad es g o m/s^2.
    if np.linalg.norm(accel) == 0:
        return (0.0, 0.0)  # Evitar división por cero si el vector es nulo
    accel_norm = accel / np.linalg.norm(accel)

    ax, ay, az = accel_norm[0], accel_norm[1], accel_norm[2]

    # Calcular Roll (phi) en radianes
    roll_rad = np.arctan2(ay, az)

    # Calcular Pitch (theta) en radianes
    # La fórmula robusta usa la raíz cuadrada de ay^2 + az^2
    pitch_rad = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    # Convertir radianes a grados
    roll_deg = np.rad2deg(roll_rad)
    pitch_deg = np.rad2deg(pitch_rad)

    return (roll_deg, pitch_deg)


def reorient_data(accel: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # obtiene la media de posicion en reposo
    # acc = np.append(accel[:75], accel[-75:], axis=0)
    # rest = np.mean(acc, axis=0)
    rest = np.mean(accel[:200], axis=0)
    rest /= np.linalg.norm(rest)
    # print(rest)
    # obtiene la matrix de rotacion para alinear el vector de gravedad con el eje z
    accel_R = rotation_matrix_from_vectors(rest)
    # accel_R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # print(rotation_matrix_from_vectors([1, 0, 0]))
    # print(accel_R)
    # reorienta los datos de aceleracion y giroscopio
    reoriented_accel = (accel_R @ accel.T).T
    reoriented_gyro = (accel_R @ gyro.T).T

    # ax = plt.subplot(111, projection='3d')
    # ax.plot(accel[:, 0], accel[:, 1], accel[:, 2], 'o', label='XYZ')
    # ax.grid(1)
    # ax.set_xlabel('X (m/s^2)')
    # ax.set_ylabel('Y (m/s^2)')
    # ax.set_zlabel('Z (m/s^2)')
    # # ax.legend(1)
    # plt.show()

    return reoriented_accel, reoriented_gyro


def get_orientation(mcd_data) -> dict[str, np.ndarray]:
    """
    Get the orientation of the sensors using the Madgwick filter.
    """
    reoriented_data = {
        "rf": reorient_data(mcd_data["acceleration_RF"], mcd_data["gyro_RF"]),
        "lf": reorient_data(mcd_data["acceleration_LF"], mcd_data["gyro_LF"]),
        "rt": reorient_data(mcd_data["acceleration_RT"], mcd_data["gyro_RT"]),
        "lt": reorient_data(mcd_data["acceleration_LT"], mcd_data["gyro_LT"]),
    }
    # # print(mcd_data.keys())
    # reoriented_data = {
    #     "rf": (mcd_data["acceleration_RF"], mcd_data["gyro_RF"]),
    #     "lf": (mcd_data["acceleration_LF"], mcd_data["gyro_LF"]),
    #     "rt": (mcd_data["acceleration_RT"], mcd_data["gyro_RT"]),
    #     "lt": (mcd_data["acceleration_LT"], mcd_data["gyro_LT"]),
    # }
    reoriented_accel = {
        "rf": reoriented_data["rf"][0],
        "lf": reoriented_data["lf"][0],
        "rt": reoriented_data["rt"][0],
        "lt": reoriented_data["lt"][0],
    }
    reoriented_gyro = {
        "rf": reoriented_data["rf"][1],
        "lf": reoriented_data["lf"][1],
        "rt": reoriented_data["rt"][1],
        "lt": reoriented_data["lt"][1],
    }
    # creacion de los filtros complementarios
    s_p = 1000 / np.mean(np.diff(mcd_data["time"]))
    # print(s_p)
    # s_p = 100
    q = [1, 0, 0, 0]
    gain = 0.033
    quats = {
        "rf": R.from_quat(
            Madgwick(
                reoriented_gyro["rf"], reoriented_accel["rf"], frequency=s_p, gain=gain
            ).Q,
            scalar_first=True,
        ),
        "lf": R.from_quat(
            Madgwick(
                reoriented_gyro["lf"], reoriented_accel["lf"], frequency=s_p, gain=gain
            ).Q,
            scalar_first=True,
        ),
        "rt": R.from_quat(
            Madgwick(
                reoriented_gyro["rt"], reoriented_accel["rt"], frequency=s_p, gain=gain
            ).Q,
            scalar_first=True,
        ),
        "lt": R.from_quat(
            Madgwick(
                reoriented_gyro["lt"], reoriented_accel["lt"], frequency=s_p, gain=gain
            ).Q,
            scalar_first=True,
        ),
        "hip": np.empty(0),
    }

    """quats = {
        "rf": R.from_quat(
            EKF(
                gyr=reoriented_gyro["rf"],
                acc=reoriented_accel["rf"],
                frequency=s_p,
                frame="NED",
            ).Q,
            scalar_first=True,
        ),
        "lf": R.from_quat(
            EKF(
                gyr=reoriented_gyro["lf"],
                acc=reoriented_accel["lf"],
                frequency=s_p,
                frame="NED",
            ).Q,
            scalar_first=True,
        ),
        "rt": R.from_quat(
            EKF(
                gyr=reoriented_gyro["rt"],
                acc=reoriented_accel["rt"],
                frequency=s_p,
                frame="NED",
            ).Q,
            scalar_first=True,
        ),
        "lt": R.from_quat(
            EKF(
                gyr=reoriented_gyro["lt"],
                acc=reoriented_accel["lt"],
                frequency=s_p,
                frame="NED",
            ).Q,
            scalar_first=True,
        ),
        "hip": np.empty(0),
    }"""

    for key in quats.keys():
        if key == "hip":
            continue
        rot = quats[key][0].inv()
        for i in range(len(quats[key])):
            quats[key][i] = rot * quats[key][i]
    # angles = quats["lf"].as_euler("zyx", degrees=True)
    # angles = quats["lf"].as_rotvec(degrees=True)
    # plt.plot(angles[:, 0], "r")
    # plt.plot(angles[:, 1], "g")
    # plt.plot(angles[:, 2], "b")
    # plt.show()
    for i in range(len(mcd_data["time"])):
        slerp = Slerp([0, 1], R.concatenate([quats["rf"][i], quats["lf"][i]]))
        # print(slerp(0.5))
        quats["hip"] = np.append(quats["hip"], [slerp(0.5)], axis=0)

    rot_r = np.empty((0, 3))
    rot_l = np.empty((0, 3))
    # rot_rk = np.empty((0, 3))
    # rot_lk = np.empty((0, 3))
    for i in range(len(quats["hip"])):
        rot_r = np.append(
            rot_r,
            [(quats["hip"][i].inv() * quats["rf"][i]).as_euler("xyz", degrees=True)],
            axis=0,
        )
        # print(R.from_quat(quats['hip'][i], scalar_first=True).inv().as_quat(scalar_first=True), quats['hip'][i])
        rot_l = np.append(
            rot_l,
            [(quats["hip"][i].inv() * quats["lf"][i]).as_euler("zxy", degrees=True)],
            axis=0,
        )
    # print(f"rot q rf: {rot_r[100:-100].mean(axis=0)}")
    # for a in rot_r:
    #     print(a)
    # plt.plot(rot_r)
    # plt.show()
    fix_rot_r = R.from_euler("z", -np.mean(rot_r[:, 0]), degrees=True).as_quat(
        scalar_first=True
    )
    fix_rot_l = R.from_euler("z", -np.mean(rot_l[:, 0]), degrees=True).as_quat(
        scalar_first=True
    )
    # for i in range(len(quats['rf'])):
    #     quats['rf'][i] = multiplicacion(fix_rot_r, quats['rf'][i])
    #     quats['rt'][i] = multiplicacion(fix_rot_r, quats['rt'][i])
    #     # quats['rt'][i] = multiplicacion(fix_rot_rk, quats['rt'][i])
    #     quats['lf'][i] = multiplicacion(fix_rot_l, quats['lf'][i])
    #     quats['lt'][i] = multiplicacion(fix_rot_l, quats['lt'][i])
    # quats['lt'][i] = multiplicacion(fix_rot_lk, quats['lt'][i])

    # p_rf = np.array([R.from_quat(r, scalar_first=True).as_matrix()
    #                  @ [0, 0, -1] for r in quats['rf']])
    # ax = plt.subplot()
    # ax.plot(p_rf[200:-200, 0], p_rf[200:-200, 1], '.')
    # ax.set_xlim(-.6, .6)
    # ax.set_ylim(-.6, .6)
    # plt.plot(quats['rf'].as_euler('ZXY'))

    # ph = np.array([r.apply([1, 0, 0]) for r in quats["hip"]])
    # mh = ph[:, 1] / ph[:, 0]
    # for b in ["rf", "lf", "rt", "lt"]:
    #     p = np.array([r.apply([0, 0, -1]) for r in quats[b]])
    #     x = p[200:-200, 0].reshape(-1, 1)
    #     y = p[200:-200, 1]
    #     model = LinearRegression()
    #     model.fit(x, y)
    #     m1 = model.coef_[0]
    #     angle = np.arctan(m1) + np.arctan(mh)
    #     # angle = np.arctan(m1)
    #     # print('pendiente: ', model.coef_[0])
    #     # print('angulo: ', np.arctan(model.coef_[0]))
    #     # rot = R.from_euler('z', [-np.arctan(model.coef_[0])])
    #     rot = R.from_euler("z", -angle)
    # quats[b] = rot * quats[b]
    # ph = np.array([r.apply([1, 0, 0]) for r in quats["hip"]])
    # mh = ph[:, 1] / ph[:, 0]
    # for b in ["rf", "lf", "rt", "lt"]:
    #     p = np.array([r.apply([0, 0, -1]) for r in quats[b]])
    #     x = p[:, 0].reshape(-1, 1)
    #     y = p[:, 1]
    #     model = LinearRegression()
    #     model.fit(x[200:300], y[200:300])
    #     init_correction_angle = model.coef_[0]
    #     for i in range(1, len(quats[b]), 1):
    #         # m1 = model.coef_[0]
    #         angle = np.arctan(init_correction_angle) + np.arctan(mh[i])
    #         # angle = np.arctan(m1)
    #         # print('pendiente: ', model.coef_[0])
    #         # print('angulo: ', np.arctan(model.coef_[0]))
    #         # rot = R.from_euler('z', [-np.arctan(model.coef_[0])])
    #         rot = R.from_euler("z", -angle)
    #         quats[b] = rot * quats[b]
    # plt.plot(quats['rf'].as_euler('ZXY'), '--')
    # plt.show()
    return quats


def mcd_reconstruction(mcd_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    direction = mcd_data["measures"]["direction"]

    points = {
        "hip_r": np.array([0, mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_r": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_r": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
        "hip_l": np.array([0, -mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_l": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_l": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
    }
    quats = get_orientation(mcd_data)

    # fix_rot_r = R.from_euler('Z', -np.mean(rot_r[:, 0])*.5, degrees=True).as_quat(scalar_first=True)
    # fix_rot_l = R.from_euler('Z', -np.mean(rot_l[:, 0])*.5, degrees=True).as_quat(scalar_first=True)
    # fix_rot_rk = R.from_euler('Z', -np.mean(rot_rk[:, 0])*.5, degrees=True).as_quat(scalar_first=True)
    # fix_rot_lk = R.from_euler('Z', -np.mean(rot_lk[:, 0])*.5, degrees=True).as_quat(scalar_first=True)
    # for i in range(len(quats['rf'])):
    #     quats['rf'][i] = multiplicacion(fix_rot_r, quats['rf'][i])
    #     quats['rt'][i] = multiplicacion(fix_rot_r, quats['rt'][i])
    #     quats['rt'][i] = multiplicacion(fix_rot_rk, quats['rt'][i])
    #     quats['lf'][i] = multiplicacion(fix_rot_l, quats['lf'][i])
    #     quats['lt'][i] = multiplicacion(fix_rot_l, quats['lt'][i])
    #     quats['lt'][i] = multiplicacion(fix_rot_lk, quats['lt'][i])

    pose = get_all_poses(quats, points)
    # plt.show()
    reconstruction = {
        "hip_r": np.empty((0, 3)),
        "hip_l": np.empty((0, 3)),
        "knee_r": np.empty((0, 3)),
        "knee_l": np.empty((0, 3)),
        "ankle_r": np.empty((0, 3)),
        "ankle_l": np.empty((0, 3)),
    }

    last_T_al = np.eye(4)
    last_T_ar = np.eye(4)
    # plt.plot(mcd_data['time'][1:], np.diff(
    #     pose['ankle_l'][:, 0]), label='ankle_l')
    # plt.plot(mcd_data['time'][1:], np.diff(
    #     pose['ankle_r'][:, 0]), label='ankle_r')
    # plt.show()
    a_r_x_speed = np.diff(pose["ankle_r"][:, 0])
    a_l_x_speed = np.diff(pose["ankle_l"][:, 0])
    for i in range(len(mcd_data["time"])):
        if i == 0:
            z_compare = 1 if direction == "R" else 0
        else:
            if abs(pose["ankle_l"][i][2] - pose["ankle_r"][i][2]) > 0.003:
                z_compare = pose["ankle_l"][i][2] - pose["ankle_r"][i][2]

            # z_compare = rot_lk[i,2] < rot_rk[i, 2]
            # if z_compare == 1:
            #     z_compare = 0 <
        v_rt = mcd_data["gyro_RT"][i, 1]
        v_lt = mcd_data["gyro_LT"][i, 1]
        if z_compare < 0:
            if v_lt > 0 and v_lt > v_rt:
                z_compare = 1
        else:
            if v_rt > 0 and v_rt > v_lt:
                z_compare = -1
        # traslacion tobillo al origen
        if z_compare > 0:
            T_a_2_o = transformation_matrix(
                pose["ankle_r"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_ar
        else:
            T_a_2_o = transformation_matrix(
                pose["ankle_l"][i], R.from_quat([1, 0, 0, 0], scalar_first=True)
            )
            T_pivot = last_T_al

        T_acu = np.linalg.inv(T_a_2_o) @ T_pivot
        reconstruction["hip_r"] = np.append(
            reconstruction["hip_r"],
            [apply_transformation(T_acu, pose["hip_r"][i])],
            axis=0,
        )
        reconstruction["hip_l"] = np.append(
            reconstruction["hip_l"],
            [apply_transformation(T_acu, pose["hip_l"][i])],
            axis=0,
        )
        reconstruction["knee_r"] = np.append(
            reconstruction["knee_r"],
            [apply_transformation(T_acu, pose["knee_r"][i])],
            axis=0,
        )
        reconstruction["knee_l"] = np.append(
            reconstruction["knee_l"],
            [apply_transformation(T_acu, pose["knee_l"][i])],
            axis=0,
        )
        reconstruction["ankle_r"] = np.append(
            reconstruction["ankle_r"],
            [apply_transformation(T_acu, pose["ankle_r"][i])],
            axis=0,
        )
        reconstruction["ankle_l"] = np.append(
            reconstruction["ankle_l"],
            [apply_transformation(T_acu, pose["ankle_l"][i])],
            axis=0,
        )

        last_T_al = transformation_matrix(
            reconstruction["ankle_l"][-1], R.from_quat([1, 0, 0, 0], scalar_first=True)
        )
        last_T_ar = transformation_matrix(
            reconstruction["ankle_r"][-1], R.from_quat([1, 0, 0, 0], scalar_first=True)
        )

    # fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(7, 4))
    # fig.suptitle(title)
    # # ax.plot(rot_rk[:, 0], label='Flexión-Extensión Z')
    # # ax.plot(rot_rk[:, 1], label='Flexión-Extensión X')
    # # ax.plot(rot_rk[:, 2], label='Flexión-Extensión Y')
    # ax[1, 1].set_title('Cadera Derecha')
    # ax[1, 1].plot(rot_r[:,0], label='rf Z')
    # ax[1, 1].plot(rot_r[:,1], label='rf x')
    # ax[1, 1].plot(rot_r[:,2], label='rf y')
    # ax[1, 1].legend()
    # ax[1, 1].grid(1)
    # ax[2, 1].set_title('Cadera Izquierda')
    # ax[2, 1].grid(1)
    # ax[2, 1].plot(rot_l[:,0], label='lf Z')
    # ax[2, 1].plot(rot_l[:,1], label='lf x')
    # ax[2, 1].plot(rot_l[:,2], label='lf y')
    # ax[2, 1].legend()
    # ax[1, 0].set_title('Rodilla Derecha')
    # ax[1, 0].grid(1)
    # ax[1, 0].plot(rot_rk[:,0], label='rk Z')
    # ax[1, 0].plot(rot_rk[:,1], label='rk x')
    # ax[1, 0].plot(rot_rk[:,2], label='rk y')
    # ax[1, 0].legend()
    # ax[2, 0].set_title('Rodilla Izquierda')
    # ax[2, 0].grid(1)
    # ax[2, 0].plot(rot_lk[:,0], label='lk Z')
    # ax[2, 0].plot(rot_lk[:,1], label='lk x')
    # ax[2, 0].plot(rot_lk[:,2], label='lk y')
    # ax[2, 0].legend()

    return reconstruction


def euclidean_error_3Dpoint(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean error between two 3D points.

    Args:
        a (numpy.ndarray): First 3D point.
        b (numpy.ndarray): Second 3D point.

    Returns:
        float: Euclidean error.
    """
    return np.linalg.norm(a - b)


def mean_euclidian_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the Euclidean error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Euclidean error.
    """
    return np.linalg.norm(a - b, axis=1).mean()


def components_error(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return a - b


def root_mean_square_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the root mean square error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Root mean square error.
    """
    return np.sqrt(np.mean(np.linalg.norm(a - b, axis=1) ** 2))


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Mean absolute error.
    """
    euclidean = np.linalg.norm(a - b, axis=1)
    return np.abs(euclidean).mean()


def mean_absolute_error2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the mean absolute error between two arrays of 3D points.

    Args:
        a (numpy.ndarray): First array of 3D points.
        b (numpy.ndarray): Second array of 3D points.

    Returns:
        float: Mean absolute error.
    """
    return np.mean(np.abs(a - b), axis=1).mean()


def interpolate_point(t, point):
    return [
        interp1d(t, point[:, 0]),
        interp1d(t, point[:, 1]),
        interp1d(t, point[:, 2]),
    ]


def homogenize_3dpoint_data(data_a, data_b, timestamp_a, timestamp_b):
    a_int = interpolate_point(timestamp_a, data_a)
    b_int = interpolate_point(timestamp_b, data_b)
    t_min = max(min(timestamp_a), min(timestamp_b))
    t_max = min(max(timestamp_a), max(timestamp_b))
    c1 = timestamp_a >= t_min
    c2 = timestamp_a <= t_max
    t = timestamp_a[c1 & c2]

    a_out = np.array([a_int[i](t) for i in range(3)]).T
    b_out = np.array([b_int[i](t) for i in range(3)]).T
    return t, a_out, b_out


def measures_calculation(aruco, direction):
    femur_common_time = set(aruco[f"knee_{direction.lower()}"]["time"]).intersection(
        set(aruco[f"hip_{direction.lower()}"]["time"])
    )
    femur_lengths = np.empty(0)
    tibia_lengths = np.empty(0)
    for t in femur_common_time:
        knee_idx = np.where(aruco[f"knee_{direction.lower()}"]["time"] == t)
        knee = aruco[f"knee_{direction.lower()}"]["data"][knee_idx][0]
        hip_idx = np.where(aruco[f"hip_{direction.lower()}"]["time"] == t)
        hip = aruco[f"hip_{direction.lower()}"]["data"][hip_idx][0]

        f_length = np.sqrt(
            (knee[0] - hip[0]) ** 2 + (knee[1] - hip[1]) ** 2 + (knee[2] - hip[2]) ** 2
        )

        femur_lengths = np.append(femur_lengths, f_length)

    tibia_common_time = set(aruco[f"knee_{direction.lower()}"]["time"]).intersection(
        set(aruco[f"ankle_{direction.lower()}"]["time"])
    )
    for t in tibia_common_time:
        knee_idx = np.where(aruco[f"knee_{direction.lower()}"]["time"] == t)
        knee = aruco[f"knee_{direction.lower()}"]["data"][knee_idx][0]
        ankle_idx = np.where(aruco[f"ankle_{direction.lower()}"]["time"] == t)
        ankle = aruco[f"ankle_{direction.lower()}"]["data"][ankle_idx][0]
        t_length = np.sqrt(
            (knee[0] - ankle[0]) ** 2
            + (knee[1] - ankle[1]) ** 2
            + (knee[2] - ankle[2]) ** 2
        )
        tibia_lengths = np.append(tibia_lengths, t_length)

    return femur_lengths.mean(), tibia_lengths.mean()


def process_files(files: tuple[str, str]) -> any:
    # Lee los datos de los archivos MCD (sensores inerciales) y ArUco (ground truth).
    mcd_data = read_mcd_file(files[0])
    aruco_data = read_aruco_file(files[1])

    # Determina la dirección de la marcha y la pierna que no se ve (oculta).
    direction = mcd_data["measures"]["direction"]
    hidden_leg = "L" if direction == "R" else "R"

    # Ajusta los datos de ArUco para que comiencen en el origen (0,0,0).
    aruco_data = adjust_aruco_data(aruco_data, direction)
    # Reorganiza los datos de ArUco en un formato más conveniente para el procesamiento.
    aruco_processed = {
        "hip_r": aruco_data["hip_position"],
        "hip_l": aruco_data["hip_position"],
        "knee_r": aruco_data["R_knee_position"],
        "knee_l": aruco_data["L_knee_position"],
        "ankle_r": aruco_data["R_ankle_position"],
        "ankle_l": aruco_data["L_ankle_position"],
    }
    # Calcula las longitudes del fémur y la tibia a partir de los datos de ArUco.
    femur_length, tibia_length = measures_calculation(aruco_processed, direction)
    # Actualiza las medidas en los datos del MCD con las longitudes calculadas.
    mcd_data["measures"]["femur_length"] = femur_length
    mcd_data["measures"]["tibia_length"] = tibia_length

    # Realiza la reconstrucción del esqueleto a partir de los datos de los sensores inerciales.
    reconstruction = mcd_reconstruction(mcd_data)
    # Define un ángulo de rotación objetivo para alinear la reconstrucción con el sistema de coordenadas global.
    z_target = 0 if direction == "L" else np.pi
    # Crea una matriz de transformación para aplicar la corrección de rotación.
    T_fix = transformation_matrix([0, 0, 0], R.from_euler("ZXY", [z_target, 0, 0]))
    # Aplica la matriz de transformación a todos los puntos de la reconstrucción.
    for i in range(reconstruction["hip_r"].shape[0]):
        reconstruction["hip_r"][i] = apply_transformation(
            T_fix, reconstruction["hip_r"][i]
        )
        reconstruction["hip_l"][i] = apply_transformation(
            T_fix, reconstruction["hip_l"][i]
        )
        reconstruction["knee_r"][i] = apply_transformation(
            T_fix, reconstruction["knee_r"][i]
        )
        reconstruction["knee_l"][i] = apply_transformation(
            T_fix, reconstruction["knee_l"][i]
        )
        reconstruction["ankle_r"][i] = apply_transformation(
            T_fix, reconstruction["ankle_r"][i]
        )
        reconstruction["ankle_l"][i] = apply_transformation(
            T_fix, reconstruction["ankle_l"][i]
        )

    # Homogeniza los datos, asegurando que los datos de MCD y ArUco tengan las mismas marcas de tiempo.
    homogeneous_data = {}
    for key in reconstruction.keys():
        # if key in [f'hip_{hidden_leg.lower()}', f'knee_{hidden_leg.lower()}', f'ankle_{hidden_leg.lower()}']:
        # Omite la cadera de la pierna oculta, ya que es la misma que la otra.
        if key == f"hip_{hidden_leg.lower()}":
            continue
        # Interpola y sincroniza los datos de reconstrucción y ArUco.
        t, a_out, b_out = homogenize_3dpoint_data(
            reconstruction[key],
            aruco_processed[key]["data"],
            mcd_data["time"],
            aruco_processed[key]["time"],
        )
        trim_min = t > 1000
        trim_max = t < t.max() - 500
        trim = trim_min & trim_max
        # trim_t = t.copy()[trim]
        # Almacena los datos sincronizados y recortados.
        homogeneous_data[key] = {
            "time": t[trim],
            "mcd": a_out[trim],
            "aruco": b_out[trim],
        }

    common_time = set()
    # Calcula varias métricas de error para cada articulación.
    # print()
    for key in homogeneous_data.keys():
        # Error euclidiano entre la reconstrucción y el ground truth.
        homogeneous_data[key]["euclidean"] = np.linalg.norm(
            homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"], axis=1
        )
        homogeneous_data[key]["euclidean_mean"] = np.mean(
            homogeneous_data[key]["euclidean"]
        )
        homogeneous_data[key]["mae"] = mean_absolute_error(
            homogeneous_data[key]["mcd"], homogeneous_data[key]["aruco"]
        )
        homogeneous_data[key]["mae_mean"] = np.mean(homogeneous_data[key]["mae"])
        # Varianza y desviación estándar del error euclidiano.
        homogeneous_data[key]["eu_var"] = np.var(homogeneous_data[key]["euclidean"])
        homogeneous_data[key]["eu_std"] = np.std(homogeneous_data[key]["euclidean"])
        # Encuentra el conjunto de tiempos comunes a todos los puntos de datos.
        if len(common_time) == 0:
            common_time = set(homogeneous_data[key]["time"])
        else:
            common_time = common_time.intersection(set(homogeneous_data[key]["time"]))

    # Calcula el error euclidiano medio en cada timestamp donde se encuentren los 5 puntos.
    common_time = sorted(list(common_time))
    five_points = np.empty((0, len(common_time)))
    for key in homogeneous_data.keys():
        # Filtra los errores euclidianos para que coincidan con los tiempos comunes.
        isin = np.isin(homogeneous_data[key]["time"], list(common_time))
        try:
            five_points = np.append(
                five_points, [homogeneous_data[key]["euclidean"][isin]], axis=0
            )
        except Exception as e:
            print(key)
            print(e)
            print(files)
            pprint(five_points)
            pprint([homogeneous_data[key]["euclidean"][isin]])
            exit()

    # Prepara arrays para almacenar las posiciones de MCD y ArUco en los tiempos comunes.
    mcd_common = np.empty((0, len(common_time), 3))
    aruco_common = np.empty((0, len(common_time), 3))
    for key in homogeneous_data.keys():
        isin = np.isin(homogeneous_data[key]["time"], common_time)
        mcd_common = np.append(mcd_common, [homogeneous_data[key]["mcd"][isin]], axis=0)
        aruco_common = np.append(
            aruco_common, [homogeneous_data[key]["aruco"][isin]], axis=0
        )

    # Calcula el error euclidiano general promediando las posiciones de todos los puntos.
    gen_eucliden_error = np.empty((0))
    d_all_position = np.empty((0, 3))
    m_all_position = np.empty((0, 3))
    for i in range(len(common_time)):
        d = mcd_common[:, i, :]
        m = aruco_common[:, i, :]
        d_position = np.mean(d, axis=0)
        m_position = np.mean(m, axis=0)
        d_all_position = np.append(d_all_position, [d_position], axis=0)
        m_all_position = np.append(m_all_position, [m_position], axis=0)
        err = euclidean_error_3Dpoint(d_position, m_position)
        gen_eucliden_error = np.append(gen_eucliden_error, err)
    # Calcula la correlación de Pearson para cada eje.
    x_r, x_p_v = pearsonr(d_all_position[:, 0], m_all_position[:, 0])
    y_r, y_p_v = pearsonr(d_all_position[:, 1], m_all_position[:, 1])
    z_r, z_p_v = pearsonr(d_all_position[:, 2], m_all_position[:, 2])

    subject = files[0].split(split_char)[-2]
    capture = files[0].split(split_char)[-1].split(".")[0][-1]
   
    '''
    # Grafica 3D de la trajectoria de las acticulaciones
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    # print(homogeneous_data.keys())
    for key in homogeneous_data.keys():
        mcd = ax.plot(
            homogeneous_data[key]["mcd"][:, 0],
            homogeneous_data[key]["mcd"][:, 1],
            homogeneous_data[key]["mcd"][:, 2],
            label=f"MCD {key}",
            linewidth=1.5,
            color="b",
            alpha=0.5,
        )
        aruco = ax.plot(
            homogeneous_data[key]["aruco"][:, 0],
            homogeneous_data[key]["aruco"][:, 1],
            homogeneous_data[key]["aruco"][:, 2],
            label=f"aruco {key}",
            linewidth=1.5,
            color="r",
            alpha=0.5,
        )
    # Pose de la reconstrucción
    reconstruction_pose = np.array(
        [
            reconstruction[key][-240]
            for key in ["ankle_l", "knee_l", "hip_l", "hip_r", "knee_r", "ankle_r"]
        ]
    )
    ax.plot(
        reconstruction_pose[:, 0],
        reconstruction_pose[:, 1],
        reconstruction_pose[:, 2],
        linewidth=3,
        color="b",
        marker="o",
        markersize=5,
        label="Reconstruction",
        alpha=0.6,
    )
    # Posicion media de las articulaciones
    ax.plot(
        d_all_position[:, 0],
        d_all_position[:, 1],
        d_all_position[:, 2],
        label="MCD",
        color="blue",
        marker="o",
        markersize=2,
        alpha=0.5,
    )
    ax.plot(
        m_all_position[:, 0],
        m_all_position[:, 1],
        m_all_position[:, 2],
        label="arUco",
        color="red",
        marker="x",
        markersize=2,
        alpha=0.5,
    )
    ax.legend([mcd[0], aruco[0]], ["Reconstruction", "Ground Truth"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)
    ax.axis("equal")
    ax.set_title("Best")
    plt.show()
    fig.savefig(
        "NewFigs/Fig5.tiff",
        dpi=300,
        bbox_inches="tight",
    )
    '''

    # plt.plot(sorted(common_time), gen_eucliden_error,
    #          label='General Euclidean Error')
    # plt.show()
    # print(f'5p-mean - {np.mean(five_points, axis=0).mean()}')
    # print(f'var - {np.var(np.mean(five_points, axis=0))}')
    # print(f'std - {np.std(np.mean(five_points, axis=0))}')
    # print(f'min - {np.mean(five_points, axis=0).min()}')
    # print(f'max - {np.mean(five_points, axis=0).max()}')

    # fig0, efp_axs = plt.subplots(5, 1, figsize=(10, 10))
    # fig1, efp_diff_axs = plt.subplots(5, 1, figsize=(10, 10))
    # fig2, comp_axis = plt.subplots(5, 1, figsize=(10, 10))
    # for i, key in enumerate(homogeneous_data.keys()):
    #     efp_axs[i].plot(homogeneous_data[key]['time'],
    #                     homogeneous_data[key]['euclidean'])
    #     efp_axs[i].set_title(f'Euclidean error {key}')
    #     efp_axs[i].set_xlabel('Time (ms)')
    #     efp_axs[i].set_ylabel('Euclidean error (m)')
    #     efp_axs[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #     efp_axs[i].set_ylim(0, 1)
    #     efp_axs[i].grid(1)

    #     efp_diff_axs[i].plot(homogeneous_data[key]['time'][1:],
    #                          np.diff(homogeneous_data[key]['euclidean']))
    #     efp_diff_axs[i].set_title(f'Euclidean error {key}')
    #     efp_diff_axs[i].set_xlabel('Time (ms)')
    #     efp_diff_axs[i].set_ylabel('Euclidean error (m)')
    #     efp_diff_axs[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #     efp_diff_axs[i].grid(1)

    #     for n in range(3):
    #         comp_axis[i].plot(homogeneous_data[key]['time'],
    #                           homogeneous_data[key]['mcd'][:, n]-homogeneous_data[key]['aruco'][:, n])
    #         comp_axis[i].set_title(f'{key} - {n}')
    #         comp_axis[i].set_xlabel('Time (ms)')
    #         comp_axis[i].set_ylabel('Position (m)')
    #         comp_axis[i].set_xlim(-500, np.max(mcd_data['time'])+500)
    #         comp_axis[i].set_ylim(-0.5, 0.5)
    #         comp_axis[i].grid(1)

    # Calcula el error absoluto medio por eje para la pierna visible.
    axis_error = np.empty((0, 3))
    for key in homogeneous_data.keys():
        if key[-1] != hidden_leg.lower():
            # print(direction, key)
            continue
        axis_error = np.append(
            axis_error,
            [
                [
                    np.mean(
                        np.abs(
                            homogeneous_data[key]["mcd"][:, i]
                            - homogeneous_data[key]["aruco"][:, i]
                        )
                    )
                    for i in range(3)
                ]
            ],
            axis=0,
        )

    # print(np.mean(axis_error, axis=0))

    # plt.show()
    # return np.mean(five_points, axis=0).mean()
    # Devuelve los resultados del procesamiento.
    # print(type(homogeneous_data))
    return (
        gen_eucliden_error,
        sorted(list(common_time)),
        reconstruction,
        aruco_processed,
        homogeneous_data,
    )


def animation(rec, aruco, title):
    fig, ax = plt.figure(), plt.axes(projection="3d")
    fig.subplots_adjust(left=0, right=0.99, top=0.98, bottom=0.03, hspace=0, wspace=0.2)
    mcd = ax.plot([], [], [], "o-", label="MCD", color="blue")[0]
    aru = ax.plot([], [], [], "o-", label="arUco", color="red")[0]
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.suptitle(title)
    ax.legend()
    fig.show()
    i = 0

    aruco_inter = {k: interpolate_point(v["time"], v["data"]) for k, v in aruco.items()}
    # print(aruco_inter.keys())
    while True:
        try:
            aruco_pose = np.empty((0, 3))
            for k in ["ankle_r", "knee_r", "hip_r", "hip_l", "knee_l", "ankle_l"]:
                try:
                    aruco_pose = np.append(
                        aruco_pose,
                        [[inter(i * 10) for inter in aruco_inter[k]]],
                        axis=0,
                    )
                except:
                    pass

            pose = np.empty((0, 3))
            for k in ["ankle_r", "knee_r", "hip_r", "hip_l", "knee_l", "ankle_l"]:
                pose = np.append(pose, [rec[k][i]], axis=0)
            mcd.set_data_3d(pose[:, 0], pose[:, 1], pose[:, 2])
            aru.set_data_3d(aruco_pose[:, 0], aruco_pose[:, 1], aruco_pose[:, 2])
            ax.set_xlim(np.mean(pose[3:5, 0]) - 0.5, np.mean(pose[3:5, 0]) + 0.5)
            ax.set_ylim(np.mean(pose[3:5, 1]) - 0.5, np.mean(pose[3:5, 1]) + 0.5)
            # ax.set_zlim(np.mean(pose[3:5, 0])-.5, np.mean(pose[3:5, 0])+.5)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

            i += 1
            if i >= len(rec["hip_r"]):
                i = 0
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    files = get_dataset()
    if len(files) == 0:
        print("No files selected")
        exit(1)
    # print(f'capture | x_r | x_p-value | y_r | y_p-value | z_r | z_p-value')
    gen_errors = [process_files(f) for f in files if f[0].split("/")[-2] != "0"]
    # print(type(gen_errors[-1]))
    # print(type(gen_errors[0][-1]))
    mean_error_per_point = np.empty(0)
    for file in gen_errors:
        mean_error_per_point = np.append(
            mean_error_per_point,
            np.mean([file[-1][key]["euclidean_mean"] for key in file[-1].keys()]),
        )
    # print(len(mean_error_per_point))

    gen_mean_error = np.array([np.mean(err[0]) for err in gen_errors])
    # print("voluntario/captura;mean;max;min")
    # for i in range(len(files)):
    #     file = "/".join(files[i][0].split(split_char)[-2:])
    #     print(
    #         f"{file};{gen_mean_error[i]:.4f};{np.max(gen_errors[i][0]):.4f};{np.min(gen_errors[i][0]):.4f}"
    #     )

    print(f"General mean error: {np.mean(gen_mean_error)}")
    print(f"General variance: {np.var(gen_mean_error)}")
    print(f"General standard deviation: {np.std(gen_mean_error)}")
    print(f"General minimum: {np.min(gen_mean_error)}")
    print(f"General maximum: {np.max(gen_mean_error)}")
    print("---")
    print(f"mean: {np.mean(mean_error_per_point)}")
    print(f"std: {np.std(mean_error_per_point)}")
    print(f"min: {np.min(mean_error_per_point)}")
    print(f"max: {np.max(mean_error_per_point)}")
    # for err in gen_errors:
    #     animation(err[2], err[3], "")

    best_idx = np.argmin(gen_mean_error)
    worst_idx = np.argmax(gen_mean_error)
    mean_idx = np.argmin(np.abs(gen_mean_error - np.mean(gen_mean_error)))
    # print(f"Best: {gen_mean_error[best_idx]}")
    # print(f"Worst: {gen_mean_error[worst_idx]}")
    # print(f"Mean: {gen_mean_error[mean_idx]}")

    # for err in np.sort(gen_mean_error):
    # print(err)

    # print(means)
    fig_best, ax_best = plt.subplots(1, 1, figsize=figsize)
    fig_worst, ax_worst = plt.subplots(1, 1, figsize=figsize)
    fig_mean, ax_mean = plt.subplots(1, 1, figsize=figsize)
    ax_best.plot(
        gen_errors[best_idx][1],
        gen_errors[best_idx][0],
        label="Best error",
        color="green",
    )
    ax_best.set_title(f"Best error")
    ax_best.set_xlabel("Time (ms)")
    ax_best.set_ylabel("Euclidean error (m)")
    ax_best.grid(1)
    ax_best.set_ylim(0, 1)
    ax_worst.plot(
        gen_errors[worst_idx][1],
        gen_errors[worst_idx][0],
        label="Worst error",
        color="red",
    )
    ax_worst.set_title(f"Worst error")
    ax_worst.set_xlabel("Time (ms)")
    ax_worst.set_ylabel("Euclidean error (m)")
    ax_worst.grid(1)
    ax_worst.set_ylim(0, 1)
    ax_mean.plot(
        gen_errors[mean_idx][1],
        gen_errors[mean_idx][0],
        label="Mean error",
        color="blue",
    )
    ax_mean.set_title(f"Average error")
    ax_mean.set_xlabel("Time (ms)")
    ax_mean.set_ylabel("Euclidean error (m)")
    ax_mean.grid(1)
    ax_mean.set_ylim(0, 1)

    all_fig, all_ax = plt.subplots(1, 1, figsize=figsize)
    for i, (err, t, *_) in enumerate(gen_errors):
        all_ax.plot(t, err, label=f"File {i+1}")
    all_ax.set_title("All errors")
    all_ax.set_xlabel("Time (ms)")
    all_ax.set_ylabel("Euclidean error (m)")
    all_ax.grid(1)
    all_ax.set_ylim(0, 1)

    # fig_best.savefig(
    #     "NewFigs/Fig3.tif",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pil_kwargs={"compression": "tiff_lzw"},
    # )
    # fig_mean.savefig(
    #     "NewFigs/Fig2.tif",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pil_kwargs={"compression": "tiff_lzw"},
    # )
    # fig_worst.savefig(
    #     "NewFigs/Fig4.tif",
    #     dpi=300,
    #     bbox_inches="tight",
    #     pil_kwargs={"compression": "tiff_lzw"},
    # )

    # fig3, ax3 = plt.subplots(1, 1, figsize=figsize)
    # ax3.plot(gen_errors[mean_idx][1], gen_errors[mean_idx][0],
    #          label='Mean error', color='blue')
    # ax3.plot(gen_errors[best_idx][1], gen_errors[best_idx][0],
    #          label='Best error', color='green')
    # ax3.plot(gen_errors[worst_idx][1], gen_errors[worst_idx][0],
    #          label='Worst error', color='red')
    # ax3.set_title('Mean, Best and Worst errors')
    # ax3.set_xlabel('Time (ms)')
    # ax3.set_ylabel('Euclidean error (m)')
    # ax3.grid(1)
    # ax3.set_ylim(0, 1)
    # ax3.legend()
    # plt.show()

    print(f"Best file: {files[best_idx]} | idx: {best_idx}")
    # print(f"Worst file: {files[worst_idx]} | idx {worst_idx}")
    # print(f"Mean file: {files[mean_idx]} | idx {mean_idx}")

    # print(f'Average: {np.mean(gen_errors)}')
    # print(f'Variance: {np.var(gen_errors)}')
    # print(f'Standard deviation: {np.std(gen_errors)}')
    # print(f'Minimum: {np.min(gen_errors)}')
    # print(f'Maximum: {np.max(gen_errors)}')
    # for i, f in enumerate(files):
    #     print(f'Processing file {i} | {f}')
    # Best = (
    #     "/media/moybadajoz/Nuevo vol/DataSet With Vid (copy)/1/mcd-0.csv",
    #     "/media/moybadajoz/Nuevo vol/DataSet With Vid (copy)/1/arUcos-0.csv",
    # )
    # # Mean = ('/home/moybadajoz/Documents/NewDataset/29/mcd-3.csv',
    # #         '/home/moybadajoz/Documents/NewDataset/29/arUcos-3.csv')
    # worst = ('/home/moybadajoz/Documents/NewDataset/11/mcd-3.csv',
    #          '/home/moybadajoz/Documents/NewDataset/11/arUcos-3.csv')
    # title = 'worst'
    # worst_mean, _, worst_reconstruction, worst_aruco = process_files(files[worst_idx])
    # # title = 'mean'
    # # # process_files(Mean)
    # title = 'best'
    # best_mean, _, best_reconstruction, best_aruco = process_files(Best)
    # best_mean, _, best_reconstruction, best_aruco =process_files(Best)
    # print(f'Best mean error: {np.mean(best_mean)}')
    # # print(f'Best var error: {np.var(best_mean)}')
    # # print(f'Worst mean error: {np.mean(worst_mean)}')
    # # # print(f'Worst var error: {np.var(worst_mean)}')
    # animation(best_reconstruction, best_aruco, "best")
    # animation(worst_reconstruction, worst_aruco, "worst")

    # process_files(Mean)
    # plt.grid(1)
    # plt.show()
