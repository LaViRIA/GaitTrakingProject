"""
Author: Moises Badajoz Martinez <m.badajozmartinez@ugto.mx>

University of Guanajuato (2025)
"""

import cv2
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from utils import calibration_matix as cm
import csv
import scipy.interpolate as sci
import argparse
from pymediainfo import MediaInfo
import sys
from scipy.signal import butter, filtfilt
from pprint import pprint
import time


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
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # Apply the filter using filtfilt for zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    return filtered_data


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


def csv_comment(comment):
    line = comment.replace("# ", "")
    comment = line.strip()
    measure = comment.split(";")[1:]
    measures = {
        m.split(":")[0]: n if (n := m.split(":")[1].strip()).isalpha() else float(n)
        for m in measure
    }
    return measures


def get_object_points(marker_length):
    obj_points = np.array(
        [
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0],
        ],
        dtype=np.float32,
    )
    return obj_points


def get_transformation_matrix(rvec, tvec):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    transformation_matrix[:3, 3] = tvec.flatten()
    return transformation_matrix


def process_mcd_file(path):
    measures = None
    mcd_file = None
    with open(path, "r") as file:
        measures = csv_comment(file.readline())
        mcd_file = [r for r in csv.DictReader(file, delimiter=";")]
    mcd_data = {
        k: np.array([np.array(data[k].split(","), dtype=float) for data in mcd_file])
        for k in mcd_file[0].keys()
    }
    mcd_t = mcd_data["time"].flatten()
    mcd_f_av = mcd_data[f'gyro_{measures["direction"].strip()}F'][:, 1] / 131.0 / 1000.0
    mcd_t_av = mcd_data[f'gyro_{measures["direction"].strip()}T'][:, 1] / 131.0 / 1000.0

    return measures, mcd_t, mcd_f_av, mcd_t_av


def apply_wiener_filter(img, kernel_size, noise_power):

    # Crear el kernel de desenfoque lineal
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[
        int((kernel_size - 1) / 2) : int((kernel_size - 1) / 2) + 1, :
    ] = (np.ones(kernel_size) / kernel_size)

    # Transformada de Fourier de la imagen y el kernel
    img_fft = np.fft.fft2(img)
    kernel_fft = np.fft.fft2(kernel_motion_blur, s=img.shape)

    # Evitar divisiones por cero
    kernel_fft = np.where(kernel_fft == 0, 1e-7, kernel_fft)

    # Filtro de Wiener (desenfoque inverso)
    kernel_wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + noise_power)
    img_wiener = np.fft.ifft2(img_fft * kernel_wiener)

    # Transformar la imagen de vuelta al dominio espacial
    img_result = np.abs(img_wiener)
    img_result = np.clip(img_result, 0, 255).astype(np.uint8)
    # Aplicar un gaussian blur para mejorar la deteccion
    result = cv2.GaussianBlur(
        img_result, (5, 9), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_CONSTANT
    )
    return result


def get_aruco_file_path(str_path):
    if sys.platform == "linux":
        split_char = "/"
    else:
        split_char = "\\"
    split = str_path.split(split_char)
    path = split_char.join(split[:-1])
    num = split[-1].split(".")[0][-1]
    return f"{path}{split_char}arUcos-{num}.csv"


if __name__ == "__main__":

    debug_mode = False

    parser = argparse.ArgumentParser(description="idk")
    parser.add_argument("mcd", type=str, help="mcd csv file path")
    parser.add_argument("vid", type=str, help="video file path")
    args = parser.parse_args()

    media_info = MediaInfo.parse(args.vid)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    # parameters.adaptiveThreshWinSizeMin = 13 # Valor inicial, prueba subiendo si no detecta
    # parameters.adaptiveThreshWinSizeMax = 33 # Valor inicial, prueba subiendo
    parameters.adaptiveThreshWinSizeStep = 5
    # parameters.polygonalApproxAccuracyRate = 0.0001
    # parameters.minCornerDistanceRate = 0.01
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.errorCorrectionRate = 0.8
    parameters.cornerRefinementMinAccuracy = 0.001
    parameters.cornerRefinementMaxIterations = 100
    # parameters.cornerRefinementWinSize = 10
    # parameters.perspectiveRemoveIgnoredMarginPerCell = 0.3
    # parameters.minMarkerPerimeterRate = 0.05
    # parameters.maxMarkerPerimeterRate = 4.0
    # parameters.useAruco3Detection= True
    parameters.aprilTagQuadSigma = 0.8
    parameters.aprilTagMinWhiteBlackDiff = 75
    # parameters.polygonalApproxAccuracyRate = 0.05
    parameters.aprilTagDeglitch = 1
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    detector_2 = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    markers_size = 0.066  # metros
    origin_size = 0.196  # metros
    arUcos_ids = {0: "hip", 1: "R_knee", 2: "L_knee", 3: "R_ankle", 4: "L_ankle"}
    arUcos_ids_inv = {v: k for k, v in arUcos_ids.items()}
    origin_id = 20
    origin_Tmatrix = None

    origin_3D = get_object_points(origin_size)
    markers_3D = get_object_points(markers_size)

    rows = []
    aruco_m = []
    aruco_t = np.array([])

    ankle_p = np.array([])
    ankle_t = np.array([])

    measures, mcd_t, mcd_f_av, mcd_t_av = process_mcd_file(args.mcd)

    last_p = {0: None, 1: None, 2: None, 3: None, 4: None, 20: None}

    if media_info.tracks[1].width == 4096:
        # camera_matrix = cm.NOTHING_CAMERA_MATRIX_4096
        # dist_coeffs = cm.NOTHING_DIST_COEFFS_4096
        with np.load("calib_cam_4096.npz") as data:
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]
    else:
        with np.load("calib_cam_4k.npz") as data:
            camera_matrix = data["camera_matrix"]
            dist_coeffs = data["dist_coeffs"]

    frame_rate = media_info.tracks[1].real_frame_rate
    cap_fps = (
        float(frame_rate)
        if frame_rate is not None
        else float(media_info.tracks[1].frame_rate)
    )

    cap = cv2.VideoCapture(args.vid)
    if not cap.isOpened():
        print("Error")
        exit()

    ret = True
    frame_number = 0
    last_origin_corners = np.empty((0, 4, 2))
    # Procesamiento de video
    last_origin_centers = np.empty((0, 3))

    while ret:
        # lectura y acondicionamiento del frame
        ret, frame = cap.read()
        if not ret:
            continue
        # cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        if debug_mode:
            print(frame_number, end="\r")
            cv2.imshow("frame", cv2.resize(gray, (1280, 720)))
            cv2.waitKey(1)

        # recorte/limpieza del frame para mejorar rendimiento
        if last_p[0] is None or last_p[20] is None:
            frame_trim = gray
        else:
            frame_trim = np.zeros(gray.shape, np.uint8)

            x = [max(0, last_p[0][0] - 650), min(gray.shape[1] - 1, last_p[0][0] + 650)]
            y = [0, int(gray.shape[0] * 0.80)]
            # frame_trim[y[0] : y[1], x[0] : x[1]] = cv2.fastNlMeansDenoising(
            #     gray[y[0] : y[1], x[0] : x[1]], None, 10, 7, 21
            # )
            frame_trim[y[0] : y[1], x[0] : x[1]] = gray[y[0] : y[1], x[0] : x[1]]

            x = [
                max(0, last_p[20][0] - 150),
                min(gray.shape[1] - 1, last_p[20][0] + 250),
            ]
            y = [
                max(0, last_p[20][1] - 150),
                min(gray.shape[0] - 1, last_p[20][1] + 250),
            ]
            # frame_trim[y[0] : y[1], x[0] : x[1]] = cv2.fastNlMeansDenoising(
            #     gray[y[0] : y[1], x[0] : x[1]], None, 10, 7, 21
            # )
            frame_trim[y[0] : y[1], x[0] : x[1]] = gray[y[0] : y[1], x[0] : x[1]]

        # frame_trim = cv2.fastNlMeansDenoising(frame_trim, None, 10, 7, 21)
        # Deteccion de arucos
        mk_corners, mk_ids, r = detector.detectMarkers(frame_trim)

        if len(mk_corners) <= 0:
            frame_number += 1
            continue

        mk_corners = list(mk_corners)
        mk_ids = mk_ids.flatten()

        # Reintentos de deteccion de arucos
        for i in arUcos_ids.keys():
            # continue
            if i in mk_ids or not isinstance(last_p[i], np.ndarray):
                continue
            if measures["direction"] == "L" and 1 == i:
                continue
            if measures["direction"] == "R" and 2 == i:
                continue
            if isinstance(last_p[0], np.ndarray):
                x_r = last_p[0][0]
            else:
                x_r = last_p[i][0]
            x = [max(0, x_r - 650), min(gray.shape[1] - 1, x_r + 650)]

            y = [max(0, last_p[i][1] - 200), min(gray.shape[0] - 1, last_p[i][1] + 300)]

            if i != 0:
                frame_retry = np.zeros(gray.shape, np.uint8)
            else:
                frame_retry = gray

            for s in range(17, 23, 2):
                for w in np.arange(0.02, 0.032, 0.002):
                    # # for s in range(1):
                    # #     for w in np.arange(1):
                    frame_retry[y[0] : y[1], x[0] : x[1]] = apply_wiener_filter(
                        gray[y[0] : y[1], x[0] : x[1]], kernel_size=s, noise_power=w
                    )
                    # frame_retry[y[0]:y[1], x[0]:x[1]] = cv2.cvtColor(cv2.fastNlMeansDenoisingColored(frame[y[0]:y[1], x[0]:x[1]], None, 10, 10, 7, 21),
                    #                                                  cv2.COLOR_BGR2GRAY)
                    mk_corners_retry, mk_ids_retry, r = detector_2.detectMarkers(
                        frame_retry
                    )
                    if len(mk_corners_retry) <= 0:
                        continue
                    mk_ids_retry = mk_ids_retry.flatten()
                    if isinstance(mk_ids_retry, np.ndarray) and i in mk_ids_retry:
                        index = np.where(mk_ids_retry == i)[0][0]
                        last_p[i] = mk_corners_retry[index][0, 0].astype(int)
                        mk_ids = np.append(mk_ids, i)
                        mk_corners.append(mk_corners_retry[index])
                        break
                else:
                    continue
                break
        timestamp = int((1000 / cap_fps) * frame_number)
        row = {"time": timestamp}
        hip = None
        knee = None
        ankle = None

        # Obtencion de la matriz de transformacion del aruco de referencia
        if origin_id in mk_ids:
            index = np.where(mk_ids == origin_id)[0][0]
            origin_corners = mk_corners[index]
            # last_p[origin_id] = origin_corners[0, 0].astype(int)
            # if last_origin_corners.shape[0] <= 15:
            #     last_origin_corners = np.append(last_origin_corners,
            #                             origin_corners, axis=0)
            # o_corners = last_origin_corners.mean(axis=0, dtype=np.double)
            # success, rvec, tvec = cv2.solvePnP(
            #     origin_3D,
            #     o_corners,
            #     camera_matrix,
            #     dist_coeffs
            # )
            # print(type(origin_corners))

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners=origin_corners,
                markerLength=origin_size,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
            )
            # cx = np.mean(origin_corners[:, 0])
            # cy = np.mean(origin_corners[:, 1])
            c_current = tvec.flatten()

            # pprint(last_origin_corners.mean(axis=0))
            # print(f'\n{np.linalg.norm(tvec)}')
            if last_origin_centers.shape[0] >= 15:
                distance = np.linalg.norm(
                    c_current - np.mean(last_origin_centers[-15:], axis=0)
                )
                # c_current - np.mean(last_origin_centers[0:30], axis=0))
                # if distance > 1.0:
                # print(f'\n{distance=:.4f}')
            else:
                distance = 0
                # print(last_origin_centers.shape, c_current.shape)
                last_origin_centers = np.append(
                    last_origin_centers, [c_current], axis=0
                )
            if distance < 0.001:

                # if origin_rvec is None:
                #     origin_rvec = rvec
                #     origin_tvec = tvec
                # else:
                #     origin_rvec = alpha * rvec + (1.0 - alpha)* origin_rvec
                #     origin_tvec = alpha * tvec + (1.0 - alpha)* origin_tvec

                origin_Tmatrix = get_transformation_matrix(rvec, tvec)

        # if type(origin_Tmatrix) is not np.ndarray:
        if origin_Tmatrix is None:
            continue
        # Procesamiento y transformacion de los arucos
        for id, corners in zip(mk_ids, mk_corners):
            if id not in arUcos_ids.keys():
                continue
            last_p[id] = corners[0, 0].astype(int)
            # success, rvec, tvec = cv2.solvePnP(
            #     markers_3D,
            #     corners[0],
            #     camera_matrix,
            #     dist_coeffs
            # )
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, markers_size, camera_matrix, dist_coeffs
            )

            # if not success:
            #     continue

            Tmatrix = get_transformation_matrix(rvec, tvec)
            Tmatrix_P_from_Origin = inv(origin_Tmatrix) @ Tmatrix
            coors = Tmatrix_P_from_Origin[:3, 3]
            # intercambia Y y Z para estandar XYZ
            coors = np.array([coors])[0, [0, 2, 1]]
            coors[1] *= -1  # invierte Y para eliminar efecto espejo
            rot_matrix = Tmatrix_P_from_Origin[:3, :3]
            orientation = R.from_matrix(rot_matrix).as_quat(scalar_first=True)

            if id == arUcos_ids_inv["hip"]:
                hip = coors[0]
            if id == arUcos_ids_inv[f"{measures['direction'].strip()}_knee"]:
                knee = coors
            if id == arUcos_ids_inv[f"{measures['direction'].strip()}_ankle"]:
                ankle = coors[0]
            row[f"{arUcos_ids[id]}_position"] = ",".join(map(str, coors))
            row[f"{arUcos_ids[id]}_orientation"] = ",".join(map(str, orientation))

        rows.append(row)
        # almacenamiento de los arucos para la sincronizacion
        # if hip.size > 0 and knee.size > 0:
        #     aruco_m.append((hip[0]-knee[0])/(hip[1]-knee[1]))
        #     aruco_t = np.append(aruco_t, int((1000/cap_fps)*frame_number))
        # if ankle.size > ankle_t.size:
        if hip is not None and ankle is not None:
            ankle_p = np.append(ankle_p, ankle - hip)
            ankle_t = np.append(ankle_t, int((1000 / cap_fps) * frame_number))

        frame_number += 1

    # Sincroninacion
    offset = None
    mcd_f_av = low_pass_filter(mcd_f_av, 7, 100)
    mcd_f_p = np.array([np.sum(mcd_f_av[:i]) for i in range(mcd_f_av.size)])
    mcd_t_av = low_pass_filter(mcd_t_av, 7, 100)
    mcd_t_p = np.array([np.sum(mcd_t_av[:i]) for i in range(mcd_t_av.size)])
    mcd_ankle_mov = measures["femur_length"] * np.sin(mcd_f_p) + measures[
        "tibia_length"
    ] * np.sin(mcd_t_p)

    # suavizado de los datos de velocidad angular
    # w = 20
    # mcd_av = np.convolve(mcd_av, np.ones(w)/w, mode='valid')
    # mcd_t = mcd_t[w//2:-(w//2)+1]

    mcd_angle_f = sci.interp1d(
        mcd_t, mcd_ankle_mov, kind="linear", fill_value="extrapolate"
    )
    # print('ankle_t', ankle_t)
    # print('ankle_p', ankle_p)
    aruco_ankle_f = sci.interp1d(
        ankle_t, ankle_p, kind="linear", fill_value="extrapolate"
    )
    # ankle_f = sci.interp1d(ankle_t, ankle, kind='linear',
    #                        fill_value='extrapolate')
    # obtencion del tiempo en comun
    # ti = max(min(aruco_t), min(mcd_t))
    # tf = min(max(aruco_t), max(mcd_t))
    # t = np.arange(ti, tf, 1)
    ti = max(min(ankle_t), min(mcd_t))
    tf = min(max(ankle_t), max(mcd_t))
    t = np.arange(ti, tf, 1)

    mcd_ankle_int = mcd_angle_f(t)
    aruco_ankle_int = aruco_ankle_f(t)
    aruco_ankle_int = (
        aruco_ankle_int if measures["direction"] == "R" else -aruco_ankle_int
    )
    # ankle_f_int = ankle_f(t)
    # calculo de la correlacion y desfase
    # correlation = np.correlate(mcd_av_int, aruco_m_int, mode='full')
    correlation = np.correlate(mcd_ankle_int, aruco_ankle_int, mode="full")
    idx_max = np.argmax(correlation)
    offset = idx_max - (len(aruco_ankle_int) - 1)
    for r in rows:
        r["time"] += offset

    # Guardado del archivo de arucos
    csv_fields = [
        "time",
        "hip_position",
        "hip_orientation",
        "R_knee_position",
        "R_knee_orientation",
        "L_knee_position",
        "L_knee_orientation",
        "R_ankle_position",
        "R_ankle_orientation",
        "L_ankle_position",
        "L_ankle_orientation",
    ]

    file, csv_file = init_csv_file(
        path=get_aruco_file_path(args.mcd), fields=csv_fields, delimiter=";"
    )

    csv_file.writerows(rows)
    file.close()
