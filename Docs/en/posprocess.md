# ArUco Motion Tracking Code Documentation

This code performs motion tracking of ArUco markers in a video and synchronizes data with measurements from an inertial sensor (MCD).

## Dependencies

*   **cv2 (OpenCV):** Library for image processing and computer vision.
*   **numpy:** Library for numerical calculations and array manipulation.
*   **scipy.spatial.transform:** For handling rotations and spatial transformations.
*   **csv:** To read and write CSV files.
*   **scipy.interpolate:** For data interpolation.
*   **argparse:** To parse command-line arguments.
*   **pymediainfo:** To obtain multimedia file information (like video).
*   **sys:** To get the platform on which it is running.
* **utils.calibration_matrix**: Module with camera data.

## Functions

### `init_csv_file(path: str, fields: list[str], delimiter: str, comments: str = None)`

*   **Purpose:** Initializes a CSV file to write results.
*   **Arguments:**
    *   `path` (`str`): The path where the CSV file will be created.
    *   `fields` (`list[str]`): A list of strings representing the CSV header column names.
    *   `delimiter` (`str`): The delimiter character used to separate fields in the CSV (e.g., '`;`', '`,`', '`\t`').
    *   `comments` (`str`, optional): Comments to add to the beginning of the file.
*   **Return:**
    *   A tuple containing:
        *   A file object opened in write mode (`file`).
        *   A `csv.DictWriter` object configured to write dictionaries to the CSV file.

*   **Operation:**
    1.  Opens the file specified in `path` in write mode (`'w'`), with `newline=''` to avoid issues with line breaks on different operating systems.
    2.  If there are comments, writes them.
    3.  Creates a `csv.DictWriter` object. This object will be used to write rows to the CSV, where each row is a dictionary. `fieldnames` define the dictionary keys and column order.
    4.  Writes the header row using `csv_writer.writeheader()`.
    5.  Returns the file object and the CSV writer.

### `csv_comment(comment)`

*   **Purpose:** Extracts data from the calibration file (mcd) comment.
*   **Arguments:**
    *   `comment` (`str`): Comment extracted from the first line of the file.
*   **Return:**
    *   A dictionary with parsed values.
*   **Operation:**
    1.  Removes comments from the string and whitespace.
    2.  Splits the string.
    3.  Creates the dictionary.
    4.  Returns the dictionary.

### `get_object_points(marker_length)`

*   **Purpose:** Defines the 3D coordinates of an ArUco marker's corners in its local coordinate system.
*   **Arguments:**
    *   `marker_length` (`float`): The length of the ArUco marker side (in meters or the unit of measurement being used).
*   **Return:**
    *   A NumPy array (`np.ndarray`) of shape (4, 3) and type `np.float32`. Each row represents a marker corner, with (x, y, z) coordinates. The marker is assumed to be centered at the origin (0, 0, 0) and on the XY plane (z=0).

*   **Operation:**
    *   Creates a NumPy array with the coordinates of the four marker corners. Coordinates are calculated based on `marker_length`.
    *   Sets the array data type to `np.float32`.
    *   Returns the array.

### `get_transformation_matrix(rvec, tvec)`

*   **Purpose:** Creates a 4x4 homogeneous transformation matrix from a rotation vector and a translation vector.
*   **Arguments:**
    *   `rvec` (`np.ndarray`): Rotation vector (3x1 or 1x3) representing object rotation (using Rodrigues representation).
    *   `tvec` (`np.ndarray`): Translation vector (3x1 or 1x3) representing object translation.
*   **Return:**
    *   `transformation_matrix` (`np.ndarray`): A 4x4 homogeneous transformation matrix.

*   **Operation:**
    1.  Creates a 4x4 identity matrix (`np.eye(4)`).
    2.  Converts the rotation vector `rvec` to a 3x3 rotation matrix using `cv2.Rodrigues(rvec)[0]`. The `cv2.Rodrigues()` function converts between Rodrigues representation (rotation vector) and rotation matrix. `[0]` is used to get only the rotation matrix, as `cv2.Rodrigues()` also returns the Jacobian. This rotation matrix is placed in the top-left 3x3 submatrix of the transformation matrix.
    3.  Converts `tvec` to a column vector using `tvec.flatten()` and places it in the last column (index 3) of the first three rows (indices 0:3) of the transformation matrix.
    4.  Returns the resulting 4x4 transformation matrix.

### `process_mcd_file(path)`

*   **Purpose:** Reads and processes an MCD file (inertial sensor data file) in CSV format.
*   **Arguments:**
    *   `path` (`str`): The path to the MCD file.
*   **Return:**
    *   A tuple containing:
        *   `measures` (`dict`): A dictionary with data from the first line of the file.
        *   `mcd_t` (`list[float]`): A list with time values from the MCD file.
        *   `mcd_av` (`list[float]`): A list with angular velocity values calculated from gyroscope data in the MCD file.

*   **Operation:**
    1.  Initializes variables.
    2.  Opens the MCD file in read mode (`'r'`).
    3.  Reads data from the first line.
    4.  Creates a `csv.DictReader` to read the rest of the file as CSV. The delimiter is set to ';'.
    5.  Iterates over rows of the CSV file:
        *   Extracts gyroscope data from the appropriate column (determined by `measures["direction"]`). The specific column is selected using an f-string: `f'gyro_{measures["direction"].strip()}F'`. Gyroscope values are assumed to be a comma-separated string, which is converted to a list of floats.
            > These specific data are relevant for synchronizing ArUco data since `measures["direction"]` indicates the leg that is always visible to the camera.
        *   Calculates angular velocity from gyroscope data. Multiplies by conversion constants (1/131.0, pi/180, and 5) to convert gyroscope units to radians per second.
        *   Adds the calculated angular velocity value to the `mcd_av` list.
        *   Adds the time value ('time' column) to the `mcd_t` list.
    6.  Returns `measures`, `mcd_t`, and `mcd_av`.

### `apply_wiener_filter(img, kernel_size, noise_power)`

*   **Purpose:** Applies a Wiener filter to an image to reduce linear motion blur.
*   **Arguments:**
    *   `img` (`np.ndarray`): The input image (grayscale).
    *   `kernel_size` (`int`): The size of the linear blur kernel (must be odd).
    *   `noise_power` (`float`): An estimate of noise power in the image.
*   **Return:**
    *   `result` (`np.ndarray`): The filtered image, with reduced blur.

*   **Operation:**
    1.  **Create blur kernel:**
        *   Creates a zero kernel (`np.zeros`) of size `kernel_size` x `kernel_size`.
        *   Creates a linear blur kernel, where pixels in the center row represent motion.
    2.  **Calculate Fourier Transforms:**
        *   Calculates the Fast Fourier Transform (FFT) of the image (`img_fft`) using `np.fft.fft2()`.
        *   Calculates the FFT of the kernel (`kernel_fft`), adjusting the kernel size to the image size with the argument `s=img.shape`.
    3.  **Avoid division by zero:**
        *   Replaces any zero value in `kernel_fft` with a very small value (`1e-7`) to avoid division by zero errors in Wiener filter calculation.
    4.  **Calculate Wiener filter:**
        *   Calculates the Wiener filter in the frequency domain. The Wiener filter is an optimal estimate of the inverse filter that minimizes mean squared error. The formula used is: `kernel_wiener = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + noise_power)`.
    5.  **Apply filter:**
        *   Multiplies the image FFT (`img_fft`) by the Wiener filter (`kernel_wiener`).
    6.  **Inverse Transform:**
        *   Calculates the Inverse Fast Fourier Transform (IFFT) of the previous result (`img_wiener`) using `np.fft.ifft2()`. This returns the filtered image to the spatial domain.
    7.  **Adjust result:**
        *   Takes the absolute value of the resulting image (`np.abs(img_wiener)`) as the IFFT may return complex numbers due to numerical errors.
        *   Clips resulting image values to be between 0 and 255 (`np.clip(img_result, 0, 255)`) and converts the image to `np.uint8` type to be a valid grayscale image.
    8.  Applies a Gaussian filter to the resulting image to improve ArUco detection.
    9.  Returns the result.

### `get_aruco_file_path(str_path)`

*   **Purpose:** Constructs the output filename to save ArUco data, based on the MCD filename and platform.
*   **Arguments:**
    *   `str_path` (`str`): The full path of the MCD file.
*   **Return:**
    *   `str`: The full path of the ArUco output file.

*   **Operation:**
    1.  Determines the directory separator character (`/` or `\`) according to the platform (Linux or Windows).
    2.  Splits the MCD file path using the directory separator.
    3.  Extracts the video number from the filename (assuming the number is at the end of the filename, before the extension).
    4.  Constructs the ArUco output filename in format `arUcos-{num}.csv`, where `{num}` is the number extracted from the MCD filename.
    5.  Concatenates to get the full path.

### `main`

*   **Purpose:** Main program function. Coordinates video reading, ArUco detection, data processing, synchronization with MCD, and saving results.
*   **Command Line Arguments:**
    *   `mcd`: Path to the MCD CSV file.
    *   `vid`: Path to the video file.
*   **Global Variables/Constants:**
    *   `debug_mode` (`bool`): Activates/deactivates debug mode (prints additional information and shows processed image). Set to `False`.
    *   `aruco_dict`: Predefined ArUco dictionary (`cv2.aruco.DICT_4X4_50`).
    *   `parameters`: `cv2.aruco.DetectorParameters` object with parameters for ArUco detection. Several parameters are configured, including:
        *   `polygonalApproxAccuracyRate`: Controls accuracy of polygonal approximation of marker contours.
        *   `minCornerDistanceRate`: Minimum distance between corners.
        *   `cornerRefinementMethod`: Corner refinement method (`cv2.aruco.CORNER_REFINE_APRILTAG` used).
        *   `errorCorrectionRate`: Error correction rate.
        *   `cornerRefinementMinAccuracy`: Minimum accuracy for corner refinement.
        *   `cornerRefinementWinSize`: Window size for corner refinement.
        *   `perspectiveRemoveIgnoredMarginPerCell`: Ignored margin around each marker cell during perspective removal.
        *   `minMarkerPerimeterRate`: Minimum marker perimeter rate relative to image size.
    *   `detector`: `cv2.aruco.ArucoDetector` object created with the dictionary and parameters.
    *   `markers_size` (`float`): Actual size of ArUco markers (in meters).
    *   `origin_size` (`float`): Actual size of origin marker (in meters).
    *   `arUcos_ids` (`dict`): Dictionary mapping ArUco IDs to descriptive names (e.g., 'hip', 'R_knee').
    *   `arUcos_ids_inv` (dict): Inverse dictionary of `arUcos_ids`.
    *   `origin_id` (`int`): ID of the ArUco to be used as coordinate system origin.
    *   `origin_Tmatrix` (`np.ndarray`): Origin ArUco transformation matrix. Initialized to `None`.
    *   `origin_3D`: Origin marker 3D points.
    *   `markers_3D`: Markers 3D points.
    *   `rows` (`list`): List to store data from each processed frame (as dictionary).
    *   `aruco_m` (`list`): List to store slopes of the line joining hip and knee, for synchronization.
    *   `aruco_t` (`np.ndarray`): NumPy array to store frame times where ArUcos are detected, for synchronization.
    *   `last_p` (`dict`): Dictionary to store last detected position of each ArUco. This is used to crop the image and improve detection in subsequent frames.
    *   `camera_matrix`: Camera matrix.
    *   `dist_coeffs`: Distortion coefficients.
    *   `frame_rate`: Video frame rate.
    *   `cap_fps`: Video frame rate, forced to float.
    *   `cap`: `cv2.VideoCapture` object to capture video.

*   **Main Flow:**
    1.  **Arguments:** Processes arguments.
    2.  **Initialization:**
        *   Gets video information (`media_info`) using `pymediainfo.MediaInfo.parse()`.
        *   Initializes ArUco detector.
        *   Defines marker and origin sizes.
        *   Creates arrays to store output data.
        *   Processes MCD file.
        *   Initializes `last_p`.
        *   Gets camera matrix and distortion coefficients.
        *   Gets video frame rate.
        *   Opens video file with `cv2.VideoCapture`.
    3.  **Main Loop (per frame):**
        *   Reads a video frame (`cap.read()`).
        *   Converts frame to grayscale (`cv2.cvtColor`).
        *   **Debug Mode:** If `debug_mode` is `True`, shows frame in a window.
        *   **Frame Cropping:** If ArUcos were detected in previous frames, crops image around last known ArUco position.
        *   **ArUco Detection:** Detects ArUcos in frame (or cropped region) using `detector.detectMarkers()`.
        *   **No Detection Handling:** If no ArUcos detected, skips to next frame.
        *   **Detection Retries:**
            *   Iterates over expected ArUco IDs.
            *   If an expected ArUco was not detected in first attempt and we have a previous position for it (`type(last_p[i]) != np.ndarray`), tries detecting it again in a smaller region around last known position.
            *    Inside retry loop, Wiener filter (`apply_wiener_filter`) is applied with different parameters (`kernel_size` and `noise_power`) to try improving detection in presence of motion blur. Several kernel sizes and noise levels are tested. Iterates over `kernel_size` from 17 to 23 in steps of 2, and over `noise_power` from 0.02 to 0.032 in steps of 0.002.
            *   If Wiener filter and redetection succeed, updates `last_p`, adds new data to `mk_ids` and `mk_corners`, and exits retry loops (`break`).
        *   **Pose Calculation:**
            *   If origin ArUco detected (`origin_id in mk_ids`):
                *   Calculates origin transformation matrix using `cv2.solvePnP()` and `get_transformation_matrix()`. `solvePnP()` estimates pose (position and orientation) of a 3D object from its 3D points and corresponding 2D image projections. Uses origin 3D points (`origin_3D`), detected origin corners (`origin_corners`), camera matrix, and distortion coefficients.
                *   If origin transformation matrix cannot be calculated (`solvePnP` fails), skips to next frame (`continue`).
            *   Iterates over detected ArUcos (`zip(mk_ids, mk_corners)`):
                *   If ArUco ID is not in expected IDs list (`id not in arUcos_ids.keys()`), ignores it (`continue`).
                *   Calculates transformation matrix for each ArUco using `cv2.solvePnP()` and `get_transformation_matrix()`. Uses `markers_3D`, ArUco corners (`corners[0]`), `camera_matrix`, and `dist_coeffs`.
                *   If `solvePnP()` fails, skips to next ArUco (`continue`).
                *   Calculates ArUco transformation matrix relative to origin using `inv(origin_Tmatrix) @ Tmatrix`. This transforms ArUco coordinates to origin coordinate system.
                *   Extracts position coordinates (`coors`) and orientation (as a quaternion, `orientation`) from relative transformation matrix. Position is extracted from last column of transformation matrix. Orientation is calculated from rotation matrix (top-left 3x3 submatrix) using `scipy.spatial.transform.Rotation.from_matrix().as_quat(scalar_first=True)`.
                *   Stores position and orientation in `row` dictionary, using keys like `'hip_position'`, `'R_knee_orientation'`, etc.
        *   **Data Storage:**
            *   Adds `row` dictionary (with current frame info, including time and detected ArUco position/orientation) to `rows` list.
            *   If hip (`hip`) and knee (`knee`) ArUcos corresponding to `measures["direction"]` were detected, calculates slope of line joining them (`(hip[0]-knee[0])/(hip[1]-knee[1])`) and stores it in `aruco_m` list. Also stores current frame time in `aruco_t`. These data will be used for synchronization.
            *   Increments frame counter (`frame_number`).
    4.  **Synchronization:**
        *   Calculates offset (`offset`) between video data (ArUcos) and MCD data (inertial sensor) using cross-correlation.
            *   Converts MCD time and angular velocity lists (`mcd_t`, `mcd_av`) to NumPy arrays.
            *   Smoothing MCD angular velocity data: Applies a moving average filter to `mcd_av` using `np.convolve(mcd_av, np.ones(w)/w, mode='valid')`. This reduces noise in MCD data and improves synchronization accuracy. `w` is filter window width (20 in this case). Times are also adjusted.
            *   Interpolation: Interpolates MCD angular velocity data (`mcd_av_f`) and ArUco slopes (`aruco_m_f`) to have same sampling frequency and be directly comparable. Linear interpolation (`scipy.interpolate.interp1d`) with extrapolation (`fill_value='extrapolate'`) is used.
            *   Defines a common time range (`t`) for both signals, spanning period where both signals have data.
            *   Calculates cross-correlation between the two interpolated signals (`np.correlate(mcd_av_int, aruco_m_int, mode='full')`). Cross-correlation measures similarity between two signals as a function of time lag between them. Argument `mode='full'` indicates full correlation should be calculated, including lags where signals do not fully overlap.
            *   Finds index of maximum correlation value (`idx_max = np.argmax(correlation)`). This index corresponds to lag maximizing similarity between signals.
            *   Calculates offset as difference between maximum index and ArUco signal length minus 1 (`offset = idx_max - (len(aruco_m_int)-1)`). This offset represents time lag between two signals, expressed in time units (milliseconds).
            *   Offset application: Applies offset to ArUco data times (`r['time'] += offset`) to synchronize them with MCD data.
    5. **Saving Results:**
        *   Defines column names (`csv_fields`) for output CSV file.
        *   Creates output CSV file using `init_csv_file()` function. Filename generated using `get_aruco_file_path()`.
        *   Writes data stored in `rows` list to CSV file using `csv_file.writerows()`. Each element of `rows` is a dictionary representing a frame, and `writerows()` writes each dictionary as a row in CSV.
        *   Closes CSV file (`file.close()`).
