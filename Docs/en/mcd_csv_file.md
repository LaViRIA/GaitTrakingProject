## Generated CSV File Format

The application generates CSV (Comma Separated Values) files when a data capture is performed. These files contain raw sensor data, along with time information and metadata.

### General Structure

The CSV file has the following general structure:

1.  **Comments (optional):** The file may start with one or more comment lines, preceded by the `#` character. These comments include information about the anthropometric measurements provided by the user (height, age, femur length, tibia length, and hip width) and the direction of movement ('R' in this case, but could be extended in the future).

2.  **Header:** A line that defines the names of the columns (fields) of the CSV file. field names are separated by the specified delimiter (semicolon `;` by default).

3.  **Data:** Each subsequent row represents a sample of sensor data at a specific instant in time. The values in each row are separated by the delimiter.

### Fields (Columns)

The CSV file contains the following fields, in the specified order:

*   **`time`:** Timestamp relative to the start of the capture, in milliseconds. It is calculated by subtracting the timestamp of the first sample received (`reference_time`) from the timestamp of the current sample.

*   **`quaternion_RF`, `quaternion_RT`, `quaternion_LF`, `quaternion_LT`:** Orientation quaternions for each of the sensors. Each of these fields contains four floating-point values separated by commas, representing the *w*, *x*, *y*, and *z* components of the quaternion. The abbreviations correspond to:
    *   `RF`: Right Femur
    *   `RT`: Right Tibia
    *   `LF`: Left Femur
    *   `LT`: Left Tibia

*   **`acceleration_RF`, `acceleration_RT`, `acceleration_LF`, `acceleration_LT`:** Acceleration values for each of the sensors. Each of these fields contains three floating-point values separated by commas, representing accelerations on the *x*, *y*, and *z* axes (in m/s²). Values are scaled by dividing the raw sensor readings by 16384 and multiplying by 9.81.

*   **`gyro_RF`, `gyro_RT`, `gyro_LF`, `gyro_LT`:** Gyroscope values for each of the sensors. Each of these fields contains three floating-point values separated by commas, representing angular velocities on the *x*, *y*, and *z* axes.

**Important Notes:**

*   **Conditional Fields:** Fields related to a specific bone (`RF`, `RT`, `LF`, `LT`) will only be included in the CSV file if the corresponding sensor is configured and assigned to that bone in the application interface. If a sensor is not assigned, its data will not be recorded in the file. This is controlled through the inverted `config` dictionary (from bone to sensor address).
*   **Delimiter:** The default delimiter is the semicolon (`;`), but this is passed as a parameter to the `init_csv_file` function and could be modified.
*   **Units:**
    *   Time: milliseconds (ms)
    *   Acceleration: raw sensor data by default (`int16_t`) range of ±2g
    *   Quaternions: dimensionless (normalized values)
    *   Gyroscope: raw sensor data by default (`int16_t`) range of ±250 °/s
* **Filename**: The filename is composed of the prefix "mcd-", followed by an incremental numeric index (`i`) to avoid overwrites. The full format is `mcd-{i}.csv`, and it is saved within a folder structure created from the ID entered in the application.
* **Raw Data**: Gyroscope and acceleration data are saved without any additional filtering, directly as received from the API.

### Example
```
# ;height: 1.75;age: 30.0;femur_length: 0.45;tibia_length: 0.4;hip_width: 0.3;direction: R
time;quaternion_RF;acceleration_RF;gyro_RF;quaternion_RT;acceleration_RT;gyro_RT;quaternion_LF;acceleration_LF;gyro_LF;quaternion_LT;acceleration_LT;gyro_LT
0;0.5914306640625,-0.0322265625,-0.803955078125,-0.05303955078125;15636,686,-4848;8,-1,-6;0.68902587890625,0.07891845703125,-0.71533203125,0.08538818359375;16526,-232,-760;4,-5,2;0.62310791015625,-0.0042724609375,-0.781982421875,-0.01312255859375;16116,322,-3698;8,12,-2;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16392,-902,-324;3,1,-1
10;0.5914306640625,-0.03216552734375,-0.803955078125,-0.05303955078125;15872,750,-4976;7,-2,-5;0.68902587890625,0.07891845703125,-0.71533203125,0.08538818359375;16400,-216,-624;6,-6,0;0.62322998046875,-0.0042724609375,-0.78192138671875,-0.01318359375;16010,294,-3642;-3,14,-3;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16460,-836,-566;2,0,-3
20;0.5914306640625,-0.0321044921875,-0.803955078125,-0.052978515625;15878,760,-4894;7,1,-6;0.68896484375,0.0789794921875,-0.71539306640625,0.08544921875;16502,-272,-720;10,-4,2;0.62322998046875,-0.0042724609375,-0.78192138671875,-0.01318359375;16116,298,-3526;-8,11,-5;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16426,-908,-354;-1,-2,-3
...
```
In this simplified example, the first three rows of data are shown.
