## Output CSV File Documentation

The script generates a CSV file (with `.csv` extension) containing the tracking data of the ArUco markers, synchronized with the inertial sensor data (MCD). The file uses the semicolon (`;`) as a delimiter.

### General Structure
The CSV file has the following structure:

*   **Header:** The first row contains the column names (fields).
*   **Data:** Each subsequent row represents a time instant (a processed video frame) and contains the position and orientation data of the ArUco markers detected in that frame.
*   **No missing data:** It is intended that there is information for every instant. In case there is no data for a column in an instant, an empty string is inserted.
### Columns (Fields)
The CSV file contains the following columns:

1.  **`time:`** Time in milliseconds, synchronized with the MCD data. It is the time elapsed since the start of the video, adjusted by the offset calculated via cross-correlation.

2.  **`hip_position:`** 3D position of the ArUco marker identified as "hip" in the origin marker's coordinate system. The coordinates are expressed as a text string with x, y, z values separated by commas (e.g., "0.1,0.2,-0.05"). Units are in meters.

3.  **`hip_orientation:`** Orientation of the "hip" ArUco marker expressed as a quaternion. The w, x, y, z values of the quaternion are stored as a text string separated by commas (e.g., "0.707,0,0.707,0"). The quaternion represents the rotation from the coordinate system of the origin marker to the coordinate system of the "hip" marker.

4.  **`R_knee_position:`** 3D position of the ArUco marker identified as "R_knee" (right knee) in the origin marker's coordinate system. Format and units same as `hip_position`.

5.  **`R_knee_orientation:`** Orientation of the "R_knee" ArUco marker expressed as a quaternion. Format same as `hip_orientation`.

6.  **`L_knee_position:`** 3D position of the ArUco marker identified as "L_knee" (left knee) in the origin marker's coordinate system. Format and units same as `hip_position`.

7.  **`L_knee_orientation:`** Orientation of the "L_knee" ArUco marker expressed as a quaternion. Format same as `hip_orientation`.

8.  **`R_ankle_position:`** 3D position of the ArUco marker identified as "R_ankle" (right ankle) in the origin marker's coordinate system. Format and units same as `hip_position`.

9.  **`R_ankle_orientation:`** Orientation of the "R_ankle" ArUco marker expressed as a quaternion. Format same as `hip_orientation`.

10. **`L_ankle_position:`** 3D position of the ArUco marker identified as "L_ankle" (left ankle) in the origin marker's coordinate system. Format and units same as `hip_position`.

11. **`L_ankle_orientation:`** Orientation of the "L_ankle" ArUco marker expressed as a quaternion. Format same as `hip_orientation`.

### Important Notes:

*   **Coordinate System:** All positions are expressed in the coordinate system of the origin ArUco marker (identified by `origin_id` in the code). This means that the position of the origin marker will always be (0, 0, 0).
*   **Units:** Positions are expressed in meters.
*   **Quaternions:** Orientations are expressed as unit quaternions. A quaternion is a way to represent a rotation in 3D.
*   **Synchronization:** Times (`time`) are synchronized with the data from the MCD file, allowing relation of inertial sensor measurements with the position and orientation of ArUco markers.
*   **Filename:** The filename is based on the MCD filename, following the pattern `arUcos-{num}.csv`, where `{num}` is a number extracted from the MCD filename.
*   **Missing Values:** If an ArUco is not detected in a particular frame, its position and orientation fields will be empty.

### Example
```
time;hip_position;hip_orientation;R_knee_position;R_knee_orientation;L_knee_position;L_knee_orientation;R_ankle_position;R_ankle_orientation;L_ankle_position;L_ankle_orientation
-915;1.0812701974126873,0.19553333992405364,1.7730267768679695;-0.03579755363431741,-0.3027704605265213,-0.07767882216617086,0.9492178801377653;;;1.1903190364076657,-0.16535499196021697,1.5738097880703292;-0.04663701743823387,-0.5796669644370976,-0.03964008542739295,0.8125514522613662;;;1.3236439703535121,-0.5518511445951062,1.3846893687834791;0.03225431442603979,0.7038767886366579,0.14632608847877795,-0.6943383911681338
-882;1.4069629577923255,0.29248264061831386,1.6587844085733763;-0.058889696552360926,-0.21736234964294096,-0.05050321232586272,0.9730031028431717;;;1.4171593853951752,-0.08855974405245881,1.5316496108576465;0.23590359826200835,-0.162307849085637,0.11842638852687046,0.9507790726308156;;;1.555216435406157,-0.45256777596924824,1.277258768385952;0.18743136721138356,-0.004164664486858101,-0.022024020895057173,0.9820219349172387
-849;1.4118625766399564,0.29080435721728504,1.65334942309934;-0.05518302337878983,-0.2166081215371079,-0.050789377709053514,0.9733736151791194;;;1.4222101738202126,-0.09045581507646427,1.5252423374779687;0.23729410460036848,-0.1644767270161073,0.11742082256555682,0.9501848581302516;;;1.5579534378611228,-0.45448027178301664,1.2729650765155127;0.19158570608621664,-0.0019353296750931896,-0.01941407573348632,0.9812819499949589
```
In this simplified example, the first three rows of data are shown.
