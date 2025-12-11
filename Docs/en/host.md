#### Defined Data Structures

1. **sensor_data**
   ```cpp
   struct sensor_data{
     Quaternion q;
     VectorInt16 a;
     VectorInt16 g;
   };
   ```
   - `q`: Quaternion representing orientation.
   - `a`: 16-bit integer vector for accelerometers.
   - `g`: 16-bit integer vector for gyroscopes.

2. **peer_data**
   ```cpp
   struct peer_data {
     sensor_data data;
     uint8_t battery;
     TimerHandle_t disconnect_timer;
   };
   ```
   - `data`: Sensor data.
   - `battery`: Battery level (0-100%).
   - `disconnect_timer`: Timer to handle inactive disconnection.

#### Defined Constants

- `ALIVE_MSG_FREQUENCY`: Frequency (in ms) at which "still alive" messages are sent (default: 1000 ms).
- `INACTIVE_TIMEOUT`: Wait time (in ms) before removing an inactive device (default: 3000 ms).
- `SYNC_TIME`: Initial synchronization time (default: 10000 ms).
- `SYNC_MSG_FREQUENCY`: Frequency (in ms) at which synchronization messages are sent (default: 1000 ms).
- `BATTERY_MSG_FREQUENCY`: Frequency (in ms) at which battery updates are sent (default: 1000 ms).
- `DATA_MSG_FREQUENCY`: Frequency (in Hz) at which sensor data is sent (default: 10 Hz).

#### Global Variables

- `local_mac[6]`: MAC address of the current device.
- `disable_sync_timer`: Timer used to disable synchronization mode.
- `send_sync_timer`: Timer used to send synchronization messages.
- `alive_timer`: Timer used to send 'alive' messages.
- `send_data_timer`: Timer used to send data packets via serial.
- `send_battery_timer`: Timer used to send battery information packets via serial.
- `reading`: Flag to control if data is being sent.

#### Main Functions

1. **OnDataRecv**
   - Callback executed when a data packet is received via ESPNOW.
   - Processes different message types according to the first byte of the packet:
     - **Case 1**: Updates sensor data.
     - **Case 2**: Updates battery level.
     - **Case 4**: Handles connection requests.
     - **Case 5**: Disconnects a device.
     - **Case 6**: Resets the 'alive' survival timer.
   > ** Note: check packages.md

2. **setup()**
   - Configures the ESP32 board as a station (STA).
   - Initializes ESP-NOW and registers the `OnDataRecv` callback.
   - Configures the different timers to send periodic messages.

3. **loop()**
   - Empty function since tasks are handled via callbacks and FreeRTOS.

4. **serialEvent()**
   - Handles incoming serial data events.
   - Responds to commands received via serial port, such as starting or stopping data transmission.
