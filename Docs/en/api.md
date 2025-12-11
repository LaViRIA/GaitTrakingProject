# Code Documentation

This module provides an interface to communicate with a device via serial port, unpacking specific data and commands.

## Functions

### `unpacket_data_to_dict(data)`

Unpackets sensor data into a dictionary.

**Arguments:**

*   `data`: Received packed data. `data[0]` is expected to be the message type, `data[1]` the sensor count, `data[2]` the timestamp, and the rest the sensor data.

**Returns:**

A dictionary with sensor information. The structure is as follows:
```python
{
    "type": message type,
    "timestamp": timestamp,
    "sensor_count": sensor count,
    "data": {
        "SENSOR_ID_HEX": {  # Sensor ID in hexadecimal format
            "quat": list with quaternion values,
            "accel": list with acceleration values,
            "gyro": list with gyroscope values
            },
        ... (data for each sensor)
        }
}
```
### `unpacket_bat_to_dict(data)`

Unpackets battery data into a dictionary.

**Arguments:**

*   `data`: Received packed battery data. `data[0]` is expected to be the message type, `data[1]` the sensor count, and the rest the battery data.

**Returns:**

A dictionary with battery information. The structure is as follows:
```python
{
    "type": message type,
    "sensor_count": sensor count,
    "data": {
        "BATTERY_ID_HEX": battery value, # Battery ID in hexadecimal format
        ... (data for each sensor)
        }
}
```

### `unpack(buffer: bytearray, FORMAT: str)`

Unpacks a byte buffer using a `struct` format.

**Arguments:**

*   `buffer`: The byte buffer to unpack.
*   `FORMAT`: The `struct` format to use (see `struct` documentation for more details).

**Returns:**

A tuple with the unpacked data, or -1 if the buffer is too short.

### `unpack_data(buffer: bytearray, FORMAT: str)`

Unpacks sensor data.

**Arguments:**

*   `buffer`: The byte buffer to unpack.
*   `FORMAT`: The `struct` format to use.

**Returns:**

A dictionary with sensor data, or -1 if unpacking fails.

### `unpack_bat(buffer: bytearray, FORMAT: str)`

Unpacks battery data.

**Arguments:**

*   `buffer`: The byte buffer to unpack.
*   `FORMAT`: The `struct` format to use.

**Returns:**

A dictionary with battery data, or -1 if unpacking fails.

## Class `McdAPI`

Class for communication with the MCD device.

### Attributes

*   `__port`: The serial port.
*   `__baudrate`: The baud rate.

### Methods

*   `set_port(self, port)`: Sets the serial port.
*   `set_baudrate(self, baudrate)`: Sets the baud rate.
*   `set_config(self, port, baudrate)`: Sets the serial port and baud rate.
*   `connect(self)`: Connects to the device. Returns `True` if connection is successful, `False` otherwise.
*   `close(self)`: Closes the connection with the device. Returns `True` if connection is closed successfully, `False` otherwise.
*   `is_open(self)`: Checks if the connection is open. Returns `True` if open, `False` otherwise.
*   `read(self)`: Reads all available data in the input buffer.
*   `in_waiting(self)`: Returns the number of bytes in the input buffer.
*   `sync_mode(self)`: Sends the command for synchronization mode.
*   `identify_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Sends the command to identify a sensor.
*   `read_start(self, addr: str = 'FFFFFFFFFFFF')`: Sends the command to start reading data from a sensor.
*   `read_stop(self, addr: str = 'FFFFFFFFFFFF')`: Sends the command to stop reading data from a sensor.
*   `reset_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Sends the command to reset a sensor.
*   `calibrate_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Sends the command to calibrate a sensor.

    > Note: Methods with parameter 'addr' will send the command to the sensor indicated by said parameter. If not indicated (default 'FFFFFFFFFFFF') or sending the default value, it will be sent to all sensors.
