## Package Types
- **1 | Data** 
    - Only sent from clients to the host and from the host to the Serial port.
    
    - **Client-Host Format**
        - **Sensor Data Format**
            | **Position** | **Size in bytes**      | **Description**                     | **Type**   |
            |-------------:|-----------------------:|:------------------------------------|:-----------|
            | 0            | 16                     | Quaternions                         | 4 * float  |
            | 16           | 6                      | Acceleration                        | 3 * int16_t|
            | 22           | 6                      | Gyroscope                           | 3 * int16_t|
            > *Note: referred to later as "sensor"*

        | **Position** | **Size in bytes**      | **Description**                     | **Type**   |
        |-------------:|-----------------------:|:------------------------------------|:-----------|
        | 0            | 1                      | Package Type                        | 1 * uint8_t|
        | 1            | 28                     | Sensor Data                         | 1 * sensor |
    - **Host-Serial Format**
        - **Sensor Data Format**
            | **Position** | **Size in bytes**      | **Description**                     | **Type**   |
            |-------------:|-----------------------:|:------------------------------------|:-----------|
            | 0            | 6                      | MAC Address                         | 6 * uint8_t|
            | 6            | 16                     | Quaternions                         | 4 * float  |
            | 22           | 6                      | Acceleration                        | 3 * int16_t|
            | 28           | 6                      | Gyroscope                           | 3 * int16_t|
            > *Note: referred to later as "sensor"*

        | **Position**      | **Size in bytes**        | **Description**                     | **Type**             |
        |------------------:|-------------------------:|:------------------------------------|:---------------------|
        | 0                 | 1                        | Package Type                        | 1 * uint8_t          |
        | 1                 | 1                        | Sensor Count                        | 1 * uint8_t          |
        | 2                 | 8                        | Timestamp                           | 1 * uint64_t         |
        | 10+(SensorNo*34)  | 34 * Sensor Count        | Sensor Data                         | Sensor Count * sensor|
- **2 | Battery** 
    - Only sent from clients to the host and from the host to the Serial port.
    - **Client-Host Format**
        | **Position** | **Size in bytes**      | **Description**                     | **Type**   |
        |-------------:|-----------------------:|:------------------------------------|:-----------|
        | 0            | 1                      | Package Type                        | 1 * uint8_t|
        | 1            | 1                      | Charge Level                        | 1 * uint8_t|
    - **Host-Serial Format**
         - **Sensor Data Format**

            | **Position** | **Size in bytes**      | **Description**                     | **Type**   |
            |-------------:|-----------------------:|:------------------------------------|:-----------|
            | 0            | 6                      | MAC Address                         | 6 * uint8_t|
            | 6            | 1                      | Charge Level                        | 1 * uint8_t|
            > *Note: referred to later as "sensor"*

        | **Position**      | **Size in bytes**      | **Description**                     | **Type**             |
        |------------------:|-----------------------:|:------------------------------------|:---------------------|
        | 0                 | 1                      | Package Type                        | 1 * uint8_t          |
        | 1                 | 1                      | Sensor Count                        | 1 * uint8_t          |
        | 2+(SensorNo*7)    | 7 * Sensor Count       | Sensor Charge                       | Sensor Count * sensor|
- **3 | Synchronization** 
    - Indicated from the serial port, initializes synchronization mode on the host.
    - The host will send the byte "0x03" to the broadcast address "0xffffffffffff", allowing any ESPNOW device to receive the message.
    - Activating synchronization mode on a client allows it to process messages from unknown addresses.
    - If it receives a synchronization type message, it will register the source MAC address as the host address and start trying to connect.
- **4 | Connection** 
    - Allows connection between the client and the host.
    - The client sends messages with byte "0x04" to the address stored in EEPROM memory as host.
    - If the host receives the packet, it returns a '0x04' byte followed by another byte (0x00 | 0x01) indicating if data reading is active or not.
- **5 | Disconnection** 
    - Allows a client to indicate to the host that it is going to disconnect.
    - This happens when a client is turned off or runs out of battery.
- **6 | 'Alive' Survival** 
    - Keeps the connection alive between the host and the client.
    - Consists of a '0x06' byte.
    - By default, one is sent every second.
    - If the host stops receiving the message from a client for 3 seconds, it will disconnect and remove the client.
    - If the client stops receiving the message from the host for 3 seconds, it will indicate that it has disconnected and start sending messages trying to connect.
    - The host will send messages in multicast, allowing messages to be sent to all clients at once, but without all devices on the ESPNOW network receiving it.
- **7 | Identification** 
    - Allows identifying one or multiple devices, changing the color of the integrated LED.
    - From the serial, the package type ('0x07') is indicated followed by the client address to identify.
    - Although the broadcast address '0xffffffffffff' can be sent from the serial, the host will filter this and send it via multicast only to connected clients.
- **8 | Start read** 
    - Allows starting data reading from clients.
    - From the serial, the package type ('0x08') is indicated followed by the client address to start reading.
    - Using address "0xffffffffffff" can start reading on all connected clients.
- **9 | Stop read** 
    - Allows stopping data reading from clients.
    - From the serial, the package type ('0x09') is indicated followed by the client address to stop.
    - Using address "0xffffffffffff" can stop reading on all connected clients.
- **10 | Reset sensor** 
    - Allows resetting sensors of one or multiple clients.
    - From the serial, the package type ('0x0A') is indicated followed by the client address with the sensor to reset.
    - Using address "0xffffffffffff" can reset the sensor of all clients at once.
- **11 | Calibration** 
    - Allows starting the calibration routine of sensors of one or multiple clients.
    - From the serial, the package type ('0x0B') is indicated followed by the client address with the sensor to calibrate.
    - Using address "0xffffffffffff" can calibrate the sensor of all clients at once.
