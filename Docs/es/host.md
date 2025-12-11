#### Estructuras de Datos Definidas

1. **sensor_data**
   ```cpp
   struct sensor_data{
     Quaternion q;
     VectorInt16 a;
     VectorInt16 g;
   };
   ```
   - `q`: Cuaternión que representa la orientación.
   - `a`: Vector entero de 16 bits para los acelerómetros.
   - `g`: Vector entero de 16 bits para los girosocopos.

2. **peer_data**
   ```cpp
   struct peer_data {
     sensor_data data;
     uint8_t battery;
     TimerHandle_t disconnect_timer;
   };
   ```
   - `data`: Datos del sensor.
   - `battery`: Nivel de batería (0-100%).
   - `disconnect_timer`: Temporizador para manejar la desconexión inactiva.

#### Constantes Definidas

- `ALIVE_MSG_FREQUENCY`: Frecuencia (en ms) con la que se envían mensajes "still alive" (default: 1000 ms).
- `INACTIVE_TIMEOUT`: Tiempo de espera (en ms) antes de eliminar un dispositivo inactivo (default: 3000 ms).
- `SYNC_TIME`: Tiempo de sincronización inicial (default: 10000 ms).
- `SYNC_MSG_FREQUENCY`: Frecuencia (en ms) con la que se envían mensajes de sincronización (default: 1000 ms).
- `BATTERY_MSG_FREQUENCY`: Frecuencia (en ms) con la que se envían actualizaciones de batería (default: 1000 ms).
- `DATA_MSG_FREQUENCY`: Frecuencia (en Hz) con la que se envían datos de sensores (default: 10 Hz).

#### Variables Globales

- `local_mac[6]`: MAC address del dispositivo actual.
- `disable_sync_timer`: Temporizador con el que se deshabilita el modo sincronización.
- `send_sync_timer`: Temporizador con el que se envian mensajes de sincronización.
- `alive_timer`: Temporizador con el que se envian mensajes 'alive'.
- `send_data_timer`: Temporizador con el que se envian paquetes de datos por serial.
- `send_battery_timer`: Temporizador con el que se envian paquetes de información de bateria por serial.
- `reading`: Bandera para controlar si se están enviando datos.

#### Funciones Principales

1. **OnDataRecv**
   - Callback que se ejecuta cuando se recibe un paquete de datos por ESPNOW.
   - Procesa diferentes tipos de mensajes según el primer byte del paquete:
     - **Case 1**: Actualiza los datos del sensor.
     - **Case 2**: Actualiza el nivel de batería.
     - **Case 4**: Maneja solicitudes de conexión.
     - **Case 5**: Desconecta un dispositivo.
     - **Case 6**: Resetea el temporizador de supervivencia 'alive'.
   > ** Nota: revisar paquetes.md

2. **setup()**
   - Configura la tarjeta ESP32 como estación (STA).
   - Inicializa ESP-NOW y registra el callback `OnDataRecv`.
   - Configura los diferentes temporizadores para enviar mensajes periódicos.

3. **loop()**
   - Función vacía ya que las tareas se manejan a través de callbacks y FreeRTOS.

4. **serialEvent()**
   - Maneja eventos de datos entrantes por serial.
   - Responde a comandos recibidos por puerto serie, como iniciar o detener el envío de datos.
