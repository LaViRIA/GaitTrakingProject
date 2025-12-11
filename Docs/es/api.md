# Documentación del Código

Este módulo proporciona una interfaz para comunicarse con un dispositivo a través del puerto serial, desempaquetando datos y comandos específicos.

## Funciones

### `unpacket_data_to_dict(data)`

Desempaqueta datos de sensores en un diccionario.

**Argumentos:**

*   `data`: Datos empaquetados recibidos. Se espera que `data[0]` sea el tipo de mensaje, `data[1]` la cantidad de sensores, `data[2]` la marca de tiempo, y el resto los datos de los sensores.

**Retorna:**

Un diccionario con la información de los sensores. La estructura es la siguiente:
```python
{
    "type": tipo de mensaje,
    "timestamp": marca de tiempo,
    "sensor_count": cantidad de sensores,
    "data": {
        "ID_SENSOR_HEX": {  # ID del sensor en formato hexadecimal
            "quat": lista con los valores del cuaternión,
            "accel": lista con los valores de aceleración,
            "gyro": lista con los valores de giroscopio
            },
        ... (datos para cada sensor)
        }
}
```
### `unpacket_bat_to_dict(data)`

Desempaqueta datos de batería en un diccionario.

**Argumentos:**

*   `data`: Datos empaquetados de la batería recibidos. Se espera que `data[0]` sea el tipo de mensaje, `data[1]` la cantidad de sensores, y el resto los datos de las baterías.

**Retorna:**

Un diccionario con la información de la batería. La estructura es la siguiente:
```python
{
    "type": tipo de mensaje,
    "sensor_count": cantidad de sensores,
    "data": {
        "ID_BATERIA_HEX": valor de la bateria, # ID de la batería en formato hexadecimal
        ... (datos para cada sensor)
        }
}
```

### `unpack(buffer: bytearray, FORMAT: str)`

Desempaqueta un buffer de bytes usando un formato de `struct`.

**Argumentos:**

*   `buffer`: El buffer de bytes a desempaquetar.
*   `FORMAT`: El formato de `struct` a utilizar (ver documentación de `struct` para más detalles).

**Retorna:**

Una tupla con los datos desempaquetados, o -1 si el buffer es demasiado corto.

### `unpack_data(buffer: bytearray, FORMAT: str)`

Desempaqueta datos de sensores.

**Argumentos:**

*   `buffer`: El buffer de bytes a desempaquetar.
*   `FORMAT`: El formato de `struct` a utilizar.

**Retorna:**

Un diccionario con los datos de los sensores, o -1 si falla el desempaquetado.

### `unpack_bat(buffer: bytearray, FORMAT: str)`

Desempaqueta datos de batería.

**Argumentos:**

*   `buffer`: El buffer de bytes a desempaquetar.
*   `FORMAT`: El formato de `struct` a utilizar.

**Retorna:**

Un diccionario con los datos de la batería, o -1 si falla el desempaquetado.

## Clase `McdAPI`

Clase para la comunicación con el dispositivo MCD.

### Atributos

*   `__port`: El puerto serial.
*   `__baudrate`: La velocidad de baudios.

### Métodos

*   `set_port(self, port)`: Establece el puerto serial.
*   `set_baudrate(self, baudrate)`: Establece la velocidad de baudios.
*   `set_config(self, port, baudrate)`: Establece el puerto serial y la velocidad de baudios.
*   `connect(self)`: Conecta con el dispositivo. Retorna `True` si la conexión es exitosa, `False` en caso contrario.
*   `close(self)`: Cierra la conexión con el dispositivo. Retorna `True` si la conexión se cierra exitosamente, `False` en caso contrario.
*   `is_open(self)`: Verifica si la conexión está abierta. Retorna `True` si está abierta, `False` en caso contrario.
*   `read(self)`: Lee todos los datos disponibles en el buffer de entrada.
*   `in_waiting(self)`: Retorna la cantidad de bytes en el buffer de entrada.
*   `sync_mode(self)`: Envía el comando para el modo de sincronización.
*   `identify_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Envía el comando para identificar un sensor.
*   `read_start(self, addr: str = 'FFFFFFFFFFFF')`: Envía el comando para iniciar la lectura de datos de un sensor.
*   `read_stop(self, addr: str = 'FFFFFFFFFFFF')`: Envía el comando para detener la lectura de datos de un sensor.
*   `reset_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Envía el comando para resetear un sensor.
*   `calibrate_sensor(self, addr: str = 'FFFFFFFFFFFF')`: Envía el comando para calibrar un sensor.

    > Nota: Los metodos con parametro 'addr' enviaran el comando hasta el sensor indicado por dicho parametro, en caso de no indicarlo (por defecto 'FFFFFFFFFFFF') o enviar el valor por defecto, se enviara a todos los sensores