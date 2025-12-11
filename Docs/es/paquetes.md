## Tipos de paquetes
- **1 | Datos** 
    - Solo se envian desde los clientes al host y del host al puerto Serial.
    
    - **Formato cliente-host**
        - **Formato datos del sensor**
            | **Posición** | **Tamaño en bytes**      | **Descripción**                     | **Tipo**   |
            |-------------:|-------------------------:|:------------------------------------|:-----------|
            | 0            | 16                       | Quaterniones                        | 4 * float  |
            | 16           | 6                        | Aceleración                         | 3 * int16_t|
            | 22           | 6                        | Giroscopio                          | 3 * int16_t|
            > *Nota: llamado mas adelante como "sensor"*

        | **Posición** | **Tamaño en bytes**      | **Descripción**                     | **Tipo**   |
        |-------------:|-------------------------:|:------------------------------------|:-----------|
        | 0            | 1                        | Tipo de paquete                     | 1 * uint8_t|
        | 1            | 28                       | Datos del sensor                    | 1 * sensor |
    - **Formato host-serial**
        - **Formato datos del sensor**
            | **Posición** | **Tamaño en bytes**      | **Descripción**                     | **Tipo**   |
            |-------------:|-------------------------:|:------------------------------------|:-----------|
            | 0            | 6                        | Direccion MAC                       | 6 * uint8_t|
            | 6            | 16                       | Quaterniones                        | 4 * float  |
            | 22           | 6                        | Aceleración                         | 3 * int16_t|
            | 28           | 6                        | Giroscopio                          | 3 * int16_t|
            > *Nota: llamado mas adelante como "sensor"*

        | **Posición**      | **Tamaño en bytes**      | **Descripción**                     | **Tipo**             |
        |------------------:|-------------------------:|:------------------------------------|:---------------------|
        | 0                 | 1                        | Tipo de paquete                     | 1 * uint8_t          |
        | 1                 | 1                        | Cantidad de sensores                | 1 * uint8_t          |
        | 2                 | 8                        | Marca de Tiempo                     | 1 * uint64_t         |
        | 10+(No.Sensor*34) | 34 * No. sensores        | Datos de los sensores               | No. Sensores * sensor|
- **2 | Bateria** 
    - Solo se envian desde los clientes al host y del host al puerto Serial.
    - **Formato cliente-host**
        | **Posición** | **Tamaño en bytes**      | **Descripción**                     | **Tipo**   |
        |-------------:|-------------------------:|:------------------------------------|:-----------|
        | 0            | 1                        | Tipo de paquete                     | 1 * uint8_t|
        | 1            | 1                        | Nivel de carga                      | 1 * uint8_t|
    - **Formato host-serial**
         - **Formato datos del sensor**

            | **Posición** | **Tamaño en bytes**      | **Descripción**                     | **Tipo**   |
            |-------------:|-------------------------:|:------------------------------------|:-----------|
            | 0            | 6                        | Direccion MAC                       | 6 * uint8_t|
            | 6            | 1                        | Nivel de carga                      | 1 * uint8_t|
            > *Nota: llamado mas adelante como "sensor"*

        | **Posición**      | **Tamaño en bytes**      | **Descripción**                     | **Tipo**             |
        |------------------:|-------------------------:|:------------------------------------|:---------------------|
        | 0                 | 1                        | Tipo de paquete                     | 1 * uint8_t          |
        | 1                 | 1                        | Cantidad de sensores                | 1 * uint8_t          |
        | 2+(No.sensor*7)   | 7 * No. sensores         | Carga de los sensores               | No. Sensores * sensor|
- **3 | Sincronizacion** 
    - Indicado desde el puerto serial, inicializa el modo sincronizacion en el host.
    - El host enviara el byte "0x03" a la direccion broadcast "0xffffffffffff", permitiendo que cualquier dispositivo ESPNOW pueda recibir el mensaje.
    - El activar el modo sincronizacion en un cliente, este permite procesar mensajes de direcciones desconocidas.
    - Si recibe un mensaje de tipo sincronizacion registrara la direccion mac de la fuente como la direccion del host y empezara a tratar de conectarse.
- **4 | Conexión** 
    - Permite la conexion entre el cliente y el host.
    - El cliente envia mensajes con el byte "0x04" a la direccion que tiene almacenada en la memoria EEPROM como host.
    - Si el host recibe el paquete, este devuelve un byte '0x04' seguido de otro byte (0x00 | 0x01) indicando si se esta haciendo lectura de datos o no.
- **5 | Desconexión** 
    - Permite a un cliente indicarle al host que se va a desconectar.
    - Esto se da cuando se apaga un cliente o cuando este se queda sin bateria.
- **6 | Supervivencia 'alive'** 
    - Permite mantener viva la conexion entre el host y el cliente.
    - Consta de un byte '0x06'.
    - Por defecto se envia uno cada segundo.
    - Si el host deja de recibir el mensaje de un cliente durante 3 segundos, desconectara y eliminara al cliente.
    - Si el cliente deja de recibir el mensaje del host durante 3 segundos, indicara que se ha desconectado y empezara a enviar mensajes intentando conectarse.
    - El host enviara los mensajes de forma multicast, permitiendo asi enviar mensajes a todos los clientes a la vez, pero sin que todos los dispositivos de la red ESPNOW lo reciban.
- **7 | Identificación** 
    - Permite identificar uno o varios dispositivos, cambiando el color del led integrado.
    - Desde el serial se indica el tipo de paquete ('0x07') seguido de la direccion de cliente a identificar.
    - Si bien desde el serial se puede enviar a la direccion broadcast '0xffffffffffff', el host filtrara esto y lo enviara de manera multicast solo a los clientes conectados.
- **8 | Start read** 
    - Permite iniciar con la lectura de datos de los clientes.
    - Desde el serial se indica el tipo de paquete ('0x08') seguido de la direccion del cliente que empezara a leer.
    - Indicando la direccion "0xffffffffffff" puede iniciar la lectura en todos los clientes conectados.
- **9 | Stop read** 
    - Permite detener con la lectura de datos de los clientes.
    - Desde el serial se indica el tipo de paquete ('0x09') seguido de la direccion del cliente a detener.
    - Indicando la direccion "0xffffffffffff" puede detener la lectura en todos los clientes conectados.
- **10 | Reset sensor** 
    - Permite resetear los sensores de uno o varios clientes.
    - Desde el serial se indica el tipo de paquete ('0x0A') seguido de la direccion del cliente con el sensor a resetear.
    - Indicando la direccion "0xffffffffffff" puede resetear el sensor de todos los cliente a la vez.
- **11 | Calibracion** 
    - Permite iniciar la rutina de calibracion de los sensores de uno o varios clientes.
    - Desde el serial se indica el tipo de paquete ('0x0B') seguido de la direccion del cliente con el sensor a calibrar.
    - Indicando la direccion "0xffffffffffff" puede calibrar el sensor de todos los clientes a la vez.