## Formato del Archivo CSV Generado

La aplicación genera archivos CSV (valores separados por comas) cuando se realiza una captura de datos. Estos archivos contienen los datos brutos de los sensores, junto con información de tiempo y metadatos.

### Estructura General

El archivo CSV tiene la siguiente estructura general:

1.  **Comentarios (opcional):** El archivo puede comenzar con una o más líneas de comentarios, precedidas por el carácter `#`. Estos comentarios incluyen información sobre las medidas antropométricas proporcionadas por el usuario (altura, edad, longitud de fémur, longitud de tibia y ancho de cadera) y la dirección de movimiento ('R' en este caso, pero podría ser extendido en el futuro).

2.  **Cabecera:** Una línea que define los nombres de las columnas (campos) del archivo CSV. Los nombres de los campos están separados por el delimitador especificado (punto y coma `;` por defecto).

3.  **Datos:** Cada fila subsiguiente representa una muestra de datos de los sensores en un instante de tiempo específico. Los valores en cada fila están separados por el delimitador.

### Campos (Columnas)

El archivo CSV contiene los siguientes campos, en el orden especificado:

*   **`time`:**  Marca de tiempo (timestamp) relativa al inicio de la captura, en milisegundos.  Se calcula restando el timestamp de la primera muestra recibida (`reference_time`) del timestamp de la muestra actual.

*   **`quaternion_RF`, `quaternion_RT`, `quaternion_LF`, `quaternion_LT`:**  Cuaterniones de orientación para cada uno de los sensores.  Cada uno de estos campos contiene cuatro valores de punto flotante separados por comas, que representan los componentes *w*, *x*, *y* y *z* del cuaternión.  Las abreviaturas corresponden a:
    *   `RF`: Right Femur (Fémur Derecho)
    *   `RT`: Right Tibia (Tibia Derecha)
    *   `LF`: Left Femur (Fémur Izquierdo)
    *   `LT`: Left Tibia (Tibia Izquierda)

*   **`acceleration_RF`, `acceleration_RT`, `acceleration_LF`, `acceleration_LT`:**  Valores de aceleración para cada uno de los sensores. Cada uno de estos campos contiene tres valores de punto flotante separados por comas, que representan las aceleraciones en los ejes *x*, *y* y *z* (en m/s²). Los valores se escalan dividiendo las lecturas en bruto del sensor por 16384 y multiplicando por 9.81.

*   **`gyro_RF`, `gyro_RT`, `gyro_LF`, `gyro_LT`:**  Valores del giroscopio para cada uno de los sensores. Cada uno de estos campos contiene tres valores de punto flotante separados por comas, que representan las velocidades angulares en los ejes *x*, *y* y *z*.

**Notas Importantes:**

*   **Campos Condicionales:** Los campos relacionados con un hueso específico (`RF`, `RT`, `LF`, `LT`) solo se incluirán en el archivo CSV si el sensor correspondiente está configurado y asignado a ese hueso en la interfaz de la aplicación.  Si un sensor no está asignado, sus datos no se registrarán en el archivo. Esto se controla a traves del diccionario `config` invertido (de hueso a dirección del sensor).
*   **Delimitador:** El delimitador por defecto es el punto y coma (`;`), pero esto se pasa como parámetro a la funcion `init_csv_file` y podria ser modificado.
*   **Unidades:**
    *   Tiempo: milisegundos (ms)
    *   Aceleración: datos brutos de los sensores por defecto (`int16_t`) rango de ±2g
    *   Cuaterniones: adimensionales (valores normalizados) 
    *   Giroscopio: datos brutos de los sensores por defecto (`int16_t`) rango de ±250 °/s
* **Nombre del Archivo**: El nombre del archivo se compone del prefijo "mcd-", seguido de un índice numérico incremental (`i`) para evitar sobrescrituras. El formato completo es `mcd-{i}.csv`, y se guarda dentro de una estructura de carpetas creada a partir del ID ingresado en la aplicacion.
* **Datos en Bruto**: Los datos del giroscopio y la aceleracion se guardan sin ningun tipo de filtrado adicional, directamente de lo que se recibe desde la API.

### Ejemplo
```
# ;height: 1.75;age: 30.0;femur_length: 0.45;tibia_length: 0.4;hip_width: 0.3;direction: R
time;quaternion_RF;acceleration_RF;gyro_RF;quaternion_RT;acceleration_RT;gyro_RT;quaternion_LF;acceleration_LF;gyro_LF;quaternion_LT;acceleration_LT;gyro_LT
0;0.5914306640625,-0.0322265625,-0.803955078125,-0.05303955078125;15636,686,-4848;8,-1,-6;0.68902587890625,0.07891845703125,-0.71533203125,0.08538818359375;16526,-232,-760;4,-5,2;0.62310791015625,-0.0042724609375,-0.781982421875,-0.01312255859375;16116,322,-3698;8,12,-2;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16392,-902,-324;3,1,-1
10;0.5914306640625,-0.03216552734375,-0.803955078125,-0.05303955078125;15872,750,-4976;7,-2,-5;0.68902587890625,0.07891845703125,-0.71533203125,0.08538818359375;16400,-216,-624;6,-6,0;0.62322998046875,-0.0042724609375,-0.78192138671875,-0.01318359375;16010,294,-3642;-3,14,-3;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16460,-836,-566;2,0,-3
20;0.5914306640625,-0.0321044921875,-0.803955078125,-0.052978515625;15878,760,-4894;7,1,-6;0.68896484375,0.0789794921875,-0.71539306640625,0.08544921875;16502,-272,-720;10,-4,2;0.62322998046875,-0.0042724609375,-0.78192138671875,-0.01318359375;16116,298,-3526;-8,11,-5;0.69683837890625,-0.030029296875,-0.716552734375,0.009765625;16426,-908,-354;-1,-2,-3
...
```
En este ejemplo simplificado, se muestran las primeras tres filas de datos.
