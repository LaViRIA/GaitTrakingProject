# Documentación del Código de Seguimiento de Movimiento con ArUco

Este código realiza el seguimiento del movimiento de marcadores ArUco en un video y sincroniza los datos con mediciones de un sensor inercial (MCD).

## Dependencias

*   **cv2 (OpenCV):** Biblioteca para procesamiento de imágenes y visión artificial.
*   **numpy:** Biblioteca para cálculos numéricos y manipulación de arrays.
*   **scipy.spatial.transform:**  Para el manejo de rotaciones y transformaciones espaciales.
*   **csv:** Para leer y escribir archivos CSV.
*   **scipy.interpolate:** Para interpolación de datos.
*   **argparse:** Para analizar argumentos de línea de comandos.
*   **pymediainfo:** Para obtener información de archivos multimedia (como el video).
*   **sys:**  Para obtener la plataforma sobre la que se está ejecutando.
* **utils.calibration_matrix**: Módulo con los datos de la camera.

## Funciones

### `init_csv_file(path: str, fields: list[str], delimiter: str, comments: str = None)`

*   **Propósito:** Inicializa un archivo CSV para escribir los resultados.
*   **Argumentos:**
    *   `path` (`str`): La ruta donde se creará el archivo CSV.
    *   `fields` (`list[str]`): Una lista de cadenas que representan los nombres de las columnas del encabezado del CSV.
    *   `delimiter` (`str`): El carácter delimitador que se usará para separar los campos en el CSV (por ejemplo, '`;`', '`,`', '`\t`').
    *   `comments` (`str`, opcional): Comentarios para añadir al inicio del archivo.
*   **Retorno:**
    *   Una tupla que contiene:
        *   Un objeto de archivo abierto en modo escritura (`file`).
        *   Un objeto `csv.DictWriter` configurado para escribir diccionarios en el archivo CSV.

*   **Funcionamiento:**
    1.  Abre el archivo especificado en `path` en modo escritura (`'w'`), con la opción `newline=''` para evitar problemas con saltos de línea en diferentes sistemas operativos.
    2.  Si hay comentarios, los escribe.
    3.  Crea un objeto `csv.DictWriter`.  Este objeto se usará para escribir filas en el CSV, donde cada fila es un diccionario.  Los `fieldnames` definen las claves del diccionario y el orden de las columnas.
    4.  Escribe la fila de encabezado usando `csv_writer.writeheader()`.
    5.  Devuelve el objeto de archivo y el escritor CSV.

### `csv_comment(comment)`

*   **Propósito:** Extrae los datos del comentario del archivo de calibración (mcd).
*   **Argumentos:**
    *   `comment` (`str`): Comentario extraído de la primera linea del archivo.
*   **Retorno:**
    *   Un diccionario con los valores parseados.
*   **Funcionamiento:**
    1.  Elimina los comentarios del string y los espacios en blanco.
    2.  Separa el string.
    3.  Crea el diccionario.
    4.  Retorna el diccionario.

### `get_object_points(marker_length)`

*   **Propósito:** Define las coordenadas 3D de las esquinas de un marcador ArUco en su sistema de coordenadas local.
*   **Argumentos:**
    *   `marker_length` (`float`): La longitud del lado del marcador ArUco (en metros o la unidad de medida que se esté utilizando).
*   **Retorno:**
    *   Un array NumPy (`np.ndarray`) de forma (4, 3) y tipo `np.float32`.  Cada fila representa una esquina del marcador, con las coordenadas (x, y, z).  El marcador se asume centrado en el origen (0, 0, 0) y en el plano XY (z=0).

*   **Funcionamiento:**
    *   Crea un array NumPy con las coordenadas de las cuatro esquinas del marcador.  Las coordenadas se calculan en función de `marker_length`.
    *   Establece el tipo de datos del array a `np.float32`.
    *   Devuelve el array.

### `get_transformation_matrix(rvec, tvec)`

*   **Propósito:** Crea una matriz de transformación homogénea 4x4 a partir de un vector de rotación y un vector de traslación.
*   **Argumentos:**
    *   `rvec` (`np.ndarray`): Vector de rotación (3x1 o 1x3) que representa la rotación del objeto (usando la representación de Rodrigues).
    *   `tvec` (`np.ndarray`): Vector de traslación (3x1 o 1x3) que representa la traslación del objeto.
*   **Retorno:**
    *   `transformation_matrix` (`np.ndarray`): Una matriz de transformación homogénea 4x4.

*   **Funcionamiento:**
    1.  Crea una matriz identidad 4x4 (`np.eye(4)`).
    2.  Convierte el vector de rotación `rvec` a una matriz de rotación 3x3 usando `cv2.Rodrigues(rvec)[0]`.  La función `cv2.Rodrigues()` convierte entre la representación de Rodrigues (vector de rotación) y la matriz de rotación.  Se usa `[0]` para obtener solo la matriz de rotación, ya que `cv2.Rodrigues()` también devuelve el Jacobiano.  Esta matriz de rotación se coloca en la submatriz superior izquierda 3x3 de la matriz de transformación.
    3.  Convierte `tvec` a un vector columna usando `tvec.flatten()` y lo coloca en la última columna (índice 3) de las primeras tres filas (índices 0:3) de la matriz de transformación.
    4.  Devuelve la matriz de transformación 4x4 resultante.

### `process_mcd_file(path)`

*   **Propósito:** Lee y procesa un archivo MCD (archivo de datos del sensor inercial) en formato CSV.
*   **Argumentos:**
    *   `path` (`str`): La ruta al archivo MCD.
*   **Retorno:**
    *   Una tupla que contiene:
        *   `measures` (`dict`): Un diccionario con los datos de la primera linea del archivo.
        *   `mcd_t` (`list[float]`): Una lista con los valores de tiempo del archivo MCD.
        *   `mcd_av` (`list[float]`): Una lista con los valores de velocidad angular calculados a partir de los datos del giroscopio en el archivo MCD.

*   **Funcionamiento:**
    1.  Inicializa las variables.
    2.  Abre el archivo MCD en modo lectura (`'r'`).
    3.  Lee los datos de la primera linea.
    4.  Crea un `csv.DictReader` para leer el resto del archivo como un CSV. El delimitador se establece en ';'.
    5.  Itera sobre las filas del archivo CSV:
        *   Extrae los datos del giroscopio de la columna apropiada (determinada por `measures["direction"]`). La columna específica se selecciona usando una f-string:  `f'gyro_{measures["direction"].strip()}F'`.  Los valores del giroscopio se asumen como una cadena separada por comas, que se convierte a una lista de flotantes.
            > Estos datos especificos son relevantes para sincronizar los datos de los ArUcos ya que `measures["direction"]` indica la pierna que es siempre visible para la camara.
        *   Calcula la velocidad angular a partir de los datos del giroscopio. Se multiplica por constantes de conversión (1/131.0, pi/180, y 5)  para convertir las unidades del giroscopio a radianes por segundo.
        *   Agrega el valor de la velocidad angular calculada a la lista `mcd_av`.
        *   Agrega el valor de tiempo (columna 'time') a la lista `mcd_t`.
    6.  Devuelve `measures`, `mcd_t`, y `mcd_av`.

### `apply_wiener_filter(img, kernel_size, noise_power)`

*   **Propósito:** Aplica un filtro de Wiener a una imagen para reducir el desenfoque de movimiento lineal.
*   **Argumentos:**
    *   `img` (`np.ndarray`): La imagen de entrada (en escala de grises).
    *   `kernel_size` (`int`): El tamaño del kernel de desenfoque lineal (debe ser impar).
    *   `noise_power` (`float`): Una estimación de la potencia del ruido en la imagen.
*   **Retorno:**
    *   `result` (`np.ndarray`): La imagen filtrada, con el desenfoque reducido.

*   **Funcionamiento:**
    1.  **Crea el kernel de desenfoque:**
        *   Crea un kernel de ceros (`np.zeros`) de tamaño `kernel_size` x `kernel_size`.
        *   Crea un kernel de desenfoque lineal, donde los pixeles en la fila central representan el movimiento.
    2.  **Calcula las Transformadas de Fourier:**
        *   Calcula la Transformada Rápida de Fourier (FFT) de la imagen (`img_fft`) usando `np.fft.fft2()`.
        *   Calcula la FFT del kernel (`kernel_fft`),  ajustando el tamaño del kernel al tamaño de la imagen con el argumento `s=img.shape`.
    3.  **Evita divisiones por cero:**
        *   Reemplaza cualquier valor cero en `kernel_fft` con un valor muy pequeño (`1e-7`) para evitar errores de división por cero en el cálculo del filtro de Wiener.
    4.  **Calcula el filtro de Wiener:**
        *   Calcula el filtro de Wiener en el dominio de la frecuencia.  El filtro de Wiener es una estimación óptima del filtro inverso que minimiza el error cuadrático medio. La fórmula utilizada es: `kernel_wiener = np.conj(kernel_fft) / (np.abs(kernel_fft)**2 + noise_power)`.
    5.  **Aplica el filtro:**
        *   Multiplica la FFT de la imagen (`img_fft`) por el filtro de Wiener (`kernel_wiener`).
    6.  **Transformada Inversa:**
        *   Calcula la Transformada Inversa Rápida de Fourier (IFFT) del resultado anterior (`img_wiener`) usando `np.fft.ifft2()`.  Esto devuelve la imagen filtrada al dominio espacial.
    7.  **Ajusta el resultado:**
        *   Toma el valor absoluto de la imagen resultante (`np.abs(img_wiener)`) ya que la IFFT puede devolver números complejos debido a errores numéricos.
        *   Recorta los valores de la imagen resultante para que estén entre 0 y 255 (`np.clip(img_result, 0, 255)`) y convierte la imagen a tipo `np.uint8` para que sea una imagen válida en escala de grises.
    8.  Aplica un filtro gaussiano a la imagen resultante para mejorar la detección de los arUcos.
    9.  Retorna el resultado.

### `get_aruco_file_path(str_path)`

*   **Propósito:** Construye el nombre del archivo de salida para guardar los datos de ArUco, basándose en el nombre del archivo MCD y la plataforma.
*   **Argumentos:**
    *   `str_path` (`str`): La ruta completa del archivo MCD.
*   **Retorno:**
    *   `str`: La ruta completa del archivo de salida de ArUco.

*   **Funcionamiento:**
    1.  Determina el carácter separador de directorios (`/` o `\`) según la plataforma (Linux o Windows).
    2.  Divide la ruta del archivo MCD usando el separador de directorios.
    3.  Extrae el número del video del nombre del archivo (asumiendo que el número está al final del nombre del archivo, antes de la extensión).
    4.  Construye el nombre del archivo de salida de ArUco en formato `arUcos-{num}.csv`, donde `{num}` es el número extraído del nombre del archivo MCD.
    5.  Concatena para obtener la ruta completa.

### `main`

*   **Propósito:**  Función principal del programa.  Coordina la lectura del video, la detección de ArUcos, el procesamiento de los datos, la sincronización con el MCD y el guardado de los resultados.
*   **Argumentos de línea de comandos:**
    *   `mcd`: Ruta al archivo CSV del MCD.
    *   `vid`: Ruta al archivo de video.
*   **Variables globales/constantes:**
    *   `debug_mode` (`bool`): Activa/desactiva el modo de depuración (imprime información adicional y muestra la imagen procesada).  Está establecido en `False`.
    *   `aruco_dict`:  Diccionario ArUco predefinido (`cv2.aruco.DICT_4X4_50`).
    *   `parameters`: Objeto `cv2.aruco.DetectorParameters` con parámetros para la detección de ArUcos. Se configuran varios parámetros, incluyendo:
        *   `polygonalApproxAccuracyRate`:  Controla la precisión de la aproximación poligonal de los contornos de los marcadores.
        *   `minCornerDistanceRate`: Distancia mínima entre esquinas.
        *   `cornerRefinementMethod`: Método de refinamiento de esquinas (se usa `cv2.aruco.CORNER_REFINE_APRILTAG`).
        *   `errorCorrectionRate`: Tasa de corrección de errores.
        *   `cornerRefinementMinAccuracy`: Precisión mínima para el refinamiento de esquinas.
        *   `cornerRefinementWinSize`: Tamaño de la ventana para el refinamiento de esquinas.
        *   `perspectiveRemoveIgnoredMarginPerCell`:  Margen ignorado alrededor de cada celda del marcador durante la eliminación de la perspectiva.
        *   `minMarkerPerimeterRate`:  Tasa mínima del perímetro del marcador en relación con el tamaño de la imagen.
    *   `detector`:  Objeto `cv2.aruco.ArucoDetector` que se crea con el diccionario y los parámetros.
    *   `markers_size` (`float`): Tamaño real de los marcadores ArUco (en metros).
    *   `origin_size` (`float`): Tamaño real del marcador de origen (en metros).
    *   `arUcos_ids` (`dict`):  Diccionario que mapea los IDs de los ArUcos a nombres descriptivos (ej. 'hip', 'R_knee').
    *   `arUcos_ids_inv` (dict):  Diccionario inverso de `arUcos_ids`.
    *   `origin_id` (`int`): ID del ArUco que se usará como origen del sistema de coordenadas.
    *   `origin_Tmatrix` (`np.ndarray`):  Matriz de transformación del ArUco de origen.  Inicializada a `None`.
    *   `origin_3D`: Puntos 3D del marcador de origen.
    *   `markers_3D`: Puntos 3D de los marcadores.
    *   `rows` (`list`): Lista para almacenar los datos de cada frame procesado (en forma de diccionario).
    *   `aruco_m` (`list`): Lista para almacenar las pendientes de la línea que une la cadera y la rodilla, para la sincronización.
    *   `aruco_t` (`np.ndarray`): Array NumPy para almacenar los tiempos de los frames en los que se detectan ArUcos, para la sincronización.
    *   `last_p` (`dict`):  Diccionario para almacenar la última posición detectada de cada ArUco. Esto se usa para recortar la imagen y mejorar la detección en frames subsiguientes.
    *   `camera_matrix`: Matriz de cámara.
    *   `dist_coeffs`: Coeficientes de distorsión.
    *   `frame_rate`: Tasa de frames del video.
    *   `cap_fps`: Tasa de frames del video, forzándola a float.
    *   `cap`:  Objeto `cv2.VideoCapture` para capturar el video.

*   **Flujo principal:**
    1.  **Argumentos:** Procesa los argumentos.
    2.  **Inicialización:**
        *   Obtiene información del video (`media_info`) usando `pymediainfo.MediaInfo.parse()`.
        *   Inicializa el detector de ArUcos.
        *   Define el tamaño de los marcadores y el origen.
        *   Crea los arrays para almacenar los datos de salida.
        *   Procesa el archivo MCD.
        *   Inicializa `last_p`.
        *   Obtiene la matriz de cámara y los coeficientes de distorsión.
        *   Obtiene la tasa de frames del video.
        *   Abre el archivo de video con `cv2.VideoCapture`.
    3.  **Bucle principal (por cada frame):**
        *   Lee un frame del video (`cap.read()`).
        *   Convierte el frame a escala de grises (`cv2.cvtColor`).
        *   **Modo debug:**  Si `debug_mode` es `True`, muestra el frame en una ventana.
        *   **Recorte del frame:**  Si se detectaron ArUcos en frames anteriores, recorta la imagen alrededor de la última posición conocida de los ArUcos.
        *   **Detección de ArUcos:** Detecta ArUcos en el frame (o en la región recortada) usando `detector.detectMarkers()`.
        *   **Manejo de no detección:** Si no se detectan ArUcos, se salta al siguiente frame.
        *   **Reintentos de detección:**
            *   Itera sobre los IDs de los ArUcos esperados.
            *   Si un ArUco esperado no se detectó en el primer intento y tenemos una posición previa para él (`type(last_p[i]) != np.ndarray`), intenta detectarlo nuevamente en una región más pequeña alrededor de la última posición conocida.
            *    Dentro del bucle de reintentos, se aplica el filtro de Wiener (`apply_wiener_filter`) con diferentes parámetros (`kernel_size` y `noise_power`) para intentar mejorar la detección en presencia de desenfoque de movimiento. Se prueban varios tamaños de kernel y niveles de ruido. Se itera sobre `kernel_size` de 17 a 23 en pasos de 2, y sobre `noise_power` de 0.02 a 0.032 en pasos de 0.002.
            *   Si el filtro de Wiener y la redetección tienen éxito, se actualiza `last_p`, se agregan los nuevos datos a `mk_ids` y `mk_corners`, y se sale de los bucles de reintento (`break`).
        *   **Cálculo de la pose:**
            *   Si se detecta el ArUco de origen (`origin_id in mk_ids`):
                *   Calcula la matriz de transformación del origen usando `cv2.solvePnP()` y `get_transformation_matrix()`. `solvePnP()` estima la pose (posición y orientación) de un objeto 3D a partir de sus puntos 3D y las correspondientes proyecciones 2D en la imagen. Se usan los puntos 3D del origen (`origin_3D`), las esquinas detectadas del origen (`origin_corners`), la matriz de la cámara y los coeficientes de distorsión.
                *   Si no se puede calcular la matriz de transformación del origen (la función `solvePnP` falla), salta al siguiente frame (`continue`).
            *   Itera sobre los ArUcos detectados (`zip(mk_ids, mk_corners)`):
                *   Si el ID del ArUco no está en la lista de IDs esperados (`id not in arUcos_ids.keys()`), lo ignora (`continue`).
                *   Calcula la matriz de transformación de cada ArUco usando `cv2.solvePnP()` y `get_transformation_matrix()`. Se usan `markers_3D`, las esquinas del ArUco (`corners[0]`), `camera_matrix` y `dist_coeffs`.
                *   Si `solvePnP()` falla, salta al siguiente ArUco (`continue`).
                *   Calcula la matriz de transformación del ArUco relativa al origen usando `inv(origin_Tmatrix) @ Tmatrix`. Esto transforma las coordenadas del ArUco al sistema de coordenadas del origen.
                *   Extrae las coordenadas de posición (`coors`) y la orientación (como un cuaternión, `orientation`) de la matriz de transformación relativa. La posición se extrae de la última columna de la matriz de transformación. La orientación se calcula a partir de la matriz de rotación (submatriz 3x3 superior izquierda) usando `scipy.spatial.transform.Rotation.from_matrix().as_quat(scalar_first=True)`.
                *   Almacena la posición y orientación en el diccionario `row`, usando claves como `'hip_position'`, `'R_knee_orientation'`, etc.
        *   **Almacenamiento de datos:**
            *   Agrega el diccionario `row` (con la información del frame actual, incluyendo el tiempo y la posición/orientación de los ArUcos detectados) a la lista `rows`.
            *   Si se detectaron los ArUcos de la cadera (`hip`) y la rodilla (`knee`) del lado correspondiente a `measures["direction"]`, calcula la pendiente de la línea que los une (`(hip[0]-knee[0])/(hip[1]-knee[1])`) y la almacena en la lista `aruco_m`. También almacena el tiempo del frame actual en `aruco_t`. Estos datos se usarán para la sincronización.
            *   Incrementa el contador de frames (`frame_number`).
    4.  **Sincronización:**

        *   Calcula el desfase (`offset`) entre los datos del video (ArUcos) y los datos del MCD (sensor inercial) usando correlación cruzada.
            *   Convierte las listas de tiempos y velocidades angulares del MCD (`mcd_t`, `mcd_av`) a arrays NumPy.
            *   Suavizado de los datos de velocidad angular del MCD: Aplica un filtro de media móvil a `mcd_av` usando `np.convolve(mcd_av, np.ones(w)/w, mode='valid')`. Esto reduce el ruido en los datos del MCD y mejora la precisión de la sincronización. `w` es el ancho de la ventana del filtro (20 en este caso). Se ajustan los tiempos tambien.
            *   Interpolación: Interpola los datos de velocidad angular del MCD (`mcd_av_f`) y las pendientes de los ArUcos (`aruco_m_f`) para que tengan la misma frecuencia de muestreo y se puedan comparar directamente. Se usa interpolación lineal (`scipy.interpolate.interp1d`) con extrapolación (`fill_value='extrapolate'`).
            *   Define un rango de tiempo común (`t`) para ambas señales, que abarca el período en el que ambas señales tienen datos.
            *   Calcula la correlación cruzada entre las dos señales interpoladas (`np.correlate(mcd_av_int, aruco_m_int, mode='full')`). La correlación cruzada mide la similitud entre dos señales en función del desfase temporal entre ellas. El argumento `mode='full'` indica que se debe calcular la correlación completa, incluyendo los desfases en los que las señales no se superponen completamente.
            *   Encuentra el índice del valor máximo de correlación (`idx_max = np.argmax(correlation)`). Este índice corresponde al desfase que maximiza la similitud entre las dos señales.
            *   Calcula el offset como la diferencia entre el índice máximo y la longitud de la señal de ArUco menos 1 (`offset = idx_max - (len(aruco_m_int)-1)`). Este offset representa el desfase temporal entre las dos señales, expresado en unidades de tiempo (milisegundos).
            *   Aplicación del offset: Aplica el offset a los tiempos de los datos de ArUco (`r['time'] += offset`) para sincronizarlos con los datos del MCD.
    5. **Guardado de resultados:**

        *   Define los nombres de las columnas (`csv_fields`) para el archivo CSV de salida.
        *   Crea el archivo CSV de salida usando la función `init_csv_file()`. El nombre del archivo se genera usando `get_aruco_file_path()`.
        *   Escribe los datos almacenados en la lista `rows` en el archivo CSV usando `csv_file.writerows()`. Cada elemento de `rows` es un diccionario que representa un frame, y `writerows()` escribe cada diccionario como una fila en el CSV.
        *   Cierra el archivo CSV (`file.close()`).
