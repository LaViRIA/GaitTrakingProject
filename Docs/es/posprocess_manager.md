# Documentación del Script de Procesamiento en Paralelo

Este script Python automatiza la ejecución en paralelo del script `posProcess.py` (que se documentó previamente) sobre múltiples pares de archivos de video (vid) y datos de sensores inerciales (mcd). Utiliza la biblioteca `subprocess` para ejecutar `posProcess.py` como procesos separados, y `psutil` para controlar la afinidad de CPU de cada proceso, permitiendo así el procesamiento en paralelo y aprovechando múltiples núcleos de CPU.

## Dependencias

*   **tkinter.filedialog:** Para abrir un diálogo de selección de directorio.
*   **glob:** Para encontrar archivos que coincidan con un patrón (usado para encontrar los archivos MCD).
*   **subprocess:** Para ejecutar comandos del sistema (en este caso, para ejecutar `posProcess.py`).
*   **pathlib.Path:** Para comprobar si existen archivos de forma más eficiente y compatible entre sistemas operativos.
*   **time:** Para medir el tiempo de ejecución total y para introducir pausas.
*   **argparse:** Para analizar argumentos de línea de comandos.
*   **os:** Para obtener el número de núcleos de CPU disponibles (`os.cpu_count()`) y para operaciones relacionadas con el sistema operativo.
*    **psutil:** Para gestionar procesos.
*    **sys:** Para obtener la plataforma sobre la que se está ejecutando.

## Funciones

### `get_subject_n_capture(str_path)`

*   **Propósito:** Extrae el nombre del sujeto y el número de captura a partir de la ruta de un archivo MCD.
*   **Argumentos:**
    *   `str_path` (`str`): La ruta completa al archivo MCD.
*   **Retorno:**
    *   Una tupla `(subject, capture_num)`:
        *   `subject` (`str`): El nombre del sujeto (el nombre del directorio que contiene los archivos).
        *   `capture_num` (`str`): El número de captura (el último dígito del nombre del archivo MCD, antes de la extensión).
*   **Funcionamiento:**
    1.  Usa `str_path.split(split_char)` para dividir la ruta en sus componentes, usando el separador de directorios correcto (dependiente del sistema operativo).
    2.  Extrae el nombre del sujeto: `split[-2]`.
    3.  Extrae la parte relevante del nombre del archivo: `split[-1].split('.')[0][-1]`.
    4.  Devuelve la tupla `(subject, capture_num)`.

### `get_vid_path(main_folder, capture)`

*   **Propósito:** Construye la ruta completa a un archivo de video (vid) a partir de la carpeta principal del dataset y la información de la captura.
*   **Argumentos:**
    *   `main_folder` (`str`): La ruta a la carpeta principal del dataset.
    *   `capture` (`tuple`): Una tupla que contiene el nombre del sujeto y el número de captura (obtenida de `get_subject_n_capture`).
*   **Retorno:**
    *   `str`: La ruta completa al archivo de video.
* **Funcionamiento:**
    1.  Usa f-strings para construir la ruta al archivo de video, combinando la carpeta principal, el nombre del sujeto, el número de captura y el nombre base del archivo de video (`vid-{capture[1]}.mp4`).

### `get_mcd_path(main_folder, capture)`

*   **Propósito:** Construye la ruta completa a un archivo MCD a partir de la carpeta principal del dataset y la información de la captura.
*   **Argumentos:**
    *   `main_folder` (`str`): La ruta a la carpeta principal del dataset.
    *   `capture` (`tuple`): Una tupla que contiene el nombre del sujeto y el número de captura (obtenida de `get_subject_n_capture`).
*   **Retorno:**
    *   `str`: La ruta completa al archivo MCD.
* **Funcionamiento:**
    1.  Usa f-strings para construir la ruta, similar a `get_vid_path`, pero usando el nombre base del archivo MCD (`mcd-{capture[1]}.csv`).

### `run_script(mcd, vid, cores)`

*   **Propósito:** Ejecuta el script `posProcess.py` en un nuevo proceso con una afinidad de CPU específica.
*   **Argumentos:**
    *   `mcd` (`str`): La ruta al archivo MCD.
    *   `vid` (`str`): La ruta al archivo de video.
    *   `cores` (`list`): Una lista de dos enteros que especifica el rango de núcleos de CPU que se asignarán al proceso (límite inferior y superior inclusivos).
*   **Retorno:**
    *   Un objeto `subprocess.Popen` que representa el proceso en ejecución.
* **Funcionamiento:**
    1. Determina el comando a ejecutar basandose en la plataforma.
    2.  Utiliza `subprocess.Popen` para iniciar un nuevo proceso que ejecuta `posProcess.py` con los argumentos `mcd` y `vid`.  `Popen` permite ejecutar comandos en subprocesos separados, de forma no bloqueante.
    3.  Obtiene el PID (Process ID) del proceso creado.
    4.  Utiliza `psutil.Process(pid)` para obtener un objeto `psutil.Process` que representa el proceso recién creado.
    5.  Utiliza el método `p.cpu_affinity()` para establecer la afinidad de CPU del proceso.  `p.cpu_affinity([i for i in range(cores[0], cores[1]+1)])`  asigna el proceso a los núcleos de CPU especificados en la lista `cores`. Esto asegura que el proceso se ejecutará solo en los núcleos especificados.
    6.  Devuelve el objeto `subprocess.Popen`.

### `main`

*   **Propósito:** Función principal del script.  Coordina la selección de archivos, la configuración de los procesos, la ejecución en paralelo y la gestión del estado de los procesos.
*   **Argumentos de línea de comandos:**
    *   `process` (`int`): El número de procesos a ejecutar en paralelo.
    *   `core_min` (`int`): El índice del primer núcleo de CPU a utilizar (límite inferior).
    *   `core_max` (`int`): El índice del último núcleo de CPU a utilizar (límite superior).
*   **Variables:**
    *   `main_folder`: Carpeta del dataset.
    *   `files`: Lista con todos los archivos mcd.
    *   `captures`: Lista de tuplas, donde cada tupla contiene el nombre del sujeto y el número de captura.
    *   `num_process`: Número de procesos en paralelo.
    *   `cores`: Lista que contiene los indices de los nucleos a utilizar.
    *   `process_list`: Lista para almacenar los objetos `subprocess.Popen`.
    *   `process_running`: Lista de indicadores (0 o 1) para rastrear el estado de cada proceso (0: detenido, 1: en ejecución).
    *   `process_file`: Lista que guarda la tupla (sujeto, captura) que está ejecutando.
    *   `running_i`:  Índice para la animación de la barra de progreso.
    *   `running_chars`:  Caracteres para la animación de la barra de progreso.
    *   `t_init`:  Tiempo de inicio del script.
    * `print_eraser`: String para limpiar la linea de la consola

*   **Flujo principal:**
    1.  **Selección del directorio:**  Utiliza `filedialog.askdirectory()` para permitir al usuario seleccionar el directorio principal del dataset (`main_folder`).
    2.  **Encontrar archivos MCD:** Usa `glob.glob()` para encontrar todos los archivos MCD dentro del directorio seleccionado y sus subdirectorios. La búsqueda se realiza recursivamente dentro de las carpetas del dataset (`f'{main_folder}{split_char}*{split_char}mcd-*.csv'`). Los ordena.
    3.  **Extracción de información de captura:** Utiliza una lista por comprensión y la función `get_subject_n_capture` para crear la lista `captures`, extrayendo el sujeto y número de captura de cada archivo MCD encontrado.
    4.  **Argumentos de la línea de comandos:**  Usa `argparse` para procesar los argumentos de la línea de comandos: `process`, `core_min` y `core_max`.
    5.  **Configuración de procesos:**
        *   Establece `num_process` a partir del argumento `process`.
        *   Valida y establece `cores` a partir de los argumentos `core_min` y `core_max`. Asegura que core_min sea mayor o igual que 0 y core_max sea menor que la cantidad de nucleos de la computadora.
        *   Inicializa `process_list`, `process_running` y `process_file` con valores iniciales (`None` o 0).
    6.  **Bucle principal (bucle `while`):**
        *   El bucle continúa hasta que se hayan procesado todas las capturas (`i < len(captures)`) *o* hasta que todos los procesos hayan terminado (`sum(process_running) > 0`).
        *   **Creación de procesos:**
            *   Si hay menos procesos en ejecución que el número máximo permitido (`sum(process_running) < num_process`) y quedan capturas por procesar (`i < len(captures)`):
                *   Obtiene las rutas a los archivos de video y MCD para la siguiente captura usando `get_vid_path` y `get_mcd_path`.
                *   **Verificación de existencia de archivos:** Verifica si los archivos de video y MCD existen usando `Path(vid_file).exists()` y `Path(mcd_file).exists()`. Si alguno de los archivos no existe, imprime un mensaje de error y pasa a la siguiente captura.
                *   Encuentra un índice disponible en `process_running` (un índice donde el valor sea 0, indicando que no hay un proceso en ejecución en esa posición).
                *   Inicia un nuevo proceso usando `run_script()`, pasando las rutas a los archivos y los núcleos de CPU a utilizar.
                *   Actualiza `process_list`, `process_running` y `process_file` para reflejar el nuevo proceso en ejecución.
                *   Incrementa `i` para pasar a la siguiente captura.
        *   **Mostrar el estado:**
           * Borra la linea anterior.
            *   Crea una cadena de texto (`print_str`) que muestra el estado de cada proceso (en ejecución o detenido) junto con una animación simple (`running_chars`).
            *   Imprime la cadena de estado en la consola, usando `end='\r'` para sobrescribir la línea anterior y crear la animación.
            *  Actualiza `running_i` para la siguiente iteracion
        *   **Comprobar finalización de procesos:**
            *   Itera sobre `process_running`.
            *   Si un proceso está marcado como en ejecución (`p` es distinto de cero):
                *   Utiliza `process_list[idx].poll()` para comprobar si el proceso ha terminado.  `poll()` devuelve `None` si el proceso sigue ejecutándose, o el código de salida del proceso si ha terminado.
                *   Si el proceso ha terminado (`process_list[idx].poll() is not None`):
                    *   Borra la linea de la consola.
                    *   Imprime un mensaje indicando que el proceso ha finalizado.
                    *   Limpia las listas `process_list`, `process_running` y `process_file` para ese índice.
        * Pausa de 0.25 segundos.
    7.  **Imprimir tiempo total:** Una vez finalizado el bucle principal, imprime el tiempo total de ejecución.

### Ejecucion
Ejecutar este script require 3 parametros como se indica en la seccion anterior, por lo que el comando seria el sigueinte:
`python posProcess_manager.py <cantidad de procesos> <limite inferior de nucleos> <limite superior de nucleos>`
Ejemplo:
`py posProcess_manager.py 12 0 24`

> **Nota:* el rango de nucleos debe estar entre 0 y el numero maximo de nucleos, aunque si es menor a 0 o mayor al numero maximo, se limitara a ese rango.

### Sistema de archivos de entrada
El sistema de archivos requiere que el archivo `mcd-*.csv` y el archivo `vid-*.mp4` correspondiente se encuentren en la misma ruta.
Ejemplo:
```Bash
DataSet
├───ID0
│   ├───vid-0.mp4
│   ├───mcd-0.csv
│   ├───vid-1.mp4
│   ├───mcd-1.csv
│   ├───vid-2.mp4
│   └───mcd-2.csv
├───ID1
│   ├───vid-0.mp4
│   ├───mcd-0.csv
│   ├───vid-1.mp4
│   ├───mcd-1.csv
│   ├───vid-2.mp4
│   └───mcd-2.csv
└───ID2
    ├───vid-0.mp4
    ├───mcd-0.csv
    ├───vid-1.mp4
    ├───mcd-1.csv
    ├───vid-2.mp4
    └───mcd-2.csv
```
En resumen, este script orquesta la ejecución en paralelo de `posProcess.py` sobre un conjunto de archivos de video y MCD.  Permite al usuario especificar el número de procesos en paralelo y los núcleos de CPU a utilizar.  El script gestiona la creación de los procesos, la asignación de núcleos de CPU, la verificación del estado de los procesos y la sincronización general de la ejecución. El uso de `subprocess.Popen` en lugar de `multiprocessing.Process` permite un mayor control sobre la ejecución de los subprocesos, en particular con respecto a la afinidad de la CPU, y evita problemas relacionados con el GIL (Global Interpreter Lock) de Python en escenarios de uso intensivo de la CPU. La interfaz es simple, utilizando un selector de archivos y argumentos de línea de comandos.