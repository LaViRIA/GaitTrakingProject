## Documentación del Archivo CSV de Salida

El script genera un archivo CSV (con extensión `.csv`) que contiene los datos de seguimiento de los marcadores ArUco, sincronizados con los datos del sensor inercial (MCD). El archivo utiliza el punto y coma (`;`) como delimitador.

### Estructura General
El archivo CSV tiene la siguiente estructura:

*   **Encabezado:** La primera fila contiene los nombres de las columnas (campos).
*   **Datos:** Cada fila subsiguiente representa un instante de tiempo (un frame del video procesado) y contiene los datos de posición y orientación de los marcadores ArUco detectados en ese frame.
*   **No hay datos faltantes:** Se intenta que para cada instante haya información. En caso de que no haya datos para una columna en un instante, se inserta un string vacio.
### Columnas (Campos)
El archivo CSV contiene las siguientes columnas:

1.  **`time:`** Tiempo en milisegundos, sincronizado con los datos del MCD.  Es el tiempo transcurrido desde el inicio del video, ajustado por el offset calculado mediante la correlación cruzada.

2.  **`hip_position:`** Posición 3D del marcador ArUco identificado como "hip" (cadera) en el sistema de coordenadas del marcador de origen.  Las coordenadas se expresan como una cadena de texto con los valores x, y, z separados por comas (por ejemplo, "0.1,0.2,-0.05").  Las unidades están en metros.

3.  **`hip_orientation:`** Orientación del marcador ArUco "hip" expresada como un cuaternión. Los valores w, x, y, z del cuaternión se almacenan como una cadena de texto separada por comas (por ejemplo, "0.707,0,0.707,0").  El cuaternión representa la rotación desde el sistema de coordenadas del marcador de origen hasta el sistema de coordenadas del marcador "hip".

4.  **`R_knee_position:`** Posición 3D del marcador ArUco identificado como "R_knee" (rodilla derecha) en el sistema de coordenadas del marcador de origen.  Formato y unidades iguales a hip_position.

5.  **`R_knee_orientation:`** Orientación del marcador ArUco "R_knee" expresada como un cuaternión. Formato igual a `hip_orientation`.

6.  **`L_knee_position:`** Posición 3D del marcador ArUco identificado como "L_knee" (rodilla izquierda) en el sistema de coordenadas del marcador de origen.  Formato y unidades iguales a `hip_position`.

7.  **`L_knee_orientation:`** Orientación del marcador ArUco "L_knee" expresada como un cuaternión. Formato igual a `hip_orientation`.

8.  **`R_ankle_position:`** Posición 3D del marcador ArUco identificado como "R_ankle" (tobillo derecho) en el sistema de coordenadas del marcador de origen.  Formato y unidades iguales a `hip_position`.

9.  **`R_ankle_orientation:`** Orientación del marcador ArUco "R_ankle" expresada como un cuaternión.  Formato igual a `hip_orientation`.

10. **`L_ankle_position:`** Posición 3D del marcador ArUco identificado como "L_ankle" (tobillo izquierdo) en el sistema de coordenadas del marcador de origen.  Formato y unidades iguales a `hip_position`.

11. **`L_ankle_orientation:`** Orientación del marcador ArUco "L_ankle" expresada como un cuaternión. Formato igual a `hip_orientation`.

### Notas importantes:

*   **Sistema de coordenadas:** Todas las posiciones se expresan en el sistema de coordenadas del marcador ArUco de origen (identificado por `origin_id` en el código). Esto significa que la posición del marcador de origen siempre será (0, 0, 0).
*   **Unidades:** Las posiciones se expresan en metros.
*   **Cuaterniones:** Las orientaciones se expresan como cuaterniones unitarios. Un cuaternión es una forma de representar una rotación en 3D.
*   **Sincronización:** Los tiempos (`time`) están sincronizados con los datos del archivo MCD, lo que permite relacionar las mediciones del sensor inercial con la posición y orientación de los marcadores ArUco.
*   **Nombre del archivo:** El nombre del archivo se basa en el nombre del archivo MCD, siguiendo el patrón `arUcos-{num}.csv`, donde `{num}` es un número extraído del nombre del archivo MCD.
*   **Valores faltantes:** Si un ArUco no se detecta en un frame en particular, sus campos de posición y orientación estaran vacíos.

### Ejemplo
```
time;hip_position;hip_orientation;R_knee_position;R_knee_orientation;L_knee_position;L_knee_orientation;R_ankle_position;R_ankle_orientation;L_ankle_position;L_ankle_orientation
-915;1.0812701974126873,0.19553333992405364,1.7730267768679695;-0.03579755363431741,-0.3027704605265213,-0.07767882216617086,0.9492178801377653;;;1.1903190364076657,-0.16535499196021697,1.5738097880703292;-0.04663701743823387,-0.5796669644370976,-0.03964008542739295,0.8125514522613662;;;1.3236439703535121,-0.5518511445951062,1.3846893687834791;0.03225431442603979,0.7038767886366579,0.14632608847877795,-0.6943383911681338
-882;1.4069629577923255,0.29248264061831386,1.6587844085733763;-0.058889696552360926,-0.21736234964294096,-0.05050321232586272,0.9730031028431717;;;1.4171593853951752,-0.08855974405245881,1.5316496108576465;0.23590359826200835,-0.162307849085637,0.11842638852687046,0.9507790726308156;;;1.555216435406157,-0.45256777596924824,1.277258768385952;0.18743136721138356,-0.004164664486858101,-0.022024020895057173,0.9820219349172387
-849;1.4118625766399564,0.29080435721728504,1.65334942309934;-0.05518302337878983,-0.2166081215371079,-0.050789377709053514,0.9733736151791194;;;1.4222101738202126,-0.09045581507646427,1.5252423374779687;0.23729410460036848,-0.1644767270161073,0.11742082256555682,0.9501848581302516;;;1.5579534378611228,-0.45448027178301664,1.2729650765155127;0.19158570608621664,-0.0019353296750931896,-0.01941407573348632,0.9812819499949589
```