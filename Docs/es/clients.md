### Documentación del Código

El código que se presenta a continuación es un ejemplo de implementación para una tarjeta ESP32, que utiliza diferentes módulos y librerías para realizar tareas de comunicación inalámbrica, adquisición de datos sensoriales, manejo de batería y control de leds. A continuación, se detalla la documentación del código:

#### **1. Inclusiones de Librerías**
El código incluye las siguientes librerías:

- `esp_now.h`: Para comunicación inalámbrica peer-to-peer.
- `WiFi.h` y `esp_wifi.h`: Para configuración y manejo del módulo Wi-Fi.
- `I2Cdev.h` y `MPU6050_6Axis_MotionApps612.h`: Para interactuar con el sensor MPU6050.
- `Wire.h`: Para comunicación I2C.
- `Adafruit_NeoPixel.h`: Para controlar LEDs NeoPixel.
- `EEPROM.h`: Para almacenamiento no volátil de datos.
- `Arduino.h`: Funciones básicas de Arduino.
- `FreeRTOS.h` y `timers.h`: Para programación multitarea en tiempo real.


Nota: La version 1.4.4 de la libreria mpu6050 de ElectronicsCats puede causar problemas al calibrar, recomiendo usar la version 1.4.0.  


#### **2. Definiciones y Configuraciones**

##### **Definiciones de GPIO**
```cpp
#define GPIO_SENSOR GPIO_NUM_18 //D10
#define GPIO_SENSOR_INT GPIO_NUM_19 //D8
#define GPIO_BUTTON GPIO_NUM_0 //D0
#define GPIO_LED_VCC GPIO_NUM_21 //D3
#define GPIO_LED_DATA GPIO_NUM_2 //D2
#define GPIO_BATTERY GPIO_NUM_6
```
- `GPIO_SENSOR`: Pin utilizado para el sensor MPU6050.
- `GPIO_SENSOR_INT`: Pin de interrupción del sensor.
- `GPIO_BUTTON`: Pin del botón de control.
- `GPIO_LED_VCC`: Pin de alimentación del LED NeoPixel.
- `GPIO_LED_DATA`: Pin de datos para el LED NeoPixel.
- `GPIO_BATTERY`:_pin para medir el voltaje de la batería.

##### **Constantes Temporales**
```cpp
#define SYNC_TIME 20000
#define IDENTIFY_TIME 10000
#define DATA_FREQUENCY_TIME 10 //ms
#define BATTERY_FREQUENCY_TIME 1000 //ms (1min)
#define CONNECT_TRY_TIME 1000 //ms (10s)
#define ALIVE_TIMEOUT 3000 //ms
#define ALIVE_MSG_FREQUENCY 1000 //ms
#define DEBOUNCING_TIME 5 //ms
```
- `SYNC_TIME`: Tiempo de sincronización (20 segundos).
- `IDENTIFY_TIME`: Duración del modo de identificación (10 segundos).
- `DATA_FREQUENCY_TIME`: Frecuencia de lectura y envío de datos sensoriales (10 ms).
- `BATTERY_FREQUENCY_TIME`: Frecuencia de chequeo de batería (1 segundo).
- `CONNECT_TRY_TIME`: Tiempo para intentar conexión (1 segundo).
- `ALIVE_TIMEOUT`:_timeout para mensajes "alive" (3 segundos).
- `ALIVE_MSG_FREQUENCY`: Frecuencia de envío de mensajes "alive" (1 segundo).
- `DEBOUNCING_TIME`: tiempo de debounce del botón (5 ms).

##### **Enum de Colores**
```cpp
enum Color {
  BLACK, WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE, PURPLE,
  LIME, TEAL, PINK, BROWN, GRAY, DARK_GRAY, LIGHT_GRAY, GOLD, SILVER,
  NAVY_BLUE, MAROON, OLIVE, SKY_BLUE, TURQUOISE, VIOLET, INDIGO, BEIGE,
  MINT, LAVENDER
};
```
- Define colores para el LED NeoPixel.

#### **3. Estructuras y Clases**

##### **Estructura sensor_read**
```cpp
struct sensor_read {
  Quaternion q;
  VectorInt16 a;
  VectorInt16 g;
};
```
- Almacena datos del sensor MPU6050: quaternion, aceleración y giroscopio.

##### **Clase SENSOR**
```cpp
class SENSOR : public MPU6050 {
private:
    uint8_t _fifoBuffer[64];
public:
    void init();
    void readSensor(sensor_read *data);
    int16_t * calibrate();
    void setOffsets(int16_t *offsets);
};
```
- Clase hija de `MPU6050` para simplificar la lectura y configuración del sensor.

#### **4. Variables Globales**
```cpp
Adafruit_NeoPixel strip(1, GPIO_LED_DATA, NEO_GRB);
SENSOR mpu;
uint8_t host_addr[6];
int host_EEPROM_addr = 0;
int offsets_EEPROM_addr = 10;
// ...otros timers y flags...
```
- `strip`: Objeto para controlar el LED NeoPixel.
- `mpu`: Instancia de la clase SENSOR para interactuar con el MPU6050.
- `host_addr`: Dirección MAC del host receptor.
- `host_EEPROM_addr`: Dirección en EEPROM para almacenar la dirección MAC del host.
- `offsets_EEPROM_addr`: Dirección en EEPROM para almacenar offsets de calibración.

#### **5. Funciones Principales**

##### **Función checkBattery**
```cpp
uint8_t checkBattery(){
  uint32_t Vbatt = 0;
  for(int i = 0; i < 50; i++) {
    Vbatt += analogReadMilliVolts(GPIO_BATTERY);   
  }
  int Vbattf = (int)(2.0 * (float)Vbatt / 50.0); 
  Vbattf = min(max(Vbattf, 3200), 4000);
  uint8_t charge = (int)map(Vbattf, 3200, 4000, 0, 100);
  return charge;
}
```
- Lee el voltaje de la batería y lo convierte en un porcentaje.

##### **Función OnDataRecv**
```cpp
void OnDataRecv(const esp_now_recv_info_t * mac, const uint8_t *incomingData, int len) {
    // Manejo de diferentes comandos recibidos...
}
```
- Función callback para manejar datos recibidos via ESP-NOW.

##### **Función btn_interrupt**
```cpp
void btn_interrupt() {
    if(!digitalRead(GPIO_BUTTON)){
      xTimerStart(debouncing_timer, 0);
    } else {
      xTimerStop(debouncing_timer, 0);
      xTimerStop(activate_deep_sleep_timer, 0);
    }
}
```
- Maneja interrupciones del botón para debounce y activación de modos.

##### **Función setup**
```cpp
void setup() {
    // Configuraciones iniciales...
    // Inicialización de pines, sensores, timers, etc.
}
```
- Configura todos los componentes: GPIO, sensor MPU6050, ESP-NOW, timers y LED.

##### **Función loop**
```cpp
void loop() {
    // Actualiza el LED según el estado del sistema...
}
```
- Función principal que actualiza el LEDs según el estado del sistema.

#### **6. Configuración de ESP-NOW**

##### **Función setHostMac**
```cpp
void setHostMac(const uint8_t *addr){
    // Configura y agrega un peer en ESP-NOW...
}
```
- Configura la dirección MAC del host receptor y establece parámetros de comunicación.

#### **7. Timers y Tareas**

El código utiliza timers para manejar diferentes tareas, como:

- Intentos de conexión (`connect_timer`).
- Timeout de desconexión (`disconnect_timer`).
- Enviar mensajes "alive" (`alive_timer`).
- Modo sincronización (`deactivate_sync_timer`).
- Contar pulsaciones del botón (`release_btn_timer`).
- Chequeo de batería (`battery_check_timer`).
- Lectura y envío de datos sensoriales (`reading_sensor_timer`).

#### **8. Modos Especiales**

- **Modo sincronización**: Permite establecer una nueva dirección MAC del host.
- **Modo identificación**: Parpadeo de LED para identificar el dispositivo.
- **Modo deep sleep**: Baja consumo de energía cuando no se usa.

#### **9. Controles y Estado**

El sistema maneja diferentes estados:

- Conexión establecida (`connected`).
- Lectura activa de sensores (`reading_sensor`).
- Modo sincronización (`sync_mode`).
- Batería baja (`low_battery`).
