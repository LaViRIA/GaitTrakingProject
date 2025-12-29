/*
Author: Moises Badajoz Martinez <m.badajozmartinez@ugto.mx>

University of Guanajuato (2025)
*/
#include <esp_now.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include "I2Cdev.h"
#include "helper_3dmath.h"
#include "MPU6050_6Axis_MotionApps612.h"
#include "Wire.h"
#include <Adafruit_NeoPixel.h>
#include <EEPROM.h>
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/timers.h>
#include "driver/rtc_io.h"

#define GPIO_SENSOR GPIO_NUM_18 //D10
#define GPIO_SENSOR_INT GPIO_NUM_19 //D8
#define GPIO_BUTTON GPIO_NUM_0 //D0
#define GPIO_LED_VCC GPIO_NUM_21 //D3
#define GPIO_LED_DATA GPIO_NUM_2 //D2
#define GPIO_BATTERY GPIO_NUM_6
#define BUTTON_PIN_BITMASK (1ULL << GPIO_NUM_0)
#define SYNC_TIME 20000 //ms (20s)
#define IDENTIFY_TIME 10000 //ms (10s)
#define DATA_FREQUENCY_TIME 10 //ms
#define BATTERY_FREQUENCY_TIME 1000 //ms (1s)
#define CONNECT_TRY_TIME 1000 //ms (1s)
#define ALIVE_TIMEOUT 3000 //ms (3s)
#define ALIVE_MSG_FREQUENCY 1000 //ms (1s)
#define DEBOUNCING_TIME 5 //ms
// enum de colores para el led
enum Color {
  BLACK         = 0x000000,   // Negro
  WHITE         = 0xFFFFFF,   // Blanco
  RED           = 0xFF0000,   // Rojo
  GREEN         = 0x00FF00,   // Verde
  BLUE          = 0x0000FF,   // Azul
  YELLOW        = 0xFFFF00,   // Amarillo
  CYAN          = 0x00FFFF,   // Cian
  MAGENTA       = 0xFF00FF,   // Magenta
  ORANGE        = 0xFF8000,   // Naranja
  PURPLE        = 0x800080,   // Púrpura
  LIME          = 0x80FF00,   // Verde Lima
  TEAL          = 0x008080,   // Verde Azulado
  PINK          = 0xFFC0CB,   // Rosa
  BROWN         = 0x964B00,   // Marrón
  GRAY          = 0x808080,   // Gris
  DARK_GRAY     = 0x404040,   // Gris Oscuro
  LIGHT_GRAY    = 0xC0C0C0,   // Gris Claro
  GOLD          = 0xFFD700,   // Dorado
  SILVER        = 0xC0C0C0,   // Plateado (igual a Gris Claro)
  NAVY_BLUE     = 0x000080,   // Azul Marino
  MAROON        = 0x800000,   // Granate
  OLIVE         = 0x808000,   // Oliva
  SKY_BLUE      = 0x87CEEB,   // Azul Cielo
  TURQUOISE     = 0x40E0D0,   // Turquesa
  VIOLET        = 0xEE82EE,   // Violeta
  INDIGO        = 0x4B0082,   // Índigo
  BEIGE         = 0xF5F5DC,   // Beige
  MINT          = 0x98FF98,   // Menta
  LAVENDER      = 0xE6E6FA    // Lavanda
};
// estructura para los datos del sensor MPU6050: quaternion, aceleracion y giroscopio.
struct sensor_read {
  Quaternion q;
  VectorInt16 a;
  VectorInt16 g;
};
// clase para simplificar la lectura y configuracion del sensor
class SENSOR : public MPU6050 {
  private:
    uint8_t _fifoBuffer[64];
    
  public:
    void init(){
      initialize();
      dmpInitialize();
      setDMPEnabled(true);
    }
    
    void readSensor(sensor_read *data) {
      dmpGetCurrentFIFOPacket(_fifoBuffer);
      dmpGetQuaternion(&data->q, _fifoBuffer);
      dmpGetAccel(&data->a, _fifoBuffer);
      dmpGetGyro(&data->g, _fifoBuffer);
    }

    int16_t * calibrate(){
      CalibrateAccel();
      CalibrateGyro();
      return GetActiveOffsets();
    }

    void setOffsets(int16_t *offsets){
      setXAccelOffset(offsets[0]);
      setYAccelOffset(offsets[1]);
      setZAccelOffset(offsets[2]);
      setXGyroOffset(offsets[3]);
      setYGyroOffset(offsets[4]);
      setZGyroOffset(offsets[5]);
    }

};

void setHostMac(const uint8_t *addr);

Adafruit_NeoPixel strip(1, GPIO_LED_DATA, NEO_GRB);
SENSOR mpu;
uint8_t host_addr[6];
int host_EEPROM_addr = 0;
int offsets_EEPROM_addr = 10;

TimerHandle_t connect_timer;
TimerHandle_t disconnect_timer;
TimerHandle_t alive_timer;
TimerHandle_t activate_deep_sleep_timer;
TimerHandle_t release_btn_timer;
TimerHandle_t deactivate_sync_timer;
TimerHandle_t debouncing_timer;
TimerHandle_t battery_check_timer;
TimerHandle_t low_battery_timer;
TimerHandle_t deactivate_identify_timer;
TimerHandle_t reading_sensor_timer;
bool calibrate_sensor = false;
bool reading_sensor = false;
bool connected = false;
bool identify_mode = false;
bool low_battery = false;
volatile uint8_t release_count = 0;
volatile bool sync_mode = false;

// Lee el nivel de la bateria
uint8_t checkBattery(){
  uint32_t Vbatt = 0;
  for(int i = 0; i < 50; i++) {
    Vbatt += analogReadMilliVolts(GPIO_BATTERY); // Registra 50 lecturas del voltaje
  }
  // Promedio de las 50 lecturas, multiplicado x2 por el divisor de voltaje
  int Vbattf = (int)(2.0 * (float)Vbatt / 50.0);
  Vbattf = min(max(Vbattf, 3200), 4000); // Calcula el voltaje entre 3.2V y 4.0V
  uint8_t charge = (int)map(Vbattf, 3200, 4000, 0, 100);
  return charge;
}

void OnDataRecv(const esp_now_recv_info_t * mac, const uint8_t *incomingData, int len) {
  static sensor_read data;
  // Si la direccion de origen es desconocida ignora el mensaje, a menos que este en modo sincronizacion
  if(!esp_now_is_peer_exist(mac->src_addr) && !sync_mode)
    return;
  // revisa el tipo de mensaje
  switch (incomingData[0]){
    // sync
    case 3:
      // Si recibe un mensaje de sincronizacion del host ya registrado, lo ignora
      if(esp_now_is_peer_exist(mac->src_addr)) return;
      EEPROM.writeBytes(host_EEPROM_addr, mac->src_addr, 6); // Escribe el nuevo host en la EEPROM
      EEPROM.commit();
      esp_now_del_peer(host_addr); // Elimina el host anterior
      setHostMac(mac->src_addr); // Registra el nuevo host en ESPNOW
      memcpy(host_addr, mac->src_addr, 6); // Registra el nuevo host en la variable
      // desactiva el sync mode y el timer
      sync_mode = false;
      xTimerStop(deactivate_sync_timer, 0);
      break;
    // connect
    case 4:
      // Ignora si ya esta conectado
      if(connected) return;
      connected = true; // Registra la conexion
      xTimerStop(connect_timer, 0); // Detiene el timer de conexion
      xTimerStart(disconnect_timer, 0); // Empieza el timer de autodesconexion
      xTimerStart(alive_timer, 0); // Empieza el timer de mensajes alive
      // Revisa si el host esta leyendo datos, inicia la lectura si es asi
      if(incomingData[1] == 1){
        reading_sensor = true;
        xTimerStart(reading_sensor_timer, 0);
      }
      break;
    // still alive
    case 6:
      xTimerReset(disconnect_timer, 0); // Resetea el timer de desconexion
      break;
    // Idendificacion
    case 7:
      identify_mode = true;
      xTimerStart(deactivate_identify_timer, 0);
      break;
    // empezar a leer y enviar datos
    case 8:
      // Aumenta la frecuencia a 160MHz
      setCpuFrequencyMhz(160);
      reading_sensor = true;
      xTimerStart(reading_sensor_timer, 0);
      break;
    // dejar de leer y enviar datos
    case 9:
      // Reduce la frecuencia a 80MHz
      setCpuFrequencyMhz(80);
      reading_sensor = false;
      xTimerStop(reading_sensor_timer, 0);
      break;
    // reiniciar sensor
    case 10:
      // Crea la tarea para resetear el sensor
      xTaskCreate(
        [&mpu, &EEPROM, &reading_sensor, &offsets_EEPROM_addr, &reading_sensor_timer](void* pvParameters){
          int16_t offsets[6];
          // Si esta leyedo el sensor, detiene la lectira durante el reseteo
          if (reading_sensor)
            xTimerStop(reading_sensor_timer, 0);
          mpu.reset(); // resetea el sensor
          vTaskDelay(pdMS_TO_TICKS(50));
          mpu.init(); // re-inicializa el sensor
          // lee los offsets y los apliques
          EEPROM.readBytes(offsets_EEPROM_addr, offsets, sizeof(int16_t)*6);
          mpu.setOffsets(offsets);
          // reactiva la lectura
          if (reading_sensor)
            xTimerStart(reading_sensor_timer, 0);
          vTaskDelete(NULL);
        },
        "reset",
        1024,
        nullptr,
        1,
        nullptr
      );
      break;
    // calibrar sensor
    case 11:
      calibrate_sensor = true;
      // Crea la tarea para calibrar el sensor
      xTaskCreate(
        [&mpu, &EEPROM, &reading_sensor, &calibrate_sensor, &offsets_EEPROM_addr, &reading_sensor_timer](void* pvParameters){
          int16_t * _offsets;
          calibrate_sensor = true;
          // Detiene la lectura si esta activa
          if (reading_sensor)
            xTimerStop(reading_sensor_timer, 0);
          // Realiza la calibracion
          _offsets = mpu.calibrate();
          for(int i=0; i<6; i++){
            Serial.print(_offsets[i]);
            Serial.print(" ");
          }
          Serial.println();
          calibrate_sensor = false;

          // Reactiva la lectura
          if (reading_sensor)
            xTimerStart(reading_sensor_timer, 0);
          // Escribe los offsets en la EEPROM
          EEPROM.writeBytes(offsets_EEPROM_addr, _offsets, sizeof(int16_t)*6);
          EEPROM.commit();
          vTaskDelete(NULL);
        }, 
        "calibrate",
        1280,
        nullptr,
        1,
        nullptr
      );
      break;

    default:
      break;
  }
  
}

// Interrupcion para el boton
void btn_interrupt() {
    if(!digitalRead(GPIO_BUTTON)){
      xTimerStart(debouncing_timer, 0); // Si se presiona, activa el dobouncing
    } else {
      // si se libera detiene el debouncing y desactiva el timer para entrar en deepsleep
      xTimerStop(debouncing_timer, 0);
      xTimerStop(activate_deep_sleep_timer, 0);
    }
}

void setup() {
  pinMode(GPIO_SENSOR, OUTPUT);
  pinMode(GPIO_BUTTON, INPUT_PULLUP);
  pinMode(GPIO_LED_VCC, OUTPUT);
  pinMode(GPIO_BATTERY, INPUT);
  //activar el pin D0 como despertador del deep sleep
  esp_deep_sleep_enable_gpio_wakeup(BUTTON_PIN_BITMASK, ESP_GPIO_WAKEUP_GPIO_LOW);

  // filtro para encender el dispositivo
  uint64_t btn_pressed_millis = millis();
  while(millis()-btn_pressed_millis < 500){
    // Si el boton no 
    if(digitalRead(GPIO_BUTTON)){
      esp_deep_sleep_start();
    }
  }
  rtc_gpio_hold_dis(GPIO_NUM_6);
  // encender el led
  // Serial.begin(115200);
  digitalWrite(GPIO_LED_VCC, HIGH);
  digitalWrite(GPIO_SENSOR, HIGH);
  strip.begin();
  strip.setBrightness(30);
  strip.setPixelColor(0, Color::BLACK);
  strip.show();
  // algunos inits y configuraciones
  Wire.begin();
  EEPROM.begin(128);
  Wire.setClock(400000);
  WiFi.mode(WIFI_STA);
  while(!WiFi.STA.started()){
    vTaskDelay(100);
  }
  // configurar el wifi en el canal 1
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  //cargar los offsets del sensor
  vTaskDelay(pdMS_TO_TICKS(50));
  mpu.init();
  int16_t offsets[6];
  EEPROM.readBytes(offsets_EEPROM_addr, offsets, sizeof(int16_t)*6);
  mpu.setOffsets(offsets);
  // for(int i=0; i<6; i++){
  //   Serial.print(offsets[i]);
  //   Serial.print(' ');
  // }
  Serial.println();
  // inicializar y verificar que el esp now inicio correctamente
  if (esp_now_init() != ESP_OK) {
    esp_deep_sleep_start();
  }
  // Agregar host
  EEPROM.readBytes(host_EEPROM_addr, host_addr, 6);
  setHostMac(host_addr);
  // activar el callback de esp now on recv
  esp_now_register_recv_cb(OnDataRecv);
  // activar interrupcion para el boton
  attachInterrupt(digitalPinToInterrupt(GPIO_BUTTON), btn_interrupt, CHANGE);

  // timer para intentar connectar
  connect_timer = xTimerCreate(
    "connect", 
    pdMS_TO_TICKS(CONNECT_TRY_TIME), 
    pdTRUE, 
    nullptr, 
    [host_addr](TimerHandle_t xTimer){
      static uint8_t buffer = 0x04;
      esp_now_send(host_addr, &buffer, 1);
    }
  );
  // timer para desconectar si no se reciben mensajes alive con el timeout indicado
  disconnect_timer = xTimerCreate(
    "disconnect", 
    pdMS_TO_TICKS(ALIVE_TIMEOUT), 
    pdFALSE, 
    nullptr, 
    [&connected, &connect_timer, &alive_timer](TimerHandle_t xTimer){
      connected = false;
      xTimerStart(connect_timer, 0);
      xTimerStop(alive_timer, 0);
    }
  );
  // timer para enviar mensajes alive periodicamentes
  alive_timer = xTimerCreate(
    "alive",
    pdMS_TO_TICKS(ALIVE_MSG_FREQUENCY), 
    pdTRUE, 
    nullptr,
    [host_addr](TimerHandle_t xTimer){
      static uint8_t buffer = 0x06;
      esp_now_send(host_addr, &buffer, 1);
    }
  );
  // timer para activar el modo deep sleep, se activara si se mantiene un segundo el boton
  activate_deep_sleep_timer = xTimerCreate(
    "deep sleep",
    pdMS_TO_TICKS(1000),
    pdFALSE, 
    nullptr,
    [host_addr, &connect_timer, &disconnect_timer, &alive_timer, 
    &release_btn_timer, &release_btn_timer, &deactivate_sync_timer, &debouncing_timer, 
    &battery_check_timer, &deactivate_identify_timer, &reading_sensor_timer](TimerHandle_t xTimer){
      digitalWrite(GPIO_LED_VCC, LOW);
      digitalWrite(GPIO_SENSOR, LOW);
      xTimerDelete(connect_timer, 0);
      xTimerDelete(disconnect_timer, 0);
      xTimerDelete(alive_timer, 0);
      xTimerDelete(release_btn_timer, 0);
      xTimerDelete(release_btn_timer, 0);
      xTimerDelete(deactivate_sync_timer, 0);
      xTimerDelete(debouncing_timer, 0);
      xTimerDelete(battery_check_timer, 0);
      xTimerDelete(deactivate_identify_timer, 0);
      xTimerDelete(reading_sensor_timer, 0);
      static uint8_t buffer = 0x05;
      esp_now_send(host_addr, &buffer, sizeof(buffer));
      vTaskDelay(pdMS_TO_TICKS(200));
      esp_now_deinit();
      Wire.end();
      esp_wifi_stop();
      WiFi.mode(WIFI_OFF);
      vTaskDelay(pdMS_TO_TICKS(10));
      rtc_gpio_isolate(GPIO_NUM_6);
      esp_deep_sleep_start();
    }
  );
  // timer para contar cuantas veces se libera el boton para poder activar el modo sincronizacion
  release_btn_timer = xTimerCreate(
    "release btn",
    pdMS_TO_TICKS(400),
    pdFALSE,
    nullptr,
    [&release_count](TimerHandle_t xTimer){
      release_count = 0;
    }
  );
  // timer que desactiva el modo sincronizacion pasado el modo sincronizacion
  deactivate_sync_timer = xTimerCreate(
    "deactivate sync",
    pdMS_TO_TICKS(SYNC_TIME),
    pdFALSE,
    nullptr,
    [&sync_mode](TimerHandle_t xTimer){
      sync_mode = false;
    }
  );
  // timer de debouncion del boton
  debouncing_timer = xTimerCreate(
    "debouncing",
    pdMS_TO_TICKS(DEBOUNCING_TIME),
    pdFALSE,
    nullptr,
    [&release_count, &sync_mode, &activate_deep_sleep_timer, &release_btn_timer, &activate_deep_sleep_timer](TimerHandle_t xTimer){
      if(!digitalRead(GPIO_BUTTON)){
        xTimerStart(activate_deep_sleep_timer, 0);
        if(release_count)
          xTimerReset(release_btn_timer, 0);
        else
          xTimerStart(release_btn_timer, 0);
        release_count++;
        if(release_count==3){
          sync_mode = true;
          xTimerStart(deactivate_sync_timer, 0);
        }
      }
    }
  );
  // timer para indicar con el led que hay poca bateria
  low_battery_timer = xTimerCreate(
    "low battery",
    pdMS_TO_TICKS(1500),
    pdFALSE,
    nullptr,
    [&low_battery](TimerHandle_t xTimer){
      low_battery = false;
    }
  );
  // timer para revisar la bateria
  battery_check_timer = xTimerCreate(
    "battery",
    pdMS_TO_TICKS(BATTERY_FREQUENCY_TIME),
    pdTRUE,
    nullptr,
    [&low_battery, &low_battery_timer, &host_addr, &connected](TimerHandle_t xTimer){
      uint8_t charge = checkBattery();
      static uint8_t buffer[2];
      buffer[0] = 0x02;
      buffer[1] = charge;
      if(connected) 
        esp_now_send(host_addr, buffer, 2);

      if(charge <= 10){
        low_battery = true;
        xTimerStart(low_battery_timer, 0);
      }
      if(charge <= 0){
        xTimerStart(activate_deep_sleep_timer, 0);
      }
    }
  );
  // timer para desactivar el identificador
  deactivate_identify_timer = xTimerCreate(
    "deactivate identify",
    pdMS_TO_TICKS(IDENTIFY_TIME), 
    pdFALSE,
    nullptr,
    [&identify_mode](TimerHandle_t xTimer){
      identify_mode = false;
    }
  );
  // timer para leer el sensor y enviar los datos
  reading_sensor_timer = xTimerCreate(
    "read", 
    pdMS_TO_TICKS(DATA_FREQUENCY_TIME), 
    pdTRUE,
    nullptr,
    [&host_addr, &mpu](TimerHandle_t xTimer){
      static uint8_t buffer[sizeof(sensor_read)+1];
      sensor_read data;
      mpu.readSensor(&data);
      buffer[0] = 0x01;
      memcpy(&buffer[1], &data, sizeof(sensor_read));
      esp_now_send(host_addr, buffer, sizeof(buffer));
    }
  );
  uint8_t charge = checkBattery();
  if(charge<=10){
      strip.setPixelColor(0, Color::RED);
      strip.show();
      vTaskDelay(pdMS_TO_TICKS(1000));
  }
  if(charge <= 0){
    esp_deep_sleep_start();
  }
  xTimerStart(connect_timer, 0);
  xTimerStart(battery_check_timer, 0);
}
// en el loop solo se ejecuta el led
void loop() {
  static uint64_t led_millis = millis();
  if(millis()-led_millis >= 300){

    led_millis = millis();
    static bool glowing = false;
    glowing = !glowing;
    if(low_battery)
      strip.setPixelColor(0, glowing ? Color::RED : Color::BLACK);
    else if(sync_mode)
      strip.setPixelColor(0, glowing ? Color::MAGENTA : Color::BLACK);
    else if(!connected)
      strip.setPixelColor(0, Color::YELLOW);
    else if(identify_mode)
      strip.setPixelColor(0, glowing ? Color::VIOLET : Color::BLACK);
    else if(reading_sensor)
      strip.setPixelColor(0, glowing ? Color::GREEN : Color::BLACK);
    else if(calibrate_sensor)
      strip.setPixelColor(0, glowing ? Color::PINK : Color::BLACK);
    else if(connected)
      strip.setPixelColor(0, Color::GREEN);
    strip.show();
  }
}
// funcion para agregar el host
void setHostMac(const uint8_t *addr){
  static esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, addr, 6);
  peerInfo.channel = 1;
  peerInfo.encrypt = false;
  peerInfo.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&peerInfo);
  esp_now_rate_config_t config = {
    .phymode = WIFI_PHY_MODE_11G,
    .rate = WIFI_PHY_RATE_11M_L,
    .ersu = false,
    .dcm = false
  };
  esp_now_set_peer_rate_config(addr, &config);
  // if (esp_now_add_peer(&peerInfo) != ESP_OK){
  //   Serial.println("Error al agregar peer host");
  // } else {
  //   Serial.println("peer host agregado correctamente");
  // }
}


