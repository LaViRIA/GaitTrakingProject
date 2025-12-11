#include "ESP32_NOW.h"
#include <WiFi.h>
#include "MPU6050_6Axis_MotionApps612.h"
#include "helper_3dmath.h"
#include <Arduino.h>
#include <freertos/FreeRTOS.h>
#include <freertos/timers.h>

#define ALIVE_MSG_FREQUENCY 1000 //ms
#define INACTIVE_TIMEOUT 3000 //ms
#define SYNC_TIME 10000
#define SYNC_MSG_FREQUENCY 1000
#define BATTERY_MSG_FREQUENCY 1000
#define DATA_MSG_FREQUENCY 10

struct sensor_data{
  Quaternion q = {1.0, 0.0, 0.0, 0.0};
  VectorInt16 a = {0,0,0};
  VectorInt16 g = {0,0,0};
};

struct peer_data {
  sensor_data data;
  uint8_t battery;
  TimerHandle_t disconnect_timer;
};

uint8_t local_mac[6];

TimerHandle_t disable_sync_timer;
TimerHandle_t send_sync_timer;
TimerHandle_t alive_timer;
TimerHandle_t send_data_timer;
TimerHandle_t send_battery_timer;
bool reading = false;

// Callback cuando se reciben datos
void OnDataRecv(const esp_now_recv_info_t * mac, const uint8_t *incomingData, int len) {
  // unsigned long t = micros();
  if(memcmp(mac->des_addr, local_mac, 6) != 0) return;
  static esp_now_peer_info_t *peer;
  static peer_data *data;
  static byte buffer[2];
  
  if(!esp_now_is_peer_exist(mac->src_addr) && incomingData[0] != 4) return;
  switch (incomingData[0]) {
    //data
    case 1:
      peer = new esp_now_peer_info_t;
      if(esp_now_get_peer(mac->src_addr, peer) != ESP_OK) return; // obtiene el peer de la lista de peer, termina si hay un error
      data = (peer_data*)peer->priv; // obtiene los datos del peer
      memcpy(&data->data, &incomingData[1], sizeof(sensor_data)); // copia los datos entrantes en los datos del peer
      xTimerReset(data->disconnect_timer, 0); // resetea el timer de desconexion
      delete peer; // elimina la memoria del peer
      /*
      Eliminar la memoria del peer no elimina el peer ya que es una copia del almacenado espnow
      sin embargo si que se modifica ya que lo almacenado en el peer.priv es un apuntador a memoria dinamica
      por lo que si se modifica aqui se modifica en lo almacenado. 
      */
      break;
    // battery
    case 2:
      peer = new esp_now_peer_info_t;
      if(esp_now_get_peer(mac->src_addr, peer) != ESP_OK) return; // obtiene el peer de la lista de peer, termina si hay un error
      data = (peer_data*)peer->priv; // obtiene los datos del peer
      data->battery = incomingData[1]; // copia los datos entrantes en los datos del peer
      xTimerReset(data->disconnect_timer, 0); // resetea el timer de desconexion
      delete peer;  
      break;
    // solicitud de coneccion
    case 4:
      // Establece el buffer, primer byte en 0x04 para indicar paquete de conexion
      buffer[0] = 0x04;
      buffer[1] = reading ? 0x01 : 0x00; // segundo byte para indicar si se esta leyendo o no
      // revisa si el peer existe, si existe envia el buffer creado ya que si bien el host lo tiene registrado
      // el cliente perdio esa informacion
      if(esp_now_is_peer_exist(mac->src_addr)){
        esp_now_send(mac->src_addr, buffer, sizeof(buffer));
        return;
      }
      // Si el 'alive_timer' esta desactivado (porque no hay clientes conectados) lo activa)
      if(xTimerIsTimerActive(alive_timer)==pdFALSE)
        xTimerStart(alive_timer, pdMS_TO_TICKS(10));
      // Crea un nuevo peer, con su data
      peer = new esp_now_peer_info_t;
      data = new peer_data;
      // crea el timer de auto-desconexion para el peer
      data->disconnect_timer = xTimerCreate(
        "disconnect",
        pdMS_TO_TICKS(INACTIVE_TIMEOUT),
        pdFALSE,
        (void*)peer,
        [](TimerHandle_t xTimer){
          esp_now_peer_info_t *peer = (esp_now_peer_info_t*)pvTimerGetTimerID(xTimer);
          delete (peer_data*)peer->priv;
          esp_now_del_peer(peer->peer_addr);
          delete peer;
        }
      );
      // configura el peer
      memcpy(peer->peer_addr, mac->src_addr, 6);
      peer->channel = 1;
      peer->encrypt = false;
      peer->ifidx = WIFI_IF_STA;
      peer->priv = (void*)data;
      // regristra el peer y envia el buffer
      esp_now_add_peer(peer);
      esp_now_send(mac->src_addr, buffer, sizeof(buffer));
      break;
    // aviso de desconeccion
    case 5:
      // desconecta el peer y elimina todo lo que tiene que eliminar
      peer = new esp_now_peer_info_t;
      if(esp_now_get_peer(mac->src_addr, peer) != ESP_OK) return;
      data = (peer_data*)peer->priv;
      xTimerChangePeriod(data->disconnect_timer, pdMS_TO_TICKS(1), 0);
      // Serial.println("peer eliminado");
      delete peer;
      break;
    // still alive
    case 6:
      // obtiene el peer y el timer de autodesconexion para resetearlo
      peer = new esp_now_peer_info_t;
      if(esp_now_get_peer(mac->src_addr, peer) != ESP_OK) return;
      data = (peer_data*)peer->priv;
      xTimerReset(data->disconnect_timer, 0);
      delete peer;
      break;
    default:
      break;
  }
  // Serial.print(" t: ");
  // Serial.println(micros()-t);
}

void setup() {
  Serial.begin(250000);
  // Inicializar WiFi
  WiFi.mode(WIFI_STA);
  while(!WiFi.STA.started()){
    vTaskDelay(100);
  }
  WiFi.macAddress(local_mac);

  // Inicializar ESP-NOW
  if (esp_now_init() != ESP_OK) return;
  // Registrar callback para recibir datos
  esp_now_register_recv_cb(OnDataRecv);
  static esp_now_peer_info_t peer;
  uint8_t broadcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  memcpy(peer.peer_addr, broadcast, 6);
  peer.channel = 1;
  peer.encrypt = false;
  peer.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&peer);
  // timer para el modo sincronizacion
  send_sync_timer = xTimerCreate(
    "send sync",
    pdMS_TO_TICKS(SYNC_MSG_FREQUENCY),
    pdTRUE,
    nullptr,
    [](TimerHandle_t xTimer){
      uint8_t broadcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
      static uint8_t buffer = 0x03;
      esp_now_send(broadcast, &buffer, 1);
    }
  );
  // timer para desactivar el modo sincronizacion
  disable_sync_timer = xTimerCreate(
    "disable sync",
    pdMS_TO_TICKS(SYNC_TIME),
    pdFALSE,
    (void*)&send_sync_timer,
    [](TimerHandle_t xTimer){
      TimerHandle_t *timer = (TimerHandle_t*)pvTimerGetTimerID(xTimer);
      xTimerStop(*timer, 0);
    }
  );
  // timer para enviar los mensajes alive
  alive_timer = xTimerCreate(
    "alive",
    pdMS_TO_TICKS(ALIVE_MSG_FREQUENCY),
    pdTRUE, 
    (void*)&alive_timer,
    [](TimerHandle_t xTimer){
      TimerHandle_t *timer = (TimerHandle_t*)pvTimerGetTimerID(xTimer);
      static esp_now_peer_num_t peers;
      esp_now_get_peer_num(&peers);
      // si el hay un peer (si no hay clientes solo deberia de estar registrado el peer broadcast)
      // se detiene a si mismo
      if(peers.total_num == 1){
        xTimerStop(*timer, 0);
        return;
      }
      uint8_t buffer = 0x06;
      esp_now_send(NULL, &buffer, 1);
    }
  );
  // timer para enviar datos por serial
  send_data_timer = xTimerCreate(
    "data",
    pdMS_TO_TICKS(DATA_MSG_FREQUENCY),
    pdTRUE,
    nullptr,
    [](TimerHandle_t xTimer){
      static esp_now_peer_num_t peer_num;
      esp_now_get_peer_num(&peer_num); // lee el numero de peers
      uint8_t peer_count = max(peer_num.total_num-1, 0); // max solo para mantenerse >=0, -1 para no contar el broadcast
      if(peer_count==0) return; // si no hay clientes no envia nada
      static esp_now_peer_info_t peer;
      static peer_data *data;
      // packet num + timestamp + sensor count
      static uint header_size = sizeof(uint8_t)+sizeof(uint64_t)+sizeof(uint8_t);
      // mac address + sensor_data
      static uint sensor_size = (sizeof(uint8_t[6])+sizeof(sensor_data));
      // header size + (sensor size) * peer count
      uint buffer_size = header_size+(sensor_size*peer_count);
      uint8_t buffer[buffer_size];
      uint64_t timestamp = millis();
      buffer[0] = 0x01; 
      memcpy(&buffer[1], &peer_count, sizeof(peer_count));
      memcpy(&buffer[2], &timestamp, sizeof(timestamp));
      static uint buffer_idx;
      uint8_t peer_index=0;
      bool from_head = true;
      while(esp_now_fetch_peer(from_head, &peer)==ESP_OK){
        from_head = false;
        data = (peer_data *)peer.priv;
        buffer_idx = header_size + peer_index*(sensor_size);
        memcpy(&buffer[buffer_idx], peer.peer_addr, sizeof(peer.peer_addr));
        memcpy(&buffer[buffer_idx+6], &data->data, sizeof(sensor_data));
        peer_index++;
      }
      Serial.write("S");
      Serial.write(buffer, buffer_size);
    }
  );
  // timer para enviar la carga de los clientes por serial
  // este timer envia algo aunque no haya peers para poder indicar que no hay peers conectados
  send_battery_timer = xTimerCreate(
    "battery",
    pdMS_TO_TICKS(BATTERY_MSG_FREQUENCY),
    pdTRUE,
    nullptr,
    [](TimerHandle_t xTimer){
      static esp_now_peer_num_t peer_num;
      esp_now_get_peer_num(&peer_num);
      uint8_t peer_count = max(peer_num.total_num-1, 0);
      static esp_now_peer_info_t peer;
      static peer_data *data;
      // packet num + sensor count
      static uint header_size = sizeof(uint8_t)+sizeof(uint8_t);
      // mac address + charge
      static uint sensor_size = (sizeof(uint8_t[6])+sizeof(uint8_t));
      // header size + (sensor size) * peer count
      uint buffer_size = header_size+(sensor_size*peer_count);
      uint8_t buffer[buffer_size];
      buffer[0] = 0x02; 
      memcpy(&buffer[header_size-1], &peer_count, sizeof(peer_count));
      if (peer_count){
        static uint buffer_idx;
        uint8_t peer_index=0;
        bool from_head = true;
        while(esp_now_fetch_peer(from_head, &peer)==ESP_OK){
          from_head = false;
          data = (peer_data *)peer.priv;
          buffer_idx = header_size + peer_index*(sensor_size);
          memcpy(&buffer[buffer_idx], peer.peer_addr, sizeof(peer.peer_addr));
          memcpy(&buffer[buffer_idx+6], &data->battery, sizeof(uint8_t));
          peer_index++;
        }
      }
      Serial.write("S");
      Serial.write(buffer, buffer_size);
    }
  );
  xTimerStart(send_battery_timer, 0);
}

void loop() {
}
// funcion callback de la comunicacion serial, se activa al recibir algo por el puerto serial
void serialEvent(){
  static uint8_t buffer[7]; // los mensajes no deberian superar los 7 bytes (1B tipo +  6B address)
  Serial.readBytes(buffer, 7); // lee el serial
  // paquete de sincronizacion, activa el modo sincronizar
  if(buffer[0] == 3){
    xTimerStart(send_sync_timer, 0);
    xTimerStart(disable_sync_timer, 0);
  // paquetes entre 7 y 11, en escencia hacen lo mismo, solo envian el mensaje a la direccion dada
  } else if(buffer[0] >= 7 && buffer[0] <= 11){
    uint8_t broadcast[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    if(!memcmp(&buffer[1], broadcast, 6)) // verifica si la direccion dada es la del broadcast y decide a quien enviarlo
      esp_now_send(NULL, &buffer[0], 1);
    else
      esp_now_send(&buffer[1], &buffer[0], 1);
    // si es del tipo 8 o 9, inicia o detiene el estado de lectura
    if(buffer[0]==8){
      reading = true;
      xTimerStart(send_data_timer, 0);
    }
    if(buffer[0]==9){
      reading = false;
      xTimerStop(send_data_timer, 0);
    }
  }
}

