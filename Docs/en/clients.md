### Code Documentation

The code presented below is an example implementation for an ESP32 board, which uses different modules and libraries to perform wireless communication tasks, sensory data acquisition, battery management, and LED control. The code documentation is detailed below:

#### **1. Library Inclusions**
The code includes the following libraries:

- `esp_now.h`: For peer-to-peer wireless communication.
- `WiFi.h` and `esp_wifi.h`: For configuration and management of the Wi-Fi module.
- `I2Cdev.h` and `MPU6050_6Axis_MotionApps612.h`: To interact with the MPU6050 sensor.
- `Wire.h`: For I2C communication.
- `Adafruit_NeoPixel.h`: To control NeoPixel LEDs.
- `EEPROM.h`: For non-volatile data storage.
- `Arduino.h`: Basic Arduino functions.
- `FreeRTOS.h` and `timers.h`: For real-time multitasking programming.


Note: Version 1.4.4 of the ElectronicsCats mpu6050 library may cause issues when calibrating, I recommend using version 1.4.0.

#### **2. Definitions and Configurations**

##### **GPIO Definitions**
```cpp
#define GPIO_SENSOR GPIO_NUM_18 //D10
#define GPIO_SENSOR_INT GPIO_NUM_19 //D8
#define GPIO_BUTTON GPIO_NUM_0 //D0
#define GPIO_LED_VCC GPIO_NUM_21 //D3
#define GPIO_LED_DATA GPIO_NUM_2 //D2
#define GPIO_BATTERY GPIO_NUM_6
```
- `GPIO_SENSOR`: Pin used for the MPU6050 sensor.
- `GPIO_SENSOR_INT`: Sensor interrupt pin.
- `GPIO_BUTTON`: Control button pin.
- `GPIO_LED_VCC`: Power pin for the NeoPixel LED.
- `GPIO_LED_DATA`: Data pin for the NeoPixel LED.
- `GPIO_BATTERY`: Pin to measure battery voltage.

##### **Time Constants**
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
- `SYNC_TIME`: Synchronization time (20 seconds).
- `IDENTIFY_TIME`: Duration of identification mode (10 seconds).
- `DATA_FREQUENCY_TIME`: Frequency of reading and sending sensory data (10 ms).
- `BATTERY_FREQUENCY_TIME`: Battery check frequency (1 second).
- `CONNECT_TRY_TIME`: Time to attempt connection (1 second).
- `ALIVE_TIMEOUT`: Timeout for "alive" messages (3 seconds).
- `ALIVE_MSG_FREQUENCY`: Frequency of sending "alive" messages (1 second).
- `DEBOUNCING_TIME`: Button debounce time (5 ms).

##### **Color Enum**
```cpp
enum Color {
  BLACK, WHITE, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE, PURPLE,
  LIME, TEAL, PINK, BROWN, GRAY, DARK_GRAY, LIGHT_GRAY, GOLD, SILVER,
  NAVY_BLUE, MAROON, OLIVE, SKY_BLUE, TURQUOISE, VIOLET, INDIGO, BEIGE,
  MINT, LAVENDER
};
```
- Defines colors for the NeoPixel LED.

#### **3. Structures and Classes**

##### **sensor_read Structure**
```cpp
struct sensor_read {
  Quaternion q;
  VectorInt16 a;
  VectorInt16 g;
};
```
- Stores MPU6050 sensor data: quaternion, acceleration, and gyroscope.

##### **SENSOR Class**
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
- Child class of `MPU6050` to simplify sensor reading and configuration.

#### **4. Global Variables**
```cpp
Adafruit_NeoPixel strip(1, GPIO_LED_DATA, NEO_GRB);
SENSOR mpu;
uint8_t host_addr[6];
int host_EEPROM_addr = 0;
int offsets_EEPROM_addr = 10;
// ...other timers and flags...
```
- `strip`: Object to control the NeoPixel LED.
- `mpu`: Instance of the SENSOR class to interact with the MPU6050.
- `host_addr`: MAC address of the host receiver.
- `host_EEPROM_addr`: EEPROM address to store the host MAC address.
- `offsets_EEPROM_addr`: EEPROM address to store calibration offsets.

#### **5. Main Functions**

##### **checkBattery Function**
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
- Reads the battery voltage and converts it to a percentage.

##### **OnDataRecv Function**
```cpp
void OnDataRecv(const esp_now_recv_info_t * mac, const uint8_t *incomingData, int len) {
    // Handling of different received commands...
}
```
- Callback function to handle data received via ESP-NOW.

##### **btn_interrupt Function**
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
- Handles button interrupts for debounce and mode activation.

##### **setup Function**
```cpp
void setup() {
    // Initial configurations...
    // Initialization of pins, sensors, timers, etc.
}
```
- Configures all components: GPIO, MPU6050 sensor, ESP-NOW, timers, and LED.

##### **loop Function**
```cpp
void loop() {
    // Updates LED according to system state...
}
```
- Main function that updates the LEDs according to the system state.

#### **6. ESP-NOW Configuration**

##### **setHostMac Function**
```cpp
void setHostMac(const uint8_t *addr){
    // Configures and adds a peer in ESP-NOW...
}
```
- Configures the host receiver MAC address and establishes communication parameters.

#### **7. Timers and Tasks**

The code uses timers to handle different tasks, such as:

- Connection attempts (`connect_timer`).
- Disconnection timeout (`disconnect_timer`).
- Sending "alive" messages (`alive_timer`).
- Synchronization mode (`deactivate_sync_timer`).
- Counting button presses (`release_btn_timer`).
- Battery check (`battery_check_timer`).
- Sensor data reading and sending (`reading_sensor_timer`).

#### **8. Special Modes**

- **Synchronization Mode**: Allows establishing a new host MAC address.
- **Identification Mode**: LED blinking to identify the device.
- **Deep Sleep Mode**: Low power consumption when not in use.

#### **9. Controls and State**

The system handles different states:

- Connection established (`connected`).
- Active sensor reading (`reading_sensor`).
- Synchronization mode (`sync_mode`).
- Low battery (`low_battery`).
