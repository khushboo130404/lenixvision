// Complete ESP sketch: Camera + continuous microphone -> WebSocket streaming
// For: Seeed XIAO ESP32S3 Sense (on-board PDM mic). Uses WebSocket binary frames for audio.

// Libraries
#include "esp_camera.h"
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <I2S.h>              // Seeed / Arduino I2S helper
#include "board_config.h"     // your camera board config header (already in your project)

// ---------- CONFIG ----------
const char* ssid = "Airtel_HarryPotter";
const char* password = "Khanna19@";

#define WS_SERVER_IP   "192.168.1.7"   // <-- change to your Flask server IP
#define WS_SERVER_PORT 5000              // <-- change to your Flask WS port
#define WS_PATH        "/ws-audio"       // <-- endpoint path (ws://IP:PORT/ws-audio)

#define SAMPLE_RATE    16000U            // 16 kHz (Seeed recommended stable rate)
#define SAMPLE_BITS    16                // 16-bit samples from PDM conversion
#define AUDIO_CHUNK_SAMPLES 512          // number of 16-bit samples per websocket frame

// Seeed XIAO Sense PDM pins: CLK = GPIO42, DATA = GPIO41. (See Seeed wiki.)
//
// (If you later change hardware, update these pin choices accordingly.)
#define PDM_CLK_PIN 42
#define PDM_DATA_PIN 41

// ---------- GLOBALS ----------
WebSocketsClient webSocket;
volatile bool wsConnected = false;

// Forward declarations (camera server provided by your existing framework)
void startCameraServer();
void setupLedFlash();

// ---------- I2S (PDM) AUDIO HELPERS ----------
void setupPdmMic() {
  // Seeed's recommended API for XIAO ESP32S3 Sense (PDMRX)
  // Set pins for PDM RX: CLK = 42, DATA = 41
  // This uses the Arduino/Seeed I2S wrapper
  I2S.setPinsPdmRx(PDM_CLK_PIN, PDM_DATA_PIN);
  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, SAMPLE_BITS, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S (PDM) microphone!");
    while (1) delay(1000);
  }
  Serial.println("PDM microphone initialized.");
}

// Read AUDIO_CHUNK_SAMPLES samples into buffer (int16_t)
void readAudioChunk(int16_t* buf, size_t samples) {
  size_t idx = 0;
  // The Seeed I2S wrapper exposes I2S.read() returning an int sample.
  // We'll read samples one-by-one and pack into buf.
  while (idx < samples) {
    int s = I2S.read(); // returns raw sample (signed int)
    if (s == 0 || s == -1 || s == 1) {
      // noise or invalid; skip small non-values but allow loop to continue
      continue;
    }
    // The sample returned is already 16-bit style for PDM mode in Seeed's wrapper.
    buf[idx++] = (int16_t)s;
  }
}

// ---------- WEBSOCKET HANDLERS ----------
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
  switch(type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.println("WebSocket connected.");
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("WebSocket disconnected.");
      break;
    case WStype_TEXT:
      // server may send control messages
      Serial.printf("WS text: %s\n", (char*)payload);
      break;
    case WStype_BIN:
      // not expecting binary from server
      break;
    default:
      break;
  }
}

void setupWebSocket() {
  // connect to ws://WS_SERVER_IP:WS_SERVER_PORT/WS_PATH
  webSocket.begin(WS_SERVER_IP, WS_SERVER_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  // optional: tryPing/pong
  webSocket.setReconnectInterval(5000); // try every 5s
}

// ---------- CAMERA (your existing camera init) ----------
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 8000000;
  config.frame_size = FRAMESIZE_UXGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (config.pixel_format == PIXFORMAT_JPEG) {
    if (psramFound()) {
      config.jpeg_quality = 10;
      config.fb_count = 2;
      config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
      config.frame_size = FRAMESIZE_QVGA;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  } else {
    config.frame_size = FRAMESIZE_QVGA;
#if CONFIG_IDF_TARGET_ESP32S3
    config.fb_count = 2;
#endif
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  if (config.pixel_format == PIXFORMAT_JPEG) {
    s->set_framesize(s, FRAMESIZE_QVGA);
  }

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif
}

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
  Serial.println("Booting...");

  // Initialize camera (your provided function)
  initCamera();

  // Setup LED flash if defined
  #if defined(LED_GPIO_NUM)
    setupLedFlash();
  #endif

  // Connect WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());

  // start camera server (your function)
  startCameraServer();

  // Initialize PDM microphone
  setupPdmMic();

  // Start WebSocket client
  setupWebSocket();

  Serial.println("Setup complete. Camera and mic streaming active.");
  Serial.printf("Connect to camera at http://%s\n", WiFi.localIP().toString().c_str());
}

// ---------- MAIN LOOP ----------
void loop() {
  // Maintain websocket connection
  webSocket.loop();

  // If ws connected, read a chunk of audio and send
  if (wsConnected) {
    static int16_t audioBuf[AUDIO_CHUNK_SAMPLES];
    readAudioChunk(audioBuf, AUDIO_CHUNK_SAMPLES);

    // Send raw PCM16 LE binary over WebSocket
    // The server must expect samples as 16-bit little-endian PCM at SAMPLE_RATE Hz
    webSocket.sendBIN((uint8_t*)audioBuf, AUDIO_CHUNK_SAMPLES * sizeof(int16_t));
    // very small delay to let other tasks run (camera server, wifi stack)
    delay(2);
  } else {
    // not connected: attempt small maintenance or delay
    delay(50);
  }

  // loop continues, camera server runs in its own task
}
