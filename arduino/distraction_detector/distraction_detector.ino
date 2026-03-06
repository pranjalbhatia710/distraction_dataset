/*
 * distraction_detector.ino
 * Real-time distraction detection on Arduino Nano ESP32 (ESP32-S3).
 *
 * Board: Arduino Nano ESP32
 *   - ESP32-S3 (8MB flash, 8MB PSRAM)
 *   - Install via Arduino Board Manager: "Arduino ESP32 Boards"
 *   - Select board: "Arduino Nano ESP32"
 *   - Set USB Mode: "Hardware CDC and JTAG"
 *
 * NOTE: MobileNetV3-Small (~1.6MB) does NOT fit on Nano 33 BLE (1MB flash).
 *       Use Arduino Nano ESP32 or ESP32-S3 DevKitC instead.
 *       For classic Nanos, see distraction_serial_bridge.ino.
 *
 * Camera: OV2640 via external module (connect to GPIO pins below)
 *
 * Libraries (install via Arduino Library Manager):
 *   - TensorFlowLite_ESP32
 *
 * Setup:
 *   1. Run: python convert_to_tflite.py --model best_distraction_model.pth
 *   2. model_data.h is auto-generated in this folder
 *   3. Upload to Arduino Nano ESP32
 */

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_camera.h"
#include "model_data.h"
#include "camera_pins.h"

// ── Model config ────────────────────────────────────────────────────────────
#define INPUT_W       224
#define INPUT_H       224
#define INPUT_CH      3
#define NUM_CLASSES   3
#define TENSOR_ARENA_SIZE (512 * 1024)

static const char* kLabels[NUM_CLASSES] = {"distracted", "empty", "focused"};

// ImageNet normalization
static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

// ── TFLite globals ──────────────────────────────────────────────────────────
static uint8_t* tensor_arena = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ── Pins ────────────────────────────────────────────────────────────────────
#define LED_RED    D2   // Distracted indicator
#define LED_GREEN  D3   // Focused indicator
#define BUZZER_PIN D4   // Optional buzzer

// ── Camera setup ────────────────────────────────────────────────────────────
void setup_camera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = CAMERA_PIN_D0;
    config.pin_d1       = CAMERA_PIN_D1;
    config.pin_d2       = CAMERA_PIN_D2;
    config.pin_d3       = CAMERA_PIN_D3;
    config.pin_d4       = CAMERA_PIN_D4;
    config.pin_d5       = CAMERA_PIN_D5;
    config.pin_d6       = CAMERA_PIN_D6;
    config.pin_d7       = CAMERA_PIN_D7;
    config.pin_xclk     = CAMERA_PIN_XCLK;
    config.pin_pclk     = CAMERA_PIN_PCLK;
    config.pin_vsync    = CAMERA_PIN_VSYNC;
    config.pin_href     = CAMERA_PIN_HREF;
    config.pin_sccb_sda = CAMERA_PIN_SIOD;
    config.pin_sccb_scl = CAMERA_PIN_SIOC;
    config.pin_pwdn     = CAMERA_PIN_PWDN;
    config.pin_reset    = CAMERA_PIN_RESET;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size   = FRAMESIZE_240X240;
    config.fb_count     = 1;
    config.grab_mode    = CAMERA_GRAB_LATEST;

    if (psramFound()) {
        config.fb_location = CAMERA_FB_IN_PSRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        while (1) delay(1000);
    }
    Serial.println("Camera ready.");
}

// ── Preprocessing ───────────────────────────────────────────────────────────
// Resize RGB565 frame → NCHW float tensor with ImageNet normalization
void preprocess_frame(camera_fb_t* fb, float* tensor_data) {
    int src_w = fb->width;
    int src_h = fb->height;
    uint16_t* pixels = (uint16_t*)fb->buf;
    int plane_size = INPUT_H * INPUT_W;

    for (int y = 0; y < INPUT_H; y++) {
        int src_y = y * src_h / INPUT_H;
        for (int x = 0; x < INPUT_W; x++) {
            int src_x = x * src_w / INPUT_W;
            uint16_t px = pixels[src_y * src_w + src_x];

            float r = ((px >> 11) & 0x1F) / 31.0f;
            float g = ((px >> 5) & 0x3F)  / 63.0f;
            float b = (px & 0x1F)         / 31.0f;

            int pixel_idx = y * INPUT_W + x;
            // NCHW layout: channel planes are contiguous
            tensor_data[0 * plane_size + pixel_idx] = (r - kMean[0]) / kStd[0];
            tensor_data[1 * plane_size + pixel_idx] = (g - kMean[1]) / kStd[1];
            tensor_data[2 * plane_size + pixel_idx] = (b - kMean[2]) / kStd[2];
        }
    }
}

// ── Softmax ─────────────────────────────────────────────────────────────────
void softmax(float* input, float* output, int len) {
    float max_val = input[0];
    for (int i = 1; i < len; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    float sum = 0;
    for (int i = 0; i < len; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < len; i++) {
        output[i] /= sum;
    }
}

// ── Setup ───────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);
    Serial.println("\n=== Distraction Detector (Nano ESP32) ===\n");

    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(LED_RED, LOW);
    digitalWrite(LED_GREEN, LOW);

    setup_camera();

    // Allocate tensor arena in PSRAM
    if (psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
        Serial.printf("PSRAM: %d KB free\n", ESP.getFreePsram() / 1024);
    } else {
        tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
        Serial.println("WARNING: No PSRAM");
    }

    if (!tensor_arena) {
        Serial.println("ERROR: Failed to allocate tensor arena");
        while (1) delay(1000);
    }

    const tflite::Model* model = tflite::GetModel(g_distraction_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Schema mismatch: %d vs %d\n", model->version(), TFLITE_SCHEMA_VERSION);
        while (1) delay(1000);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed");
        while (1) delay(1000);
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("Arena: %zu / %d bytes\n", interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
    Serial.println("Running...\n");
}

// ── Main loop ───────────────────────────────────────────────────────────────
void loop() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Capture failed");
        delay(100);
        return;
    }

    float* input_data = input_tensor->data.f;
    preprocess_frame(fb, input_data);
    esp_camera_fb_return(fb);

    unsigned long t0 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed");
        return;
    }
    unsigned long dt = millis() - t0;

    float* raw_output = output_tensor->data.f;
    float probs[NUM_CLASSES];
    softmax(raw_output, probs, NUM_CLASSES);

    int best_idx = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > probs[best_idx]) best_idx = i;
    }

    // LED + buzzer feedback
    digitalWrite(LED_RED, best_idx == 0 ? HIGH : LOW);    // distracted
    digitalWrite(LED_GREEN, best_idx == 2 ? HIGH : LOW);  // focused
    digitalWrite(BUZZER_PIN, best_idx == 0 ? HIGH : LOW); // buzz when distracted

    // Serial output
    Serial.printf("[%4lums] %-11s %.1f%%", dt, kLabels[best_idx], probs[best_idx] * 100);
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.printf("  %s:%.0f%%", kLabels[i], probs[i] * 100);
    }
    Serial.println();

    delay(50);
}
