/*
 * distraction_detector.ino
 * Real-time distraction detection on ESP32-S3 with camera.
 *
 * Hardware:
 *   - ESP32-S3 DevKitC or XIAO ESP32S3 Sense
 *   - OV2640 / OV5640 camera module
 *
 * Libraries (install via Arduino Library Manager):
 *   - TensorFlowLite_ESP32 (or EloquentTinyML)
 *   - ESP32 Camera driver (included with ESP32 board package)
 *
 * Setup:
 *   1. Run convert_to_tflite.py to generate model_data.h
 *   2. Select board: ESP32S3 Dev Module
 *   3. Set PSRAM: OPI PSRAM
 *   4. Set Flash Size: 8MB or 16MB
 *   5. Upload
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
#define TENSOR_ARENA_SIZE (512 * 1024)  // 512 KB for ESP32-S3 PSRAM

static const char* kLabels[NUM_CLASSES] = {"distracted", "empty", "focused"};

static const float kMean[3] = {0.485f, 0.456f, 0.406f};
static const float kStd[3]  = {0.229f, 0.224f, 0.225f};

// ── TFLite globals ──────────────────────────────────────────────────────────
static uint8_t* tensor_arena = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input_tensor = nullptr;
static TfLiteTensor* output_tensor = nullptr;

// ── Status LED ──────────────────────────────────────────────────────────────
#define LED_PIN 21  // Built-in LED on most ESP32-S3 boards

// ── Camera config (XIAO ESP32S3 Sense pins) ────────────────────────────────
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
// Resize RGB565 frame to 224x224 float tensor with ImageNet normalization
void preprocess_frame(camera_fb_t* fb, float* tensor_data) {
    int src_w = fb->width;
    int src_h = fb->height;
    uint16_t* pixels = (uint16_t*)fb->buf;

    for (int y = 0; y < INPUT_H; y++) {
        int src_y = y * src_h / INPUT_H;
        for (int x = 0; x < INPUT_W; x++) {
            int src_x = x * src_w / INPUT_W;
            uint16_t px = pixels[src_y * src_w + src_x];

            // RGB565 → float [0,1] → normalized
            float r = ((px >> 11) & 0x1F) / 31.0f;
            float g = ((px >> 5) & 0x3F)  / 63.0f;
            float b = (px & 0x1F)         / 31.0f;

            int idx = (y * INPUT_W + x) * INPUT_CH;
            tensor_data[idx + 0] = (r - kMean[0]) / kStd[0];
            tensor_data[idx + 1] = (g - kMean[1]) / kStd[1];
            tensor_data[idx + 2] = (b - kMean[2]) / kStd[2];
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
    Serial.println("\n=== Distraction Detector ===\n");

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Initialize camera
    setup_camera();

    // Allocate tensor arena in PSRAM if available
    if (psramFound()) {
        tensor_arena = (uint8_t*)ps_malloc(TENSOR_ARENA_SIZE);
        Serial.println("Using PSRAM for tensor arena.");
    } else {
        tensor_arena = (uint8_t*)malloc(TENSOR_ARENA_SIZE);
        Serial.println("WARNING: No PSRAM — using heap.");
    }

    if (!tensor_arena) {
        Serial.println("ERROR: Failed to allocate tensor arena.");
        while (1) delay(1000);
    }

    // Load TFLite model
    const tflite::Model* model = tflite::GetModel(g_distraction_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema mismatch: %d vs %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        while (1) delay(1000);
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: AllocateTensors() failed.");
        while (1) delay(1000);
    }

    input_tensor  = interpreter->input(0);
    output_tensor = interpreter->output(0);

    Serial.printf("Input:  %d x %d x %d x %d\n",
        input_tensor->dims->data[0], input_tensor->dims->data[1],
        input_tensor->dims->data[2], input_tensor->dims->data[3]);
    Serial.printf("Arena used: %zu / %d bytes\n",
        interpreter->arena_used_bytes(), TENSOR_ARENA_SIZE);
    Serial.println("\nRunning inference...\n");
}

// ── Main loop ───────────────────────────────────────────────────────────────
void loop() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed.");
        delay(100);
        return;
    }

    // Preprocess into input tensor
    float* input_data = input_tensor->data.f;
    preprocess_frame(fb, input_data);
    esp_camera_fb_return(fb);

    // Run inference
    unsigned long t0 = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke failed.");
        return;
    }
    unsigned long dt = millis() - t0;

    // Get results
    float* raw_output = output_tensor->data.f;
    float probs[NUM_CLASSES];
    softmax(raw_output, probs, NUM_CLASSES);

    // Find top prediction
    int best_idx = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (probs[i] > probs[best_idx]) best_idx = i;
    }

    // LED feedback: ON if distracted
    digitalWrite(LED_PIN, best_idx == 0 ? HIGH : LOW);

    // Print results
    Serial.printf("[%4lums] %-11s %.1f%%  |", dt, kLabels[best_idx], probs[best_idx] * 100);
    for (int i = 0; i < NUM_CLASSES; i++) {
        Serial.printf("  %s:%.0f%%", kLabels[i], probs[i] * 100);
    }
    Serial.println();

    delay(50);  // ~20 FPS target
}
