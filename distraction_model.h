/*
 * distraction_model.h
 * Distraction Detection Model — Arduino Configuration Header
 *
 * Model:    MobileNetV3-Small (transfer learning)
 * Classes:  distracted, empty, focused
 * Input:    224x224 RGB
 * Backend:  TensorFlow Lite Micro
 *
 * Usage:
 *   #include "distraction_model.h"
 *   // model data is in g_distraction_model[]
 *   // use kNumClasses, kClassLabels, etc. for inference post-processing
 */

#ifndef DISTRACTION_MODEL_H
#define DISTRACTION_MODEL_H

// ── Model metadata ──────────────────────────────────────────────────────────
#define MODEL_INPUT_WIDTH   224
#define MODEL_INPUT_HEIGHT  224
#define MODEL_INPUT_CHANNELS 3

// ImageNet normalization (used during training)
static const float kMeanRGB[3] = {0.485f, 0.456f, 0.406f};
static const float kStdRGB[3]  = {0.229f, 0.224f, 0.225f};

// ── Classes ─────────────────────────────────────────────────────────────────
#define NUM_CLASSES 3

static const char* const kClassLabels[NUM_CLASSES] = {
    "distracted",
    "empty",
    "focused"
};

// Confidence threshold for a valid prediction
#define CONFIDENCE_THRESHOLD 0.6f

// ── TFLite model data ───────────────────────────────────────────────────────
// To generate: convert .pth → ONNX → TFLite → xxd -i model.tflite
//
// Example conversion pipeline:
//   1. torch.onnx.export(model, dummy, "model.onnx")
//   2. onnx2tf -i model.onnx -o tflite_out
//   3. xxd -i tflite_out/model.tflite > model_data.h
//
// Then paste the array below:
//
// alignas(8) const unsigned char g_distraction_model[] = {
//     0x20, 0x00, 0x00, 0x00, ...
// };
// const unsigned int g_distraction_model_len = sizeof(g_distraction_model);

// ── Arena size (tune per board) ─────────────────────────────────────────────
// Arduino Nano 33 BLE Sense: ~150 KB available
// ESP32-S3: ~512 KB available
#define TENSOR_ARENA_SIZE (150 * 1024)

#endif // DISTRACTION_MODEL_H
