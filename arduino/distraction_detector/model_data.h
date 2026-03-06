/*
 * model_data.h
 * PLACEHOLDER — Replace with actual model data.
 *
 * Generate this file by running:
 *   python convert_to_tflite.py --model best_distraction_model.pth
 *
 * Or with int8 quantization (recommended for ESP32):
 *   python convert_to_tflite.py --model best_distraction_model.pth --quantize
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <cstdint>

// TODO: Run convert_to_tflite.py to fill this array with actual model weights.
// This placeholder will cause a build error as a reminder.
#error "model_data.h: Run convert_to_tflite.py to generate the model data array."

alignas(8) const unsigned char g_distraction_model[] = {
    0x00
};

const unsigned int g_distraction_model_len = 0;

#endif // MODEL_DATA_H
