/*
 * camera_pins.h
 * Pin definitions for common ESP32-S3 camera boards.
 * Uncomment the section matching your board.
 */

#ifndef CAMERA_PINS_H
#define CAMERA_PINS_H

// ── XIAO ESP32S3 Sense ─────────────────────────────────────────────────────
#define CAMERA_PIN_PWDN   -1
#define CAMERA_PIN_RESET  -1
#define CAMERA_PIN_XCLK   10
#define CAMERA_PIN_SIOD   40
#define CAMERA_PIN_SIOC   39
#define CAMERA_PIN_D7     48
#define CAMERA_PIN_D6     11
#define CAMERA_PIN_D5     12
#define CAMERA_PIN_D4     14
#define CAMERA_PIN_D3     16
#define CAMERA_PIN_D2     18
#define CAMERA_PIN_D1     17
#define CAMERA_PIN_D0     15
#define CAMERA_PIN_VSYNC  38
#define CAMERA_PIN_HREF   47
#define CAMERA_PIN_PCLK   13

/*
// ── ESP32-S3-EYE ───────────────────────────────────────────────────────────
#define CAMERA_PIN_PWDN   -1
#define CAMERA_PIN_RESET  -1
#define CAMERA_PIN_XCLK   15
#define CAMERA_PIN_SIOD    4
#define CAMERA_PIN_SIOC    5
#define CAMERA_PIN_D7     16
#define CAMERA_PIN_D6     17
#define CAMERA_PIN_D5     18
#define CAMERA_PIN_D4     12
#define CAMERA_PIN_D3     10
#define CAMERA_PIN_D2      8
#define CAMERA_PIN_D1      9
#define CAMERA_PIN_D0     11
#define CAMERA_PIN_VSYNC   6
#define CAMERA_PIN_HREF    7
#define CAMERA_PIN_PCLK   13
*/

/*
// ── AI-Thinker ESP32-CAM (original ESP32, not S3) ──────────────────────────
#define CAMERA_PIN_PWDN   32
#define CAMERA_PIN_RESET  -1
#define CAMERA_PIN_XCLK    0
#define CAMERA_PIN_SIOD   26
#define CAMERA_PIN_SIOC   27
#define CAMERA_PIN_D7     35
#define CAMERA_PIN_D6     34
#define CAMERA_PIN_D5     39
#define CAMERA_PIN_D4     36
#define CAMERA_PIN_D3     21
#define CAMERA_PIN_D2     19
#define CAMERA_PIN_D1     18
#define CAMERA_PIN_D0      5
#define CAMERA_PIN_VSYNC  25
#define CAMERA_PIN_HREF   23
#define CAMERA_PIN_PCLK   22
*/

#endif // CAMERA_PINS_H
