/*
 * distraction_serial_bridge.ino
 * For Arduino Nano / Nano 33 BLE / Nano Every — boards that can't run
 * the model on-device. Inference runs on your computer (Python), and
 * results are sent to the Nano over serial to drive LEDs/buzzer.
 *
 * Protocol (serial @ 115200):
 *   Computer sends one character per frame:
 *     'D' = distracted
 *     'F' = focused
 *     'E' = empty
 *
 * Wiring:
 *   D2 → Red LED (+ 220Ω resistor) → GND    (distracted)
 *   D3 → Green LED (+ 220Ω resistor) → GND  (focused)
 *   D4 → Piezo buzzer → GND                  (alert)
 *
 * Pair with: python run_live.py --serial /dev/ttyUSB0
 *   (or add --serial flag to run_live.py)
 */

#define LED_RED    2
#define LED_GREEN  3
#define BUZZER_PIN 4

#define ALERT_FRAMES 10  // Buzz after N consecutive distracted frames

int distracted_count = 0;

void setup() {
    Serial.begin(115200);

    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    // Startup blink
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_RED, HIGH);
        digitalWrite(LED_GREEN, HIGH);
        delay(100);
        digitalWrite(LED_RED, LOW);
        digitalWrite(LED_GREEN, LOW);
        delay(100);
    }

    Serial.println("READY");
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = Serial.read();

        switch (cmd) {
            case 'D':  // Distracted
                digitalWrite(LED_RED, HIGH);
                digitalWrite(LED_GREEN, LOW);
                distracted_count++;
                if (distracted_count >= ALERT_FRAMES) {
                    tone(BUZZER_PIN, 2000, 200);  // 2kHz beep
                }
                break;

            case 'F':  // Focused
                digitalWrite(LED_RED, LOW);
                digitalWrite(LED_GREEN, HIGH);
                distracted_count = 0;
                noTone(BUZZER_PIN);
                break;

            case 'E':  // Empty
                digitalWrite(LED_RED, LOW);
                digitalWrite(LED_GREEN, LOW);
                distracted_count = 0;
                noTone(BUZZER_PIN);
                break;
        }
    }
}
