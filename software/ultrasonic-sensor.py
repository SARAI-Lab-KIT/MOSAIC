#!/usr/bin/env python3
import time
from gpiozero import DistanceSensor
import board
import neopixel

# -----------------------------
# LED configuration
# -----------------------------

STRIP_PIN = board.D19
STRIP_NUM = 30
RING_NUM = 16

strip = neopixel.NeoPixel(STRIP_PIN, STRIP_NUM, brightness=0.1, auto_write=False)
ring1 = neopixel.NeoPixel(board.D18, RING_NUM, brightness=0.1, auto_write=False)
ring2 = neopixel.NeoPixel(board.D13, RING_NUM, brightness=0.1, auto_write=False)

# -----------------------------
# Ultrasonic configuration
# -----------------------------

sensor = DistanceSensor(echo=6, trigger=5, max_distance=2.0)

APPROACH_START_CM = 150
APPROACH_NEAR_CM = 40

MOVEMENT_THRESHOLD = 3.0
STILL_TIMEOUT = 2.5

prev_dist = None
last_motion = time.time()


def set_led(brightness):
    strip.brightness = brightness
    ring1.brightness = brightness
    ring2.brightness = brightness

    strip.fill((0,255,0))
    ring1.fill((0,255,0))
    ring2.fill((0,255,0))

    strip.show()
    ring1.show()
    ring2.show()


def leds_off():
    strip.fill((0,0,0))
    ring1.fill((0,0,0))
    ring2.fill((0,0,0))

    strip.show()
    ring1.show()
    ring2.show()


print("Ultrasonic interaction test running")

while True:

    dist = sensor.distance * 100

    if prev_dist is None:
        prev_dist = dist

    # detect motion
    if abs(dist - prev_dist) > MOVEMENT_THRESHOLD:
        last_motion = time.time()

    prev_dist = dist

    # check if user present
    if dist < APPROACH_START_CM:

        # brightness ramp
        d = max(APPROACH_NEAR_CM, min(dist, APPROACH_START_CM))
        t = (APPROACH_START_CM - d) / (APPROACH_START_CM - APPROACH_NEAR_CM)

        brightness = 0.05 + t * 0.45

        # if user stopped moving close
        if (time.time() - last_motion) > STILL_TIMEOUT and dist < APPROACH_NEAR_CM:
            leds_off()
        else:
            set_led(brightness)

    else:
        leds_off()

    time.sleep(0.05)