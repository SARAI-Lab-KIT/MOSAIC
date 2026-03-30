#!/usr/bin/env python3
import time
import spidev
import board
import neopixel

# -----------------------------
# LED setup
# -----------------------------

STRIP_NUM = 30
RING_NUM = 16

strip = neopixel.NeoPixel(board.D19, STRIP_NUM, brightness=0.3, auto_write=False)
ring1 = neopixel.NeoPixel(board.D18, RING_NUM, brightness=0.3, auto_write=False)
ring2 = neopixel.NeoPixel(board.D13, RING_NUM, brightness=0.3, auto_write=False)

LEFT_RANGE = range(0, STRIP_NUM//2)
RIGHT_RANGE = range(STRIP_NUM//2, STRIP_NUM)

# -----------------------------
# MCP3008 setup
# -----------------------------

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz = 1350000

# -----------------------------
# Piezo channel mapping
# -----------------------------

LEFT_CHANNELS = [0,2]
RIGHT_CHANNELS = [1,4,3]

THRESHOLD = 10
COOLDOWN = 0.25

# -----------------------------
# ADC read
# -----------------------------

def read_adc(ch):
    adc = spi.xfer2([1,(8+ch)<<4,0])
    return ((adc[1]&3)<<8) + adc[2]

# -----------------------------
# LED helpers
# -----------------------------

def clear_leds():
    strip.fill((0,0,0))
    ring1.fill((0,0,0))
    ring2.fill((0,0,0))

    strip.show()
    ring1.show()
    ring2.show()


def flash_left():

    strip.fill((0,0,0))

    for i in LEFT_RANGE:
        strip[i] = (255,0,0)

    ring1.fill((255,0,0))

    strip.show()
    ring1.show()

    time.sleep(0.2)
    clear_leds()


def flash_right():

    strip.fill((0,0,0))

    for i in RIGHT_RANGE:
        strip[i] = (255,0,0)

    ring2.fill((255,0,0))

    strip.show()
    ring2.show()

    time.sleep(0.2)
    clear_leds()


def flash_all():

    strip.fill((255,0,0))
    ring1.fill((255,0,0))
    ring2.fill((255,0,0))

    strip.show()
    ring1.show()
    ring2.show()

    time.sleep(0.2)
    clear_leds()


# -----------------------------
# Initial baselines
# -----------------------------

channels = LEFT_CHANNELS + RIGHT_CHANNELS
prev_vals = {ch: read_adc(ch) for ch in channels}

last_event = 0

print("5 piezo sensors active (3 left / 2 right)")

# -----------------------------
# Main loop
# -----------------------------

while True:

    now = time.time()

    vals = {ch: read_adc(ch) for ch in channels}

    left_hit = False
    right_hit = False

    for ch in LEFT_CHANNELS:
        if abs(vals[ch] - prev_vals[ch]) > THRESHOLD:
            left_hit = True

    for ch in RIGHT_CHANNELS:
        if abs(vals[ch] - prev_vals[ch]) > THRESHOLD:
            right_hit = True

    if (now - last_event) > COOLDOWN:

        if left_hit and right_hit:
            print("BOTH SIDES TOUCH")
            flash_all()
            last_event = now

        elif left_hit:
            print("LEFT TOUCH")
            flash_left()
            last_event = now

        elif right_hit:
            print("RIGHT TOUCH")
            flash_right()
            last_event = now

    prev_vals = vals

    time.sleep(0.002)