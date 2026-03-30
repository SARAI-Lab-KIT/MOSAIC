#!/usr/bin/env python3
import time
import math
import board
import neopixel
import qwiic_icm20948

# -----------------------------
# LED configuration
# -----------------------------

STRIP_NUM = 30
RING_NUM = 16

strip = neopixel.NeoPixel(board.D19, STRIP_NUM, brightness=0.3, auto_write=False)
ring1 = neopixel.NeoPixel(board.D18, RING_NUM, brightness=0.3, auto_write=False)
ring2 = neopixel.NeoPixel(board.D13, RING_NUM, brightness=0.3, auto_write=False)

# -----------------------------
# IMU Setup
# -----------------------------

imu = qwiic_icm20948.QwiicIcm20948()

if not imu.connected:
    print("IMU not detected")
    exit()

imu.begin()
print("IMU initialized")


ACCEL_THRESHOLD = 10000
GYRO_THRESHOLD = 1200


def vec_delta(a,b):
    return math.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)


def flash_blue():

    for _ in range(3):

        strip.fill((0,0,255))
        ring1.fill((0,0,255))
        ring2.fill((0,0,255))

        strip.show()
        ring1.show()
        ring2.show()

        time.sleep(0.15)

        strip.fill((0,0,0))
        ring1.fill((0,0,0))
        ring2.fill((0,0,0))

        strip.show()
        ring1.show()
        ring2.show()

        time.sleep(0.15)


# initial sample
while not imu.dataReady():
    time.sleep(0.01)

imu.getAgmt()

prev_accel = (imu.axRaw, imu.ayRaw, imu.azRaw)
prev_gyro = (imu.gxRaw, imu.gyRaw, imu.gzRaw)


print("Move the robot to test pickup detection")

while True:

    if imu.dataReady():

        imu.getAgmt()

        accel = (imu.axRaw, imu.ayRaw, imu.azRaw)
        gyro = (imu.gxRaw, imu.gyRaw, imu.gzRaw)

        d_accel = vec_delta(prev_accel, accel)
        d_gyro = vec_delta(prev_gyro, gyro)

        if d_accel > ACCEL_THRESHOLD or d_gyro > GYRO_THRESHOLD:

            print("Robot moved")
            flash_blue()

        prev_accel = accel
        prev_gyro = gyro

    time.sleep(0.05)