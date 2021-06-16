#!/usr/bin/env python3
import sys
import time
import pyftdi.serialext

#port = pyftdi.serialext.serial_for_url('ftdi://ftdi:2232h/2', baudrate=115200)
port = pyftdi.serialext.serial_for_url('ftdi://ftdi:2232h/2', baudrate=1000000)
print(port)

while 1:
  #port.write(b'a')
  data = port.read(1)
  sys.stdout.write(data.decode('utf-8'))
  #time.sleep(0.01)



