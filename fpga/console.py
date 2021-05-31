#!/usr/bin/env python3
import time
import pyftdi.serialext

port = pyftdi.serialext.serial_for_url('ftdi://ftdi:2232h/2', baudrate=115200)
print(port)

while 1:
  port.write(b'a')
  data = port.read(1)
  print(data)
  time.sleep(0.01)



