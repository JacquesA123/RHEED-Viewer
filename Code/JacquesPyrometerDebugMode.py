#!/usr/bin/env python3
import minimalmodbus
import serial

instrument = minimalmodbus.Instrument('COM4', 1, debug = True),  # port name, slave address (in decimal)
print(instrument)

# instrument.serial.port = 'COM4'                     # this is the serial port name
# instrument.serial.baudrate = 115200         # Baud
# # instrument.serial.baudrate = 950         # Baud
# instrument.serial.bytesize = 8
# instrument.serial.parity   = serial.PARITY_NONE
# instrument.serial.stopbits = 1
# instrument.serial.timeout  = 0.05          # seconds

# instrument.mode = minimalmodbus.MODE_RTU   # rtu or ascii mode
# instrument.clear_buffers_before_each_transaction = True
print(instrument.read_register(289, 1))
## Read temperature (PV = ProcessValue) ##
# temperature = instrument.read_register(1)  # Registernumber, number of decimals
# print(temperature)
