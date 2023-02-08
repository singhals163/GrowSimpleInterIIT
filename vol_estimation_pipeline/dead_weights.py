import serial

def get_dead_weights(portname):
    try:
        arduino = serial.Serial(portname, timeout=1, baudrate=57600)
    except:
        print('Please check the port')

    dead_weight = 0    
    count = 0
    while count<3:
        weight = arduino.realine()
        weight = weight.decode()
        weight = weight.strip()
        weight = float(weight)
        dead_weight += weight
        count+=1
    dead_weight /= 3
    print(dead_weight)
    arduino.close()
    return dead_weight