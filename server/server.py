import socket
import threading
import json
import time
import datetime
import pytz
import pymodm
from db import connect_to_db
from models.data_model import PMData
# Request handler
def handler(sock, addr):
    msg = 'YEE from server. YEEEEEEEEEE'
    print(sock.gettimeout())
    sock.send(msg.encode('utf-8'))
    data = {}
    while True:
        try:
            # socket is in blocking-mode by default
            # so recv() will fail if 90s has passed
            msg = sock.recv(1024)
            if not msg:
                continue
            else:
                try:
                    # json format must be double quoted instead of being single quoted
                    data = json.loads(msg.decode('utf-8').replace("\'", "\""))
                    print(f"{datetime.datetime.now()}: {data}")
                    # unpacking the tuple
                    PMData(pm10=data.get('pm10'), pm25=data.get('pm25'), pm100=data.get('pm100'), temp=data.get('temp'),
                        humidity=data.get('humidity'), position=data.get('position'), date=datetime.datetime.now()).save()
                except pymodm.errors.ValidationError as err:
                    print(err)
                except ValueError as err:
                    print(err)
        except socket.timeout as err:
            print(f"Which bitch data {data}")
            print(f"{data.get('position', addr)} disconnects")
            sock.close()
            break
        except NameError as err:
            print(err)

def main():
    # Turn on server
    main_sock = socket.socket()
    main_sock.bind(('0.0.0.0', 8080))# port
    main_sock.listen(5)
    # Connect to mongodb
    connect_to_db()
    # testing for insertion

    print('Waiting for connection...')
    while True:
        (socket_accept, addr) = main_sock.accept()
        # Create a new thread to handle requests
        socket_accept.settimeout(90)
        thread = threading.Thread(target=handler, args=(socket_accept, addr))
        thread.start()

if __name__ == '__main__':
    main()
