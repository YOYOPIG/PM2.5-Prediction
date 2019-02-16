import socket
import threading
from pymodm.connection import connect

# Request handler
def handler(sock,addr):
    msg = 'YEE from server. YEEEEEEEEEE'

    sock.send(msg.encode('utf-8'))

    while True:
        data=sock.recv(1024)
        if not data:
            print("ERROR : No data")
            msg = 'Send something u idiot'
            sock.send(msg.encode('utf-8'))
        else:
            msg = 'Data received'
            sock.send(msg.encode('utf-8'))
            print(data)
    msg = 'Closing connection...'
    sock.send(msg.encode('utf-8'))
    sock.close()

if __name__ == '__main__':
    # Turn on server
    sock = socket.socket()
    sock.bind(('0.0.0.0', 8080))# port
    sock.listen(5)
    # Connect to mongodb
    connect("mongodb://mongo:27017/test")
    print('Waiting for connection...')
    while True:
        (socket,addr) = sock.accept()
        # Create a new thread to handle requests
        thread = threading.Thread(target=handler,args=(socket,addr))
        thread.start()

