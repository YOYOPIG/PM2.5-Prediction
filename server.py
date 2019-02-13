import socket
import threading
from pymongo import MongoClient

# Request handler
def handler(sock,addr, collection):
    sock.send('YEE from server. YEEEEEEEEEE')
    #while True: to loop
    data=sock.recv(1024)
    if not data:
        print("ERROR : No data")
    #create new dict to insert
    key_list = ["id", "time", "pm1.0", "pm2.5", "pm10", "temp", "humid"]
    value_list = data.split()
    new_dict = dict(zip(key_list, value_list))
    collection.insert_one(new_dict)
    sock.send('Data received. Closing connection...')
    sock.close()

# Main function
def mainfunc():
    # Open mongodb
    url = "mongodb://USERNAME:password@host?authSource=source" 
    client = MongoClient(url)
    db = client.<database_name> # change db name here !!!!!!!!!!
    collection = db.<collection_name> # change collection name here !!!!!!!!!!
    
    # Turn on server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1',5000))# port
    sock.listen(5)
    print('Waiting for connection...')
    while True:
        socket,addr = sock.accept()
        # Create a new thread to handle requests
        thread = threading.Thread(target=handler,args=(socket,addr, collection))
        thread.start()

if __name__ == '__main__':
    mainfunc()
