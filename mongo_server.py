import socket
import threading
import os
from pymongo import MongoClient

# Request handler
def handler(sock, addr, collection):
    msg = 'YEE from server. YEEEEEEEEEE'
    sock.send(msg.encode('utf-8'))
    #while True: to loop
    data=sock.recv(1024)
    if not data:
        print("ERROR : No data")
    #create new dict to insert
    key_list = ["id", "time", "pm1.0", "pm2.5", "pm10", "temp", "humid"]
    value_list = data.split()
    new_dict = dict(zip(key_list, value_list))
    collection.insert_one(new_dict)
    
    msg = 'Data received. Closing connection...'
    sock.send(msg.encode('utf-8'))
    sock.close()

# Main function
if __name__ == '__main__':
    # Open mongodb
    #url = "mongodb://USERNAME:password@host?authSource=source" 
    #client = MongoClient(url)
    client = MongoClient(os.environ['DB_PORT_27017_TCP_ADDR'], 27017)
    db = client.<database_name> # change db name here !!!!!!!!!!
    collection = db.<collection_name> # change collection name here !!!!!!!!!!
    
    # Turn on server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1',5000))# port
    sock.listen(5)
    print('Waiting for connection...')
    
    # Keep on accepting connection, create a thread for each
    while True:
        socket,addr = sock.accept()
        # Create a new thread to handle requests
        thread = threading.Thread(target=handler,args=(socket, addr, collection))
        thread.start()
