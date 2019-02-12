import socket
import threading
from pymongo import MongoClient

# Request handler
def handler(sock,addr):
    sock.send('YEE from server. YEEEEEEEEEE')
    while True:
        data=sock.recv(1024)
        if not data:
            break;
        sock.send('Echo : %s' % data)
    sock.close()

# Testing only, don't use this zzzzzzzzz
def add_to_db():
    # Create new post
    #for testing
    new_dict = {"name": "John",
             "id": "100",
             "work": ["mongodb", "python"]}
    # insert into collection
    collection.insert_one(new_dict)

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
        thread = threading.Thread(target=handler,args=(socket,addr))
        thread.start()

if __name__ == '__main__':
    mainfunc()
