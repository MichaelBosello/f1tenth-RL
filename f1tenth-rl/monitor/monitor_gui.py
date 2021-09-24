import socket
import pickle

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

while True:
    data_msg, addr = sock.recvfrom(32768)
    data = pickle.loads(data_msg)
    print("received message: {}".format(data))