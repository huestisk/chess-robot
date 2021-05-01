import io
import sys
import subprocess

from threading import Thread
from connection.server import Server, ImageBuffer
from detector.detector import Detector

HOST = sys.argv[1]
PORT = sys.argv[2]

""" Run Server """
server = Server()
t = Thread(target=server.run_server, args=[HOST, int(PORT)])
t.start()

""" Connect to Raspberry Pi & run client """
# Note: Pub key must be added to raspberry pi
user = 'pi'
host = '192.168.1.123'
options = ''
cmd = 'python3 client.py ' + HOST + ' ' + PORT
ssh_cmd = 'ssh %s@%s %s "%s"' % (user, host, options, cmd)
p = subprocess.Popen([ssh_cmd], shell=True)

# Wait for server thread to finish
t.join()

""" Read Images """
buffer = ImageBuffer()
t = Thread(target=server.read_images, args=[buffer])
t.start()

""" Detect the chess board """
detector = Detector()
detector.show_stream(buffer, detect=True)

print(0)

