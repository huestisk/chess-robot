import io
import socket
import struct
from PIL import Image
from collections import deque


class ImageBuffer():
    image_buffer = deque([])


class Server():
    def __init__(self):
        self.conn = None
        self.sock = None

    def run_server(self, host, port: int):
        """
        Run the server to allow Raspberry Pi to connect

        """
        self.sock = socket.socket()
        self.sock.bind((host, port))
        self.sock.listen(0)
        print("Listening")
        self.conn = self.sock.accept()[0].makefile('rb')

    def read_images(self, buffer):
        """
        Write incoming images to Buffe
        TODO: kill Signal

        """
        try:
            while True:
                image_len = struct.unpack(
                    '<L', self.conn.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break

                image_stream = io.BytesIO()
                image_stream.write(self.conn.read(image_len))
                image_stream.seek(0)
                image = Image.open(image_stream)

                buffer.image_buffer.append(image)

        finally:
            self.conn.close()
            self.sock.close()
