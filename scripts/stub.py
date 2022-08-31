#!/usr/bin/env python3

import zmq
import sys

PORT = 5555


def get_obj_val(parameters):

    context = zmq.Context()

    socket = context.socket(zmq.REQ)

    socket.connect(f'tcp://localhost:{PORT}')

    socket.send(parameters.encode("utf-8"))

    message = bytes(socket.recv())

    return message.decode("utf-8")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("ERROR: At least one parameter is required")
        exit(1)
    else:
        input_str = ' '.join(sys.argv[1:])
        result = get_obj_val(input_str)
        print(result)

