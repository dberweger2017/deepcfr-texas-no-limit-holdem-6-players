"""Bounded single-stream TCP measurement, without peer files or services.

Run the server from SSH stdin; its socket binds only to the peer's Tailscale IP.
The application byte rate excludes SSH and includes socket backpressure.
"""
import argparse
import json
import socket
import time


def transfer(connection, send):
    connection.settimeout(20)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    block = b'\xa5' * (1024 * 1024)
    total = 0
    start = time.monotonic()
    if send:
        while time.monotonic() - start < 15:
            connection.sendall(block)
            total += len(block)
        connection.shutdown(socket.SHUT_WR)
    else:
        while True:
            data = connection.recv(len(block))
            if not data:
                break
            total += len(data)
    elapsed = time.monotonic() - start
    return {'bytes': total, 'seconds': elapsed,
            'mbit_s': total * 8 / elapsed / 1e6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=['server', 'client'])
    p.add_argument('host')
    p.add_argument('port', type=int)
    p.add_argument('direction', choices=['forward', 'reverse'])
    args = p.parse_args()
    if args.role == 'server':
        with socket.socket() as listener:
            listener.settimeout(45)
            listener.bind((args.host, args.port))
            listener.listen(1)
            print('READY', flush=True)
            connection, address = listener.accept()
            if address[0] != '100.99.57.8':
                raise SystemExit(f'Unexpected client {address[0]}')
            with connection:
                result = transfer(connection, args.direction == 'reverse')
    else:
        with socket.create_connection((args.host, args.port), timeout=10) as connection:
            result = transfer(connection, args.direction == 'forward')
    print(json.dumps({'role': args.role, 'peer': args.host,
                      'direction': args.direction, **result}), flush=True)


if __name__ == '__main__':
    main()
