"""Coordinator-side ephemeral peer listeners and guarded cygnus clients."""
import json
from pathlib import Path
import select
import shlex
import subprocess

ROOT = '/home/dberweger/cygnus-qualification-20261014'
OUT = Path('results/cygnus-qualification/network')
SOURCE = Path(__file__).with_name('network_probe.py').read_text()


def ssh(host, command, **kwargs):
    return subprocess.run(['ssh', '-o', 'ConnectTimeout=15', host, command],
                          check=True, text=True, **kwargs)


def launch(label, seconds, command):
    guarded = ['python3', 'watchdog-overnight.py', '--output', label,
               '--seconds', str(seconds), '--deadline-utc',
               '2026-10-15T04:50:00Z', '--', *command]
    shell = (f'cd {ROOT} && {shlex.join(guarded)} > {label}-supervisor.log 2>&1; '
             f'code=$?; tmux -L cygnus-qualification wait-for -S {label}-complete; exit "$code"')
    ssh('bb128', f'test ! -e {ROOT}/{label} && tmux -L cygnus-qualification new-session -d -s {label} {shlex.quote(shell)}')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    peers = [('db-ocean', '100.96.144.26'), ('m4', '100.122.216.94')]
    ssh('bb128', f'cd {ROOT} && tailscale status --json',
        stdout=(OUT / 'tailscale-status.json').open('w'))
    results = []
    for host, address in peers:
        for index, direction in enumerate(['forward', 'reverse']):
            label = f'network-{host}-{direction}'
            port = 52381 + index
            server = subprocess.Popen(
                ['ssh', '-o', 'ConnectTimeout=15', host,
                 shlex.join(['python3', '-u', '-', 'server', address, str(port), direction])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True)
            try:
                server.stdin.write(SOURCE)
                server.stdin.close()
                server.stdin = None
                if not select.select([server.stdout], [], [], 20)[0]:
                    raise RuntimeError(f'{host} listener readiness timed out')
                ready = server.stdout.readline().strip()
                if ready != 'READY':
                    raise RuntimeError(f'{host} listener failed: {ready}')
                launch(label, 45, ['python3', 'network_probe.py', 'client',
                                   address, str(port), direction])
                output, error = server.communicate(timeout=50)
                (OUT / f'{label}-peer.log').write_text(output + error)
                if server.returncode:
                    raise RuntimeError(f'{label} peer exited {server.returncode}')
                peer = json.loads(output.strip())
                raw = ssh('bb128', f'timeout 50 tmux -L cygnus-qualification wait-for {label}-complete && cat {ROOT}/{label}/result.json && cat {ROOT}/{label}/output.log',
                          capture_output=True).stdout
                (OUT / f'{label}-client.log').write_text(raw)
                # The pretty-printed guard receipt precedes the one-line client result.
                decoder = json.JSONDecoder()
                guard, end = decoder.raw_decode(raw)
                client = json.loads(raw[end:].strip())
                if guard['status'] != 'passed' or client['bytes'] != peer['bytes']:
                    raise RuntimeError(f'{label} guard or byte-count mismatch')
                results.append({'peer': host, 'direction': direction,
                                'client': client, 'server': peer, 'guard': guard})
                (OUT / 'throughput.json').write_text(json.dumps(results, indent=2) + '\n')
                print(f'{label}: {client["mbit_s"]:.2f} Mbit/s', flush=True)
            finally:
                if server.poll() is None:
                    server.terminate()
                    server.wait(timeout=10)
    # ICMP is a separate phase so TCP transfer does not distort latency.
    command = 'set -e; ' + '; '.join(
        f'tailscale ping --c 3 --timeout 2s {address}; ping -n -c 30 -i 0.2 {address}'
        for _, address in peers)
    launch('network-latency', 60, ['bash', '-c', command])
    print('TCP complete; guarded latency phase launched.', flush=True)


if __name__ == '__main__':
    main()
