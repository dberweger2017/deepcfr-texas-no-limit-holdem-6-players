"""Bounded JSON-object parsing and disposable on-disk exact key matching."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory


class JsonStream:
    """Keep only one JSON value plus a read chunk; never decode the entries object."""

    def __init__(self, source):
        self.source = source
        self.buffer = ''
        self.eof = False
        self.decoder = json.JSONDecoder()

    def fill(self):
        chunk = self.source.read(65536)
        self.buffer += chunk
        self.eof = not chunk

    def whitespace(self):
        while True:
            self.buffer = self.buffer.lstrip(' \t\r\n')
            if self.buffer or self.eof:
                return
            self.fill()

    def token(self, expected):
        self.whitespace()
        if not self.buffer.startswith(expected):
            raise ValueError('Malformed inference JSON delimiter')
        self.buffer = self.buffer[len(expected):]

    def value(self):
        self.whitespace()
        try:
            value, end = self.decoder.raw_decode(self.buffer)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return self.scalar()
            self.buffer = self.buffer[end:]
            return value
        except json.JSONDecodeError:
            # An incomplete chunk and malformed syntax are indistinguishable to
            # raw_decode. Parse structure incrementally instead of retaining the
            # entire remaining file while waiting for a successful decode.
            return self.incremental_value()

    def scalar(self):
        pieces = []
        while True:
            end = 0
            while end < len(self.buffer) and self.buffer[end] not in ' \t\r\n,]}':
                end += 1
            pieces.append(self.buffer[:end])
            self.buffer = self.buffer[end:]
            if self.buffer or self.eof:
                break
            self.fill()
        try:
            return self.decoder.decode(''.join(pieces))
        except json.JSONDecodeError as error:
            raise ValueError('Malformed inference JSON scalar') from error

    def incremental_value(self):
        self.whitespace()
        if self.buffer.startswith('['):
            self.token('['); self.whitespace(); values = []
            if self.buffer.startswith(']'):
                self.token(']'); return values
            while True:
                values.append(self.value()); self.whitespace()
                if self.buffer.startswith(']'):
                    self.token(']'); return values
                self.token(',')
        if self.buffer.startswith('{'):
            values = {}
            for key in self.object_items():
                values[key] = self.value()
            return values
        if self.buffer.startswith('"'):
            pieces = ['"']; self.token('"'); escaped = False
            while True:
                for i, character in enumerate(self.buffer):
                    if character == '"' and not escaped:
                        pieces.append(self.buffer[:i+1]); self.buffer = self.buffer[i+1:]
                        try:
                            return self.decoder.decode(''.join(pieces))
                        except json.JSONDecodeError as error:
                            raise ValueError('Malformed inference JSON string') from error
                    escaped = character == '\\' and not escaped
                pieces.append(self.buffer); self.buffer = ''
                if self.eof:
                    raise ValueError('Truncated inference JSON string')
                self.fill()
        if self.buffer[:1] in ('-', *'0123456789', 't', 'f', 'n'):
            return self.scalar()
        raise ValueError('Malformed or truncated inference JSON value')

    def object_items(self):
        self.token('{')
        self.whitespace()
        if self.buffer.startswith('}'):
            self.token('}')
            return
        while True:
            key = self.value()
            if not isinstance(key, str):
                raise ValueError('Invalid inference JSON key')
            self.token(':')
            yield key
            self.whitespace()
            if self.buffer.startswith('}'):
                self.token('}')
                return
            self.token(',')


def current_rows(source, metadata):
    """Yield entry pairs in file order and finish validating all metadata/trailing bytes."""
    stream = JsonStream(source)
    fields = set()
    for field in stream.object_items():
        if field in fields:
            raise ValueError('Duplicate inference JSON field')
        fields.add(field)
        if field == 'entries':
            for key in stream.object_items():
                yield key, stream.value()
        else:
            metadata[field] = stream.value()
    stream.whitespace()
    if stream.buffer or 'entries' not in fields:
        raise ValueError('Trailing inference JSON or missing entries')


@contextmanager
def disk_index():
    """An exact, disk-backed ledger; its cache and Python working set do not grow."""
    with TemporaryDirectory(prefix='cfr-audit-') as directory:
        connection = sqlite3.connect(Path(directory) / 'keys.sqlite')
        try:
            connection.execute('PRAGMA cache_size=-4096')
            connection.execute('PRAGMA mmap_size=0')
            # This index is disposable, not a checkpoint or published artifact.
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('CREATE TABLE entries (key TEXT PRIMARY KEY, row TEXT) WITHOUT ROWID')
            yield connection
        finally:
            connection.close()


def insert_unique(index, key, row=None):
    try:
        index.execute('INSERT INTO entries VALUES (?, ?)', (key, row))
    except sqlite3.IntegrityError as error:
        raise ValueError('Duplicate retained node') from error
