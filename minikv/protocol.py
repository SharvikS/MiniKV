r"""
MiniKV - Module 3: The RESP Wire Protocol

Created: 2026-09-04 (Module 3)

Modules 1 and 2 moved bytes around without ever asking what they meant. That is
fine for an echo server and for nothing else, because TCP delivers a *stream*:
recv() hands back whatever happened to arrive, which may be half a command, one
command, or six commands glued together. Nothing in TCP marks where one command
ends and the next begins - that framing is our job, and RESP (the REdis
Serialization Protocol) is how we do it.

RESP in one screen. Every value starts with one byte naming its type and ends
with CRLF:

    +OK\r\n                            simple string - short, no newlines
    -ERR unknown command\r\n           error         - a simple string, bad news
    :42\r\n                            integer       - signed 64-bit, decimal
    $5\r\nhello\r\n                    bulk string   - length-prefixed, binary safe
    $-1\r\n                            null bulk     - "no value", not ""
    *2\r\n$4\r\nECHO\r\n$2\r\nhi\r\n   array         - a count, then that many values

Clients send a command as an array of bulk strings; the server replies with
whichever type fits the answer. The length prefix is what makes bulk strings
binary safe: we never scan the payload looking for a terminator, so a value may
contain CRLF, NUL bytes, a JPEG - anything.

The file is split into two halves that have very different characters:

    encoding - pure functions, Python value in, bytes out. Trivial.
    decoding - RESPParser, which must survive *arbitrary* chunk boundaries.

Everything here is bytes, never str. A Redis key or value is a byte string, and
decoding it as text would corrupt anything that is not valid UTF-8.
"""

from collections.abc import Iterator, Sequence

# --- Wire constants ----------------------------------------------------------

CRLF = b"\r\n"

# Type bytes, named so the parser reads like the spec above.
SIMPLE_STRING_BYTE = b"+"
ERROR_BYTE = b"-"
INTEGER_BYTE = b":"
BULK_STRING_BYTE = b"$"
ARRAY_BYTE = b"*"


# --- Safety limits -----------------------------------------------------------
#
# A parser that trusts its input is a denial-of-service waiting to happen: the
# five bytes "*9999999999\r\n" would otherwise talk us into preallocating a list
# for ten billion elements. Every length that arrives off the wire gets checked
# against one of these before we act on it. The values match real Redis.

MAX_BULK_LENGTH = 512 * 1024 * 1024     # 512 MB, Redis' proto-max-bulk-len
MAX_ARRAY_LENGTH = 1024 * 1024          # elements in a single command
MAX_LINE_LENGTH = 64 * 1024             # a header or inline command line

# Sentinel meaning "the bytes for this value have not all arrived yet". It is a
# unique object rather than None because None is a perfectly good parse result
# (a null bulk string), and confusing the two is exactly the kind of bug that
# only shows up under a chunk boundary in production.
INCOMPLETE = object()


class ProtocolError(Exception):
    """The peer sent bytes that are not RESP, and never will be.

    Added: 2026-09-04 (Module 3)

    This is unrecoverable by design. A stream framed by length prefixes has no
    resynchronisation point: once we cannot trust the framing, every byte after
    it is suspect. Real Redis reports the error and closes the connection, and
    so does MiniKV - see handle_client() in server.py.
    """


# =============================================================================
# Encoding: Python values -> RESP bytes
# =============================================================================

def simple_string(text: str) -> bytes:
    r"""Encode a status reply: `+OK\r\n`.

    Added: 2026-09-04 (Module 3)

    Simple strings are unframed - the reply ends at the first CRLF - so a CR or
    LF inside `text` would end the value early and desynchronise the client's
    parser. That makes this the one encoder that can produce a corrupt stream
    from valid Python, so it refuses instead. Use bulk_string() for any text
    that comes from a client or from user data.
    """
    if "\r" in text or "\n" in text:
        raise ValueError("simple strings cannot contain CR or LF; use bulk_string()")
    return SIMPLE_STRING_BYTE + text.encode("utf-8") + CRLF


def error(message: str) -> bytes:
    r"""Encode an error reply: `-ERR unknown command 'FOO'\r\n`.

    Added: 2026-09-04 (Module 3)

    Framed exactly like a simple string; the leading '-' is the only difference,
    and it is what tells a client library to raise instead of return. By
    convention the first word is a machine-readable error code - ERR for the
    generic case, then WRONGTYPE, NOAUTH and friends as we grow.
    """
    if "\r" in message or "\n" in message:
        raise ValueError("error messages cannot contain CR or LF")
    return ERROR_BYTE + message.encode("utf-8") + CRLF


def integer(value: int) -> bytes:
    r"""Encode an integer reply: `:42\r\n`.

    Added: 2026-09-04 (Module 3)

    Sent in decimal ASCII, not binary, which costs a few bytes and buys us not
    having to care about the endianness of either machine. Redis guarantees the
    value fits in a signed 64-bit integer, which Python's unbounded ints do not,
    so callers of commands like INCR have to do that range check themselves.
    """
    return INTEGER_BYTE + str(value).encode("ascii") + CRLF


def bulk_string(value: bytes | str | None) -> bytes:
    r"""Encode a bulk string: `$5\r\nhello\r\n`, or `$-1\r\n` for None.

    Added: 2026-09-04 (Module 3)

    The workhorse: every key, every value, every argument travels this way. The
    length prefix means the payload is never scanned, so it is binary safe, and
    it means a length of -1 can mean something a byte string cannot - "there is
    no value here". That null is how GET reports a missing key in Module 4, and
    it is genuinely different from the empty string, which a key really can hold.
    """
    if value is None:
        return NULL_BULK_STRING
    if isinstance(value, str):
        value = value.encode("utf-8")
    return BULK_STRING_BYTE + str(len(value)).encode("ascii") + CRLF + value + CRLF


def array(items: Sequence[bytes] | None) -> bytes:
    r"""Encode an array: a count, then the already-encoded `items` verbatim.

    Added: 2026-09-04 (Module 3)

    Note the signature: this takes *encoded* RESP values, not Python ones. RESP
    arrays are heterogeneous - a single reply can mix bulk strings, integers and
    nested arrays - so the caller is the one who knows what each element is:

        array([bulk_string(b"key"), integer(1)])

    None encodes as the null array `*-1\r\n`, which is not the same as the empty
    array `*0\r\n`: "no result at all" versus "a result, which is empty".
    """
    if items is None:
        return NULL_ARRAY
    return ARRAY_BYTE + str(len(items)).encode("ascii") + CRLF + b"".join(items)


# Replies we send constantly. Building them once keeps the hot path free of
# string formatting, and gives the common cases a name worth reading.
OK = simple_string("OK")
PONG = simple_string("PONG")
NULL_BULK_STRING = b"$-1\r\n"
NULL_ARRAY = b"*-1\r\n"
EMPTY_ARRAY = b"*0\r\n"


# =============================================================================
# Decoding: RESP bytes -> commands
# =============================================================================

class RESPParser:
    """Turns an arbitrarily chopped-up byte stream into whole commands.

    Added: 2026-09-04 (Module 3)

    One parser belongs to one connection, because the leftover half-command it
    is holding belongs to that connection. Usage is feed-then-drain:

        parser = RESPParser()
        while True:
            data = conn.recv(4096)
            if not data:
                break
            parser.feed(data)
            for command in parser.commands():
                reply(dispatch(command))

    The contract that makes this safe is that bytes are consumed only when a
    command is *complete*. A partial command stays in the buffer untouched until
    the rest of it arrives, however many recv() calls that takes - so the two
    cases that break naive parsers, one command split across reads and several
    commands arriving in one read, are the same case here.

    A command comes out as a list of byte strings: [b"SET", b"name", b"minikv"].
    """

    def __init__(self) -> None:
        # bytearray, not bytes: we append on every read and delete from the
        # front on every complete command, and bytes would copy the whole
        # buffer each time.
        self._buffer = bytearray()

    def feed(self, data: bytes) -> None:
        """Add freshly received bytes to the buffer. Never parses.

        Added: 2026-09-04 (Module 3)
        """
        self._buffer += data

    def commands(self) -> Iterator[list[bytes]]:
        """Yield every complete command now in the buffer, then stop.

        Added: 2026-09-04 (Module 3)

        Stopping cleanly at a partial command is the whole point: the generator
        simply ends, and the leftover bytes wait for the next feed(). This is
        also what makes pipelining work for free - a client that writes ten
        commands without waiting for replies just means ten iterations here.
        """
        while True:
            command = self._parse_command()
            if command is INCOMPLETE:
                return
            # A client can legitimately send `*0\r\n` or a blank inline line.
            # Redis ignores those rather than erroring, and so do we; yielding
            # an empty list would push the special case onto every caller.
            if command:
                yield command

    # --- internals -----------------------------------------------------------

    def _parse_command(self) -> object:
        """Parse one command, or return INCOMPLETE without consuming anything.

        Added: 2026-09-04 (Module 3)
        """
        if not self._buffer:
            return INCOMPLETE

        if self._buffer[:1] == ARRAY_BYTE:
            result = self._parse_array(0)
        else:
            # Anything not starting with '*' is an inline command. Real clients
            # never send these; humans with telnet and netcat do, which is
            # exactly why Redis supports them and why we do too. It means you
            # can `nc 127.0.0.1 6379`, type PING, and get PONG.
            result = self._parse_inline(0)

        if result is INCOMPLETE:
            return INCOMPLETE

        command, consumed = result
        # The commit point. Everything above this line worked on an index into
        # the buffer and mutated nothing, so bailing out early is free; only a
        # complete command gets to move the buffer forward.
        del self._buffer[:consumed]
        return command

    def _read_line(self, pos: int, terminator: bytes = CRLF) -> object:
        """Read up to the next `terminator`. Returns (line, next_pos).

        Added: 2026-09-04 (Module 3)

        RESP headers are terminated by CRLF and nothing else, so the default is
        strict. Inline commands pass terminator=b"\n" instead, because they are
        typed by humans and half the terminals in the world send a bare LF.
        """
        end = self._buffer.find(terminator, pos)
        if end == -1:
            # No terminator yet. Bound how long we are willing to wait for one,
            # or a client that opens a connection and streams megabytes with no
            # line ending in them makes us buffer all of it.
            if len(self._buffer) - pos > MAX_LINE_LENGTH:
                raise ProtocolError("too big inline request")
            return INCOMPLETE
        return bytes(self._buffer[pos:end]), end + len(terminator)

    def _parse_array(self, pos: int) -> object:
        """Parse `*<count>\\r\\n` followed by that many bulk strings.

        Added: 2026-09-04 (Module 3)
        """
        line = self._read_line(pos)
        if line is INCOMPLETE:
            return INCOMPLETE
        header, pos = line

        count = _parse_length(header[1:], "invalid multibulk length")
        if count > MAX_ARRAY_LENGTH:
            raise ProtocolError("invalid multibulk length")
        if count <= 0:
            # *0 (empty) and *-1 (null) are both "no command here". Consuming
            # the header and moving on keeps them from wedging the stream.
            return [], pos

        parts: list[bytes] = []
        for _ in range(count):
            item = self._parse_bulk_string(pos)
            if item is INCOMPLETE:
                # Note what we throw away: `parts`, and the position we had
                # reached. Re-parsing those elements when the rest arrives costs
                # microseconds and buys us a parser with no resumption state to
                # get wrong. Redis, which cares about the microseconds, keeps
                # that state; at MiniKV's scale the simpler code wins.
                return INCOMPLETE
            value, pos = item
            parts.append(value)

        return parts, pos

    def _parse_bulk_string(self, pos: int) -> object:
        r"""Parse `$<length>\r\n<length bytes>\r\n` at `pos`.

        Added: 2026-09-04 (Module 3)
        """
        line = self._read_line(pos)
        if line is INCOMPLETE:
            return INCOMPLETE
        header, pos = line

        # Command arguments are always bulk strings. A client sending an array
        # of anything else is broken, not merely unusual, so we refuse rather
        # than guess at what it meant.
        if header[:1] != BULK_STRING_BYTE:
            raise ProtocolError(
                f"expected '$', got '{_printable(header[:1])}'"
            )

        length = _parse_length(header[1:], "invalid bulk length")
        # A negative length is a *null* bulk string, which is a valid reply but
        # not a valid argument - there is no such thing as a null argument.
        if length < 0 or length > MAX_BULK_LENGTH:
            raise ProtocolError("invalid bulk length")

        end = pos + length
        # The payload plus its trailing CRLF must both have arrived. Checking
        # the CRLF here, not just the payload, is why a value ending mid-CRLF
        # cannot be mistaken for a complete one.
        if len(self._buffer) < end + len(CRLF):
            return INCOMPLETE
        if self._buffer[end:end + len(CRLF)] != CRLF:
            raise ProtocolError("bulk string is not terminated by CRLF")

        return bytes(self._buffer[pos:end]), end + len(CRLF)

    def _parse_inline(self, pos: int) -> object:
        """Parse one whitespace-separated line, the telnet-friendly form.

        Added: 2026-09-04 (Module 3)
        """
        # LF, not CRLF: an inline command is what you get when a person types
        # into netcat or telnet, and plenty of those send a bare newline. RESP
        # proper still demands CRLF - being lenient there would let a malformed
        # length header look well-formed.
        line = self._read_line(pos, terminator=b"\n")
        if line is INCOMPLETE:
            return INCOMPLETE
        text, pos = line

        # bytes.split() with no argument splits on runs of whitespace and drops
        # empty fields, so the trailing CR of a CRLF line disappears here too.
        # Real Redis additionally honours quotes, so an inline value can contain
        # a space; MiniKV does not, because inline commands exist for poking at
        # the server by hand and not much else.
        return text.split(), pos


def _parse_length(raw: bytes, message: str) -> int:
    """Parse a RESP length header strictly. Raises ProtocolError if malformed.

    Added: 2026-09-04 (Module 3)

    int() is far too permissive to point at the network: it accepts surrounding
    whitespace, a leading '+', and underscores as digit separators, so b"1_0"
    would quietly become 10 and b" 5" would parse a length the peer never sent.
    We spell out the grammar RESP actually uses instead.
    """
    body = raw[1:] if raw[:1] == b"-" else raw
    if not body or not body.isdigit():
        raise ProtocolError(message)
    return int(raw)


def _printable(byte: bytes) -> str:
    """Render one wire byte for an error message without letting it escape.

    Added: 2026-09-04 (Module 3)

    Whatever we quote here came from the peer and goes straight back out in an
    error reply, so it cannot be allowed to contain a CRLF and cut the reply in
    half. repr() of a latin-1 decode escapes anything dangerous.
    """
    if not byte:
        return ""
    return repr(byte.decode("latin-1"))[1:-1]
