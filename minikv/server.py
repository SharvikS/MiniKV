"""
MiniKV - Module 3: Speaking RESP

Created: 2026-09-04 (Module 1 - bare echo server)
Updated: 2026-09-04 (Module 2 - one OS thread per connection)
Updated: 2026-09-04 (Module 3 - RESP framing and a command table)

Modules 1 and 2 echoed bytes. That was the point: get a socket open, get many
clients served, and prove we are moving bytes rather than strings. But an echo
server never has to answer the only question TCP refuses to answer for us -
where does one command end and the next begin?

Module 3 answers it. Bytes off the wire go into a RESPParser (minikv/protocol.py)
that hands back whole commands and quietly keeps any half-command until the rest
arrives; whole commands go into dispatch(), which looks them up in a table and
returns encoded bytes. The store itself is still missing - GET and SET arrive in
Module 4 - so the vocabulary here is small on purpose. What matters is that the
shape is now right: read, frame, dispatch, encode, write.

The payoff is that MiniKV is now a real Redis endpoint. `redis-cli ping` works,
and so does netcat, because the parser accepts inline commands too.

Run:   python3 -m minikv.server
Test:  redis-cli -p 6379 ping            ->  PONG
       redis-cli -p 6379 echo hello      ->  "hello"
       printf 'PING\r\n' | nc 127.0.0.1 6379
"""

import socket
import threading

from minikv import protocol

# --- Configuration -----------------------------------------------------------

# 6379 is Redis' default port. We borrow it deliberately: from Module 3 onward
# MiniKV speaks the real Redis wire protocol, so real Redis clients will be
# able to connect without any extra flags.
HOST = "127.0.0.1"
PORT = 6379

# Maximum bytes we pull off the kernel's receive buffer in a single read.
# This is a ceiling, NOT a promise: recv() is allowed to return fewer bytes than
# we asked for, and it regularly does. As of Module 3 that no longer matters to
# the code below - whatever arrives goes into the parser, which is the one place
# that knows what a whole command looks like. The number is now purely a
# throughput knob: too small and we make extra syscalls, too large and every
# idle connection holds a buffer it is not using.
READ_BUFFER_SIZE = 4096

# How many fully-established connections the kernel queues before refusing new
# ones. Now that we accept() in a tight loop this queue drains fast, but a
# burst of simultaneous connects can still outrun us for a few microseconds.
BACKLOG = 128


# --- Logging -----------------------------------------------------------------

def log(message: str) -> None:
    """Write one complete log line, so concurrent threads cannot interleave.

    Added: 2026-09-04 (Module 2)

    print("hello") is really two writes - the text, then the newline - so two
    threads printing at the same instant can splice their output together
    mid-line. You can watch it happen with three simultaneous clients. Building
    the finished line first and handing it over in a single write avoids it.
    """
    print(message + "\n", end="", flush=True)


# --- Shared server state -----------------------------------------------------
#
# Added: 2026-09-04 (Module 2)
#
# This is the first state in MiniKV touched by more than one thread, so it is
# also the first place we need a lock. A counter looks harmless, but
#
#     _active_connections += 1
#
# is really three operations - read, add, write back - and two threads can
# interleave them so that one increment vanishes. The lock makes the trio
# indivisible. Module 4 applies the exact same reasoning to the key-value dict.

_state_lock = threading.Lock()
_next_connection_id = 1     # monotonically increasing label for log lines
_active_connections = 0     # how many client threads are alive right now


def _register_connection() -> tuple[int, int]:
    """Claim a connection id and count the new client. Returns (id, active).

    Added: 2026-09-04
    """
    global _next_connection_id, _active_connections
    with _state_lock:
        connection_id = _next_connection_id
        _next_connection_id += 1
        _active_connections += 1
        return connection_id, _active_connections


def _unregister_connection() -> int:
    """Count a client out. Returns how many are still connected.

    Added: 2026-09-04
    """
    global _active_connections
    with _state_lock:
        _active_connections -= 1
        return _active_connections


# --- Commands ----------------------------------------------------------------
#
# Added: 2026-09-04 (Module 3)
#
# Every handler has the same shape: it receives the arguments *after* the
# command name, already decoded from RESP into byte strings, and returns the
# encoded reply. It never touches the socket. That separation is what lets the
# whole command set be tested without opening a connection, and it is why
# Module 4 can add GET and SET as two more entries in the table below rather
# than as two more branches inside the network loop.
#
# Handlers return (reply, close_connection). Only QUIT sets the flag, but the
# alternative - having the network loop re-inspect the command name to notice -
# would put knowledge of one command back in the transport layer.


def _cmd_ping(args: list[bytes]) -> tuple[bytes, bool]:
    """PING [message] - liveness check, and the traditional first command.

    Added: 2026-09-04 (Module 3)

    With an argument, PING echoes it back as a bulk string instead of replying
    +PONG. That is not decoration: it is how a client with several requests in
    flight can put a known token into the stream and find its place again.
    """
    if not args:
        return protocol.PONG, False
    if len(args) == 1:
        return protocol.bulk_string(args[0]), False
    return _wrong_arity("ping"), False


def _cmd_echo(args: list[bytes]) -> tuple[bytes, bool]:
    """ECHO message - send one argument straight back.

    Added: 2026-09-04 (Module 3)

    Module 1's whole server, reduced to a single command. Worth keeping around:
    it round-trips one argument through the parser and the encoder, so if a
    binary value survives ECHO the wire format is honest about being binary safe.
    """
    if len(args) != 1:
        return _wrong_arity("echo"), False
    return protocol.bulk_string(args[0]), False


def _cmd_command(args: list[bytes]) -> tuple[bytes, bool]:
    """COMMAND [...] - introspection, answered with an empty list.

    Added: 2026-09-04 (Module 3)

    redis-cli sends COMMAND DOCS the moment it connects, to build tab
    completion. Real Redis answers with a description of every command it knows;
    an empty array is a truthful "I am not telling you", and the client accepts
    it and carries on. Without this, the very first thing a user tries prints an
    error before they have typed a command.
    """
    return protocol.EMPTY_ARRAY, False


def _cmd_quit(args: list[bytes]) -> tuple[bytes, bool]:
    """QUIT - acknowledge, then hang up.

    Added: 2026-09-04 (Module 3)

    The reply goes out before the close, which is the entire point of the
    command: the client learns the server got everything it sent, rather than
    guessing from a socket that just went away.
    """
    return protocol.OK, True


# The dispatch table. Names are upper case because that is what we normalise to;
# Redis command names are case-insensitive on the wire.
COMMANDS = {
    "PING": _cmd_ping,
    "ECHO": _cmd_echo,
    "COMMAND": _cmd_command,
    "QUIT": _cmd_quit,
}


def _wrong_arity(name: str) -> bytes:
    """The standard reply for a command called with the wrong argument count.

    Added: 2026-09-04 (Module 3)
    """
    return protocol.error(f"ERR wrong number of arguments for '{name}' command")


def _quote(raw: bytes) -> str:
    """Render a client-supplied byte string for inclusion in an error message.

    Added: 2026-09-04 (Module 3)

    These bytes came from the network and are about to go back out inside a
    simple-string error reply, where an embedded CRLF would truncate the reply
    and desynchronise the client. So: latin-1 decodes any byte without raising
    (every byte is a code point), unicode_escape turns the control characters
    into visible backslash escapes, and the apostrophes get escaped by hand
    because the message wraps this in quotes. Long arguments are truncated - an
    error message is not a mirror.
    """
    text = raw[:32].decode("latin-1").encode("unicode_escape").decode("ascii")
    quoted = text.replace("'", "\\'")
    return quoted + "..." if len(raw) > 32 else quoted


def dispatch(command: list[bytes]) -> tuple[bytes, bool]:
    """Route one parsed command to its handler. Returns (reply, close).

    Added: 2026-09-04 (Module 3)

    The command name is bytes off the wire, so it can be any garbage at all;
    latin-1 decodes any byte without raising, and upper() normalises the case
    that clients are free to choose.
    """
    name = command[0].decode("latin-1").upper()
    handler = COMMANDS.get(name)

    if handler is None:
        # Redis' wording, including the trailing comma - client test suites and
        # more than one shell script match on this string.
        args = "".join(f"'{_quote(arg)}', " for arg in command[1:])
        return protocol.error(
            f"ERR unknown command '{_quote(command[0])}', "
            f"with args beginning with: {args}"
        ), False

    return handler(command[1:])


# --- Connection handling -----------------------------------------------------

def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
    """Read, frame, dispatch and answer one client's commands until they hang up.

    Runs on its own thread, one per connection.

    Added:   2026-09-04 (Module 1)
    Updated: 2026-09-04 (Module 2 - connection accounting, TCP_NODELAY)
    Updated: 2026-09-04 (Module 3 - RESP framing instead of echoing)
    """
    connection_id, active = _register_connection()
    peer = f"{addr[0]}:{addr[1]}"
    log(f"[server] #{connection_id} connected from {peer} ({active} active)")

    try:
        # `with` guarantees the socket is closed even if the client vanishes
        # mid-loop and recv() raises. A leaked socket is a leaked file
        # descriptor, and a server that leaks those eventually stops serving.
        with conn:
            # Disable Nagle's algorithm. Nagle buffers small writes to avoid
            # flooding the network with tiny packets, which is great for bulk
            # transfer and terrible for a request/response protocol: it can
            # sit on our reply for up to 40ms waiting for more data that will
            # never come. This one line is worth real latency in Module 10.
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # One parser per connection, because the half-command it may be
            # holding belongs to this connection and no other. This is also the
            # only per-client state in the server - so far.
            parser = protocol.RESPParser()

            while True:
                # Blocking call. Only THIS thread sleeps here now - the accept
                # loop and every other client thread keep running.
                data = conn.recv(READ_BUFFER_SIZE)

                # An empty bytes object does NOT mean "no data yet" - a
                # blocking recv() never returns empty for that reason. It
                # means the peer performed an orderly shutdown. This is our
                # only reliable end-of-stream signal.
                if not data:
                    break

                parser.feed(data)

                # Collect this batch's replies rather than writing each one as
                # we go. A pipelining client can send fifty commands in a single
                # packet, and with TCP_NODELAY set every individual sendall()
                # would be its own segment on the wire - fifty syscalls and
                # fifty packets to answer one read.
                replies: list[bytes] = []
                closing = False

                try:
                    for command in parser.commands():
                        reply, close_after = dispatch(command)
                        replies.append(reply)
                        if close_after:
                            closing = True
                            # Anything the client pipelined behind QUIT is
                            # deliberately dropped: they told us they were done.
                            break

                except protocol.ProtocolError as exc:
                    # The framing is broken, so every byte after this point is
                    # untrustworthy - there is no resynchronisation point in a
                    # length-prefixed stream. Say why, then hang up.
                    log(f"[server] #{connection_id} protocol error: {exc}")
                    replies.append(protocol.error(f"ERR Protocol error: {exc}"))
                    closing = True

                if replies:
                    # sendall() keeps looping until every byte has been handed
                    # to the kernel. Plain send() may accept only part of the
                    # buffer and return the count, silently dropping the rest.
                    conn.sendall(b"".join(replies))

                if closing:
                    break

    except OSError as exc:
        # A client that is killed rather than closed cleanly (Ctrl-C in netcat,
        # a dropped Wi-Fi link) surfaces as ECONNRESET here. That is one
        # client's problem, not the server's - log it and let the thread end.
        log(f"[server] #{connection_id} dropped: {exc}")

    finally:
        # `finally` matters: without it, any unexpected exception would leak a
        # phantom entry in our active-connection count forever.
        remaining = _unregister_connection()
        log(f"[server] #{connection_id} disconnected ({remaining} active)")


# --- Server loop -------------------------------------------------------------

def main() -> None:
    """Bind the listening socket and hand every client to its own thread.

    Added:   2026-09-04 (Module 1)
    Updated: 2026-09-04 (Module 2 - spawn a thread per connection)
    """
    # AF_INET     -> IPv4 addressing.
    # SOCK_STREAM -> TCP: an ordered, reliable, byte-oriented *stream*.
    #                Note "stream", not "messages" - TCP has no idea where one
    #                of our commands ends and the next begins. We have to say
    #                so ourselves, which is what the RESP protocol is for.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # Without SO_REUSEADDR the port lingers in TIME_WAIT for up to a minute
        # after we exit, and the next `python3 -m minikv.server` dies with
        # "Address already in use". This makes restarts instant.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # bind() claims the address. listen() then flips the socket from
        # "a socket" into "a socket that accepts incoming connections".
        listener.bind((HOST, PORT))
        listener.listen(BACKLOG)
        log(f"[server] listening on {HOST}:{PORT} (Ctrl-C to stop)")

        while True:
            # accept() blocks until someone connects, then returns a *new*
            # socket dedicated to that one client. The listener stays open.
            conn, addr = listener.accept()

            # daemon=True means these threads do not keep the process alive.
            # When the main thread exits on Ctrl-C, the interpreter tears them
            # down instead of hanging forever waiting on idle clients' recv().
            #
            # The cost of this design: one OS thread (~8 MB of virtual stack,
            # a real scheduler entry) per connection. Fine for tens or
            # hundreds of clients, wasteful at ten thousand - which is exactly
            # why real Redis uses a single-threaded event loop instead. That
            # tradeoff is the story Module 10's README gets to tell.
            thread = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                name=f"minikv-client-{addr[0]}:{addr[1]}",
                daemon=True,
            )
            thread.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C is the normal way to stop the server. Don't scare the user
        # with a traceback for something they asked for on purpose.
        log("\n[server] shutting down")
