"""
MiniKV - Module 2: Concurrent TCP Server

Created: 2026-09-04 (Module 1 - bare echo server)
Updated: 2026-09-04 (Module 2 - one OS thread per connection)

Module 1 could only ever talk to one client. The accept loop called
handle_client() directly, so it sat inside that client's recv() until they
hung up; anyone else who connected got queued by the kernel and ignored.

Module 2 fixes that the simplest way that actually works: hand each accepted
connection to its own thread and get straight back to accept(). The threads
block independently, so one slow client can no longer starve the others.

Run:   python3 server.py
Test:  open three clients at once - all three get echoed independently.
"""

import socket
import threading

# --- Configuration -----------------------------------------------------------

# 6379 is Redis' default port. We borrow it deliberately: from Module 3 onward
# MiniKV speaks the real Redis wire protocol, so real Redis clients will be
# able to connect without any extra flags.
HOST = "127.0.0.1"
PORT = 6379

# Maximum bytes we pull off the kernel's receive buffer in a single read.
# This is a ceiling, NOT a promise: recv() is allowed to return fewer bytes
# than we asked for, and it regularly does. Module 3 is where that stops being
# a curiosity and starts being a bug we have to design around.
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


# --- Connection handling -----------------------------------------------------

def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
    """Echo everything one client sends, until that client disconnects.

    Runs on its own thread, one per connection.

    Added:   2026-09-04 (Module 1)
    Updated: 2026-09-04 (Module 2 - connection accounting, TCP_NODELAY)
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

                # sendall() keeps looping until every byte has been handed to
                # the kernel. Plain send() may accept only part of the buffer
                # and return the count, silently dropping the rest.
                conn.sendall(data)

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
        # after we exit, and the next `python3 server.py` dies with
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
