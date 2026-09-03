"""
MiniKV - Module 1: Bare TCP Server
Created: 2026-09-04

The smallest thing that can honestly be called a server. It claims a port,
waits for a client, and echoes back every byte that client sends.

There is no protocol here yet and no key-value store yet. The only goal of
this module is the four socket calls that every network server on earth is
built from:  socket() -> bind() -> listen() -> accept().

Run:   python3 server.py
Test:  connect on port 6379, type anything, watch it come back unchanged.
"""

import socket

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
READ_BUFFER_SIZE = 1024

# How many fully-established connections the kernel will queue for us before
# it starts refusing new ones. We only accept() one at a time in this module,
# so the queue is what keeps a second client from being rejected outright.
BACKLOG = 16


# --- Connection handling -----------------------------------------------------

def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
    """Echo everything one client sends, until that client disconnects.

    Added: 2026-09-04
    """
    print(f"[server] client connected: {addr[0]}:{addr[1]}")

    # `with` guarantees the socket is closed even if the client vanishes
    # mid-loop and recv() raises. A leaked socket is a leaked file descriptor.
    with conn:
        while True:
            # Blocking call: execution stops here until bytes arrive. Nothing
            # else in this program can make progress while we wait, which is
            # precisely the limitation Module 2 removes.
            data = conn.recv(READ_BUFFER_SIZE)

            # An empty bytes object does NOT mean "no data yet" - a blocking
            # recv() never returns empty for that reason. It means the peer
            # performed an orderly shutdown. This is our only reliable
            # end-of-stream signal.
            if not data:
                break

            # sendall() keeps looping until every byte has been handed to the
            # kernel. Plain send() may accept only part of the buffer and
            # return the count, silently dropping the rest if you ignore it.
            conn.sendall(data)

    print(f"[server] client disconnected: {addr[0]}:{addr[1]}")


# --- Server loop -------------------------------------------------------------

def main() -> None:
    """Bind the listening socket and serve clients one at a time.

    Added: 2026-09-04
    """
    # AF_INET     -> IPv4 addressing.
    # SOCK_STREAM -> TCP: an ordered, reliable, byte-oriented *stream*.
    #                Note "stream", not "messages" - TCP has no idea where one
    #                of our commands ends and the next begins. We will have to
    #                tell it, which is what the RESP protocol is for.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # Without SO_REUSEADDR the port lingers in TIME_WAIT for up to a minute
        # after we exit, and the next `python3 server.py` dies with
        # "Address already in use". This makes restarts instant.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # bind() claims the address. listen() then flips the socket from
        # "a socket" into "a socket that accepts incoming connections".
        listener.bind((HOST, PORT))
        listener.listen(BACKLOG)
        print(f"[server] listening on {HOST}:{PORT} (Ctrl-C to stop)")

        while True:
            # accept() blocks until someone connects, then returns a *new*
            # socket dedicated to that one client. The listener stays open for
            # the next arrival.
            conn, addr = listener.accept()

            # Module 1 serves strictly one client at a time. A second client
            # can connect - the kernel's backlog queue holds it - but it gets
            # no reply until the first one hangs up. Module 2 fixes this.
            handle_client(conn, addr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl-C is the normal way to stop the server. Don't scare the user
        # with a traceback for something they asked for on purpose.
        print("\n[server] shutting down")
