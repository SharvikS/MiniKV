"""MiniKV - a Redis-compatible key-value server, built module by module.

Created: 2026-09-04

Layout:
    minikv/server.py    the TCP server, its per-client threads and the command
                        table (Modules 1-3)
    minikv/protocol.py  RESP wire-format encoder and incremental parser (Module 3)

Everything here is standard library only - no dependencies to install.
"""

__version__ = "0.3.0"
