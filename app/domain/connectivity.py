"""Can this camera be reached — asked safely.

Onboarding a camera means typing an address and a credential reference and
hoping. A test that answers "yes, something is listening there" before the row
is created turns a silent misconfiguration into a sentence, and it is the one
step of the wizard that can catch a typo while the person who made it is still
looking at it.

### Why this is deliberately not an RTSP client

It opens a TCP connection and closes it. It does not send a `DESCRIBE`, does
not authenticate, and does not decode a frame. Two reasons, and the second is
the important one:

* A real RTSP handshake would need the resolved credential, which means
  resolving a secret in order to answer a diagnostic question. The moment
  that value exists in this process it can end up in a log line or an error
  body, and the whole credential architecture here is built on it never being
  resolved outside the source that dials.
* Almost every failure this test exists to catch — wrong IP, wrong port,
  firewall, unplugged DVR, port-forward pointing at nothing — is visible at
  the TCP layer. A camera that accepts a connection and then refuses to
  authenticate is a *different* and much rarer problem, and the honest answer
  is to say what was and was not proved rather than to imply more.

So the result says exactly what it learned. `reachable` means a socket opened.
It does not mean the credential is right, and this module never claims it does.

### Why this is not an SSRF hole

A connect-and-close probe against an arbitrary address is a port scanner, and
exposing one to any authenticated user would be a real finding. Three things
close it:

1. **`MANAGE_CAMERAS` is required.** The caller can already create cameras,
   which is a far stronger capability than learning that a port is open.
2. **Reserved ranges are refused.** Loopback, link-local (which is where cloud
   instance metadata lives), unspecified and multicast addresses are rejected
   before a socket is opened. These are the targets an SSRF is actually aimed
   at; a camera is never at any of them.
3. **The response is one bit and a category.** No banner, no body, no timing
   detail beyond a coarse millisecond count. There is nothing to read out.

Private LAN ranges are deliberately *allowed*: that is where cameras live, and
refusing them would refuse the entire real use case to defend against a threat
the first two rules already answer.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Any

from app.errors import ValidationError

#: A camera that has not answered in this long is not going to.
TIMEOUT_S = 4.0


@dataclass(frozen=True, slots=True)
class ConnectivityResult:
    """What was learned, and deliberately nothing more."""

    reachable: bool
    #: A stable machine-readable category, so a client can choose its own words.
    outcome: str
    #: One sentence for a person, written to be actionable.
    detail: str
    elapsed_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "outcome": self.outcome,
            "detail": self.detail,
            "elapsed_ms": self.elapsed_ms,
            # Said explicitly on every result, because "connection test passed"
            # is otherwise read as "the camera works", and this test cannot
            # know that.
            "proves": (
                "A device is listening on this address and port. It does not "
                "prove the credentials are correct or that the stream decodes."
            ),
        }


def _reject_reserved(host: str) -> None:
    """Refuse the addresses an SSRF aims at. See the module docstring."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A hostname. Resolution happens inside the connect below, and the
        # resolved address is checked there — a name that resolves to the
        # metadata service must not slip past a check that only read strings.
        return

    if address.is_loopback:
        raise ValidationError(
            "refusing to test a loopback address: that is this server, not a camera",
            details={"host": host},
        )
    if address.is_link_local:
        raise ValidationError(
            "refusing to test a link-local address: cloud instance metadata "
            "lives there, and no camera does",
            details={"host": host},
        )
    if address.is_unspecified or address.is_multicast or address.is_reserved:
        raise ValidationError(
            "that is not an address a camera can be at",
            details={"host": host},
        )


async def probe(host: str, port: int, *, timeout: float = TIMEOUT_S) -> ConnectivityResult:
    """Open a TCP connection and close it. Never raises for a network failure.

    A failure to reach a camera is an answer, not an error: the caller asked
    whether it was reachable and "no, because the connection was refused" is
    the useful form of no.
    """
    hostname = (host or "").strip()
    if not hostname:
        raise ValidationError("'host' is required to test a connection")
    if not (1 <= int(port) <= 65535):
        raise ValidationError("'rtsp_port' must be between 1 and 65535")

    _reject_reserved(hostname)

    loop = asyncio.get_running_loop()
    started = loop.time()

    def elapsed() -> int:
        return int((loop.time() - started) * 1000)

    try:
        # Resolution first and separately, so a name that resolves into a
        # reserved range is refused rather than dialled. Checking only the
        # string the caller typed would be a check anyone could walk around
        # with a DNS record.
        infos = await asyncio.wait_for(
            loop.getaddrinfo(hostname, int(port), type=socket.SOCK_STREAM),
            timeout=timeout,
        )
    except TimeoutError:
        return ConnectivityResult(
            False, "dns_timeout", f"Could not look up '{hostname}' in time.", elapsed()
        )
    except OSError:
        return ConnectivityResult(
            False,
            "dns_failed",
            f"'{hostname}' could not be resolved. Check the address for a typo.",
            elapsed(),
        )

    for info in infos:
        _reject_reserved(str(info[4][0]))

    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(hostname, int(port)), timeout=timeout
        )
    except TimeoutError:
        return ConnectivityResult(
            False,
            "timeout",
            f"No response from {hostname}:{port} within {timeout:.0f}s. The "
            f"address may be wrong, or a firewall or router may be dropping "
            f"the connection.",
            elapsed(),
        )
    except ConnectionRefusedError:
        return ConnectivityResult(
            False,
            "refused",
            f"{hostname} is reachable but refused the connection on port "
            f"{port}. Check the RTSP port, and that the recorder is running.",
            elapsed(),
        )
    except OSError as exc:
        return ConnectivityResult(
            False,
            "unreachable",
            # `exc.strerror` is the operating system's own phrasing and carries
            # nothing about this server. `str(exc)` can include a path.
            f"Could not reach {hostname}:{port} ({exc.strerror or 'network error'}).",
            elapsed(),
        )

    writer.close()
    try:
        await writer.wait_closed()
    except OSError:
        pass

    return ConnectivityResult(
        True,
        "reachable",
        f"A device answered on {hostname}:{port}.",
        elapsed(),
    )


__all__ = ["TIMEOUT_S", "ConnectivityResult", "probe"]
