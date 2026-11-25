"""Authenticated-multicast transport carrier; no RBC or agreement rounds.

Length-framed canonical JSON avoids deserializing executable pickle objects.
Protocol authentication is checked by NativeAuthentication before dispatch.
Retries only establish a connection; they never choose a protocol value.
"""
import asyncio
from collections import defaultdict
from beaver.baselines.ru22_certificate import canonical
from beaver.baselines.ru22_dpss import _decode_object


class NativeTransport:
    def __init__(self, peers, node, *, max_frame_bytes=256 * 1024 * 1024):
        self.peers, self.node, self.max_frame_bytes = peers, node, max_frame_bytes
        self.inbox = asyncio.Queue()
        self.bytes_sent = defaultdict(int)
        self.messages_sent = defaultdict(int)
        self.server = None
        self.tasks = set()
        self.errors = []

    async def start(self, *, bind_host="0.0.0.0"):
        host, port = self.peers[self.node].rsplit(":", 1)
        self.server = await asyncio.start_server(self._receive, host=bind_host, port=int(port))
        # Python 3.8 can suppress socket creation errors and return an empty
        # server. Fail at startup instead of timing out in the protocol barrier.
        if not self.server.sockets:
            await self.close()
            raise OSError("native TCP listener has no sockets; check bind permissions")

    async def _receive(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            # A physical node can perform expensive synchronous crypto for
            # several roles. Per-frame wall-clock deadlines can therefore
            # close a valid connection while its peer is still computing or
            # draining a large frame. The runner owns the startup/online
            # deadlines; close() cancels any incomplete receive on shutdown.
            size = int.from_bytes(await reader.readexactly(8), "big")
            if not 0 < size <= self.max_frame_bytes:
                raise ValueError("native frame outside configured bound")
            payload = await reader.readexactly(size)
            self.inbox.put_nowait(_decode_object(payload, name="native network frame"))
        except (ValueError, ConnectionError, asyncio.IncompleteReadError, asyncio.TimeoutError) as exc:
            self.errors.append(type(exc).__name__)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self.tasks.discard(task)

    async def multicast(self, recipients, envelope):
        recipients = tuple(recipients)
        if len(set(recipients)) != len(recipients):
            raise ValueError("duplicate multicast recipient")
        if any(type(peer) is not int or not 0 <= peer < len(self.peers) for peer in recipients):
            raise ValueError("unknown multicast recipient")
        if not isinstance(envelope.get("phase"), str):
            raise ValueError("missing multicast phase")
        payload = canonical(envelope)
        if len(payload) > self.max_frame_bytes:
            raise ValueError("native message exceeds frame bound")
        data = len(payload).to_bytes(8, "big") + payload
        async def send(peer):
            if peer == self.node:
                self.inbox.put_nowait(_decode_object(payload, name="native local frame"))
                return
            host, port = self.peers[peer].rsplit(":", 1)
            while True:
                try:
                    _, writer = await asyncio.open_connection(host, int(port))
                    break
                except (OSError, ConnectionError):
                    await asyncio.sleep(.05)
            try:
                writer.write(data)
                await writer.drain()
                phase = envelope["phase"]
                self.bytes_sent[phase] += len(data)
                self.messages_sent[phase] += 1
            finally:
                writer.close()
                await writer.wait_closed()
        sends = [asyncio.create_task(send(peer)) for peer in recipients]
        try:
            await asyncio.gather(*sends)
        finally:
            for task in sends:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*sends, return_exceptions=True)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks), return_exceptions=True)
