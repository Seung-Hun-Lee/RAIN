"""Official OpenPI msgpack websocket protocol with bounded waits.

Only `actions` is required in replies. Additional metadata fields such as
`model`, `camera_views`, and `input_evidence` are optional.
"""
import hashlib
import numpy as np


def validate_actions(reply):
    if not isinstance(reply, dict) or "actions" not in reply:
        raise ValueError("Policy reply must be a mapping containing actions")
    actions = np.asarray(reply["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite float actions [T,7], T>=1; received {actions.shape}")
    return actions


def input_evidence(payload):
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for key, value in payload.items() if isinstance(value, np.ndarray)}


class OpenPIClient:
    def __init__(self, host="127.0.0.1", port=8000, timeout=900):
        from ._vendor.openpi import msgpack_numpy
        from websockets.sync.client import connect
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        self.timeout = timeout
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.socket = connect(uri, compression=None, max_size=None, ping_interval=None, open_timeout=30)
        self.metadata = self.codec.unpackb(self.socket.recv(timeout=30))

    def infer(self, payload):
        self.socket.send(self.packer.pack(payload))
        value = self.socket.recv(timeout=self.timeout)
        if isinstance(value, str):
            raise RuntimeError(f"Policy server error: {value}")
        return self.codec.unpackb(value)

    def reset(self):
        # Official OpenPI websocket wire protocol has no episode-reset command.
        pass

    def close(self):
        self.socket.close()
