"""Real-ESRGAN-Netze als ONNX-Graph – aus den offiziellen Gewichten, ganz ohne PyTorch.

Die Graphen bilden die Original-Architekturen exakt nach:
``SRVGGNetCompact`` (realesr-general-x4v3) und ``RRDBNet`` (RealESRGAN_x4plus),
siehe https://github.com/xinntao/Real-ESRGAN.
"""

from __future__ import annotations

import re

import numpy as np

from .torchfile import WeightFileError

OPSET = 17


class _Graph:
    """Kleiner Helfer zum Aufbau eines ONNX-Graphen."""

    def __init__(self) -> None:
        from onnx import helper

        self.helper = helper
        self.nodes: list = []
        self.initializers: list = []
        self._counter = 0

    def name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def const(self, array: np.ndarray, prefix: str = "c") -> str:
        from onnx import numpy_helper

        name = self.name(prefix)
        self.initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def op(self, op_type: str, inputs: list[str], **attrs) -> str:
        out = self.name(op_type.lower())
        self.nodes.append(self.helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def conv(self, x: str, weight: np.ndarray, bias: np.ndarray) -> str:
        k = weight.shape[-1]
        return self.op("Conv", [x, self.const(weight, "w"), self.const(bias, "b")], kernel_shape=[k, k], pads=[k // 2] * 4)

    def upsample_nearest(self, x: str, factor: int) -> str:
        scales = self.const(np.array([1, 1, factor, factor], dtype=np.float32), "scales")
        return self.op(
            "Resize", [x, "", scales], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor"
        )

    def model_bytes(self, output: str) -> bytes:
        from onnx import TensorProto

        h = self.helper
        self.nodes.append(h.make_node("Identity", [output], ["output"]))
        graph = h.make_graph(
            self.nodes,
            "bildwerk_upscaler",
            [h.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 3, "H", "W"])],
            [h.make_tensor_value_info("output", TensorProto.FLOAT, ["N", 3, "H4", "W4"])],
            self.initializers,
        )
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", OPSET)], producer_name="bildwerk")
        model.ir_version = 8  # von allen aktuellen ONNX-Runtime-Versionen unterstützt
        return model.SerializeToString()


def _get(state: dict[str, np.ndarray], key: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if key not in state:
        raise WeightFileError(f"In den Modellgewichten fehlt „{key}“ – passt die Datei zum Modell?")
    value = state[key]
    if shape is not None and value.shape != shape:
        raise WeightFileError(f"„{key}“ hat die Form {value.shape}, erwartet {shape}.")
    return value


def srvgg_onnx(state: dict[str, np.ndarray], upscale: int = 4) -> bytes:
    """SRVGGNetCompact: Faltungen mit PReLU, PixelShuffle und Addition des Eingangsbilds."""
    convs = sorted({int(m.group(1)) for k in state if (m := re.fullmatch(r"body\.(\d+)\.bias", k))})
    if len(convs) < 2:
        raise WeightFileError("Die Gewichte passen nicht zu einem SRVGG-Netz (Real-ESRGAN General).")
    feat = _get(state, "body.0.weight").shape[0]
    out_ch = 3 * upscale * upscale
    g = _Graph()
    x = "input"
    for index in convs:
        weight = _get(state, f"body.{index}.weight")
        x = g.conv(x, weight, _get(state, f"body.{index}.bias", (weight.shape[0],)))
        if index != convs[-1]:
            slope = _get(state, f"body.{index + 1}.weight", (feat,))
            x = g.op("PRelu", [x, g.const(slope.reshape(feat, 1, 1), "slope")])
    if _get(state, f"body.{convs[-1]}.weight").shape[0] != out_ch:
        raise WeightFileError("Die letzte Faltung passt nicht zum Vergrößerungsfaktor 4.")
    x = g.op("DepthToSpace", [x], blocksize=upscale, mode="CRD")  # entspricht torch.nn.PixelShuffle
    return g.model_bytes(g.op("Add", [x, g.upsample_nearest("input", upscale)]))


def rrdbnet_onnx(state: dict[str, np.ndarray]) -> bytes:
    """RRDBNet (ESRGAN) mit Faktor 4: 23 Residual-in-Residual-Dense-Blöcke."""
    blocks = sorted({int(m.group(1)) for k in state if (m := re.match(r"body\.(\d+)\.rdb1\.conv1\.weight$", k))})
    if not blocks:
        raise WeightFileError("Die Gewichte passen nicht zu einem RRDB-Netz (RealESRGAN x4plus).")
    g = _Graph()

    def conv(x: str, key: str) -> str:
        return g.conv(x, _get(state, f"{key}.weight"), _get(state, f"{key}.bias"))

    def lrelu(x: str) -> str:
        return g.op("LeakyRelu", [x], alpha=0.2)

    def scaled_residual(x: str, residual: str) -> str:
        return g.op("Add", [g.op("Mul", [x, g.const(np.array(0.2, dtype=np.float32), "s")]), residual])

    def dense_block(x: str, key: str) -> str:
        features = [x]
        for i in range(1, 5):
            cat = g.op("Concat", features, axis=1) if len(features) > 1 else features[0]
            features.append(lrelu(conv(cat, f"{key}.conv{i}")))
        out = conv(g.op("Concat", features, axis=1), f"{key}.conv5")
        return scaled_residual(out, x)

    feat = conv("input", "conv_first")
    x = feat
    for b in blocks:
        inner = x
        for r in (1, 2, 3):
            inner = dense_block(inner, f"body.{b}.rdb{r}")
        x = scaled_residual(inner, x)
    feat = g.op("Add", [feat, conv(x, "conv_body")])
    feat = lrelu(conv(g.upsample_nearest(feat, 2), "conv_up1"))
    feat = lrelu(conv(g.upsample_nearest(feat, 2), "conv_up2"))
    return g.model_bytes(conv(lrelu(conv(feat, "conv_hr")), "conv_last"))


def blend_states(states: list[dict[str, np.ndarray]], weights: tuple[float, ...]) -> dict[str, np.ndarray]:
    """Deep Network Interpolation: gewichtete Mischung mehrerer Gewichtssätze gleicher Form."""
    keys = states[0].keys()
    if any(s.keys() != keys for s in states[1:]):
        raise WeightFileError("Die zu mischenden Modelle haben unterschiedliche Gewichte.")
    return {k: sum(w * s[k] for w, s in zip(weights, states)).astype(np.float32) for k in keys}
