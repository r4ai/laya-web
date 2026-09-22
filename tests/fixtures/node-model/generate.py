"""Regenerate the tiny Node integration model with the project's Python environment."""
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

root = Path(__file__).parent
inputs = [
    helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, "seq", 1]),
    helper.make_tensor_value_info("marker_pos", TensorProto.INT64, ["markers"]),
    helper.make_tensor_value_info("marker_mask", TensorProto.BOOL, ["markers"]),
    helper.make_tensor_value_info("qtype", TensorProto.INT64, [1]),
]
nodes = [
    helper.make_node("Gather", ["embeddings", "marker_pos"], ["selected"], axis=1),
    helper.make_node("Squeeze", ["selected", "axes"], ["flat"]),
    helper.make_node("Cast", ["marker_mask"], ["mask"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["flat", "mask"], ["masked"]),
    helper.make_node("Cast", ["qtype"], ["type"], to=TensorProto.FLOAT),
    helper.make_node("Add", ["masked", "type"], ["logits"]),
    helper.make_node("Identity", ["action"], ["action_logits"]),
]
action = helper.make_tensor("action", TensorProto.FLOAT, [2], b"\0" * 8, raw=True)
onnx.external_data_helper.set_external_data(action, "model.onnx.data", offset=0, length=8)
action.ClearField("raw_data")
graph = helper.make_graph(nodes, "node-fixture", inputs, [
    helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["markers"]),
    helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [2]),
], [helper.make_tensor("axes", TensorProto.INT64, [2], [0, 2]), action])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
(root / "model.onnx").write_bytes(model.SerializeToString())
(root / "model.onnx.data").write_bytes(b"\0" * 8)
# Four FP16 rows: UNK=0, CLS=0, SEP=0, MASK=1.
(root / "embeddings.f16.bin").write_bytes(bytes.fromhex("000000000000003c"))
(root / "tokenizer/tokenizer.json").write_text(json.dumps({
    "decoder": None, "normalizer": None, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": None,
    "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}, "unk_token": "[UNK]"},
    "added_tokens": [{"id": i, "content": token, "special": True} for i, token in enumerate(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])],
}))
(root / "tokenizer/tokenizer_config.json").write_text(json.dumps({
    "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]", "unk_token": "[UNK]",
}))
files = ["model.onnx", "model.onnx.data", "embeddings.f16.bin", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
(root / "config.json").write_text(json.dumps({
    "format": "laya-web-v1", "maxLength": 64, "headMaxLength": 32,
    "hiddenSize": 1, "vocabSize": 4, "temperature": [1, 1, 1], "temperatureByOptions": {},
    "files": {name: {"bytes": len((root / name).read_bytes()), "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()} for name in files},
}))
