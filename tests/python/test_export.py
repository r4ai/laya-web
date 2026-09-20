import numpy as np
import onnxruntime as ort
import torch
from architecture import DecisionModel
from export_model import export_graph


def test_dynamic_graph_matches_torch(tmp_path):
    torch.manual_seed(7)
    c = dict(
        model_type="modernbert",
        hidden_size=16,
        intermediate_size=24,
        num_attention_heads=2,
        num_hidden_layers=3,
        local_attention=4,
    )
    model = DecisionModel(c, dict(max_len=64, head_layers=1)).eval()
    export_graph(model, tmp_path / "model.onnx", 16)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"))
    for n, count in [(8, 2), (19, 3), (64, 5)]:
        for qt in range(3):
            args = (
                torch.randn(1, n, 16),
                torch.arange(count),
                torch.ones(count, dtype=torch.bool),
                torch.tensor([qt]),
            )
            with torch.no_grad():
                expected = model(*args)
            actual = session.run(
                None,
                dict(
                    zip(
                        ["embeddings", "marker_pos", "marker_mask", "qtype"],
                        [a.numpy() for a in args],
                    )
                ),
            )
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b.numpy(), atol=2e-5, rtol=1e-5)
