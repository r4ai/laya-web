"""Check exported tensors against pinned MLX FP32, emit browser parity fixtures."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from export_model import digest

CASES = [
    (
        "I was billed twice. Please refund the duplicate.",
        {
            "department": dict(
                type="choice",
                instructions="Which team should handle this request?",
                criteria={
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs and outages",
                    "sales": "new purchases",
                },
            ),
            "urgency": dict(
                type="score",
                instructions="How urgent is this request?",
                criteria=["not urgent", "soon", "critical"],
            ),
            "refund": dict(
                type="noul", instructions="Does the customer ask for money back?"
            ),
        },
    ),
    (
        "料金が二重に請求されています。重複分を返金してください。",
        {
            "department": dict(
                type="choice",
                instructions="この問い合わせを担当する部署は？",
                criteria=["請求・返金", "技術サポート", "営業"],
            ),
            "refund": dict(type="noul", instructions="顧客は返金を求めていますか？"),
        },
    ),
    (
        {"message": "The app crashes every time I sign in.", "history": [1, True]},
        {
            "department": dict(
                type="choice",
                instructions="Which team?",
                criteria=["billing", "technical", "sales"],
            ),
            "only": dict(
                type="choice", instructions="Choose the option", criteria=["only"]
            ),
            "rubric": dict(
                type="score",
                instructions="How severe is this?",
                criteria=[{"level": "low"}, {"level": "high"}],
            ),
        },
    ),
    (
        "hello <mask> world " * 400,
        {
            "long": dict(
                type="choice",
                instructions="Choose: <mask>",
                criteria=["a", "b", "c", "d", "e", "f"],
            ),
        },
    ),
]


def verify(path):
    import laya_mlx

    cfg = json.loads((path / "config.json").read_text())
    for name, info in cfg["files"].items():
        assert digest(path / name) == info["sha256"], f"Corrupt asset: {name}"
    # Run independently first and release the reference before loading ONNX.
    agent = laya_mlx.load(
        cfg["source"],
        revision=cfg["revision"],
        dtype="float32",
        device="cpu",
        batch_size=1,
    )
    fixtures = []
    for state, questions in CASES:
        items, _ = agent.prepare(state, questions)
        result = agent.predict(state, questions)
        from laya_mlx.agent import collate_items

        rows = []
        for item in items:
            logits, action = agent.forward(
                collate_items([item], agent.tok.pad_token_id)
            )
            rows.append(
                dict(
                    ids=item["ids"],
                    markers=item["markers"],
                    qtype=item["qtype"],
                    logits=np.asarray(logits)[0].tolist(),
                    action=np.asarray(action)[0].tolist(),
                )
            )
        fixtures.append(
            dict(state=state, questions=questions, result=result, rows=rows)
        )
    del agent
    gc.collect()
    embeddings = np.memmap(
        path / "embeddings.f16.bin",
        dtype="<f2",
        mode="r",
        shape=(cfg["vocabSize"], cfg["hiddenSize"]),
    )
    session = ort.InferenceSession(
        str(path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    errors = []
    for fixture in fixtures:
        for row in fixture["rows"]:
            markers = row["markers"]
            positions = markers if len(markers) >= 2 else [*markers, 0]
            inputs = dict(
                embeddings=np.asarray(embeddings[row["ids"]], dtype=np.float32)[None],
                marker_pos=np.array(positions, dtype=np.int64),
                marker_mask=np.arange(len(positions)) < len(markers),
                qtype=np.array([row["qtype"]], dtype=np.int64),
            )
            outputs = session.run(None, inputs)
            for actual, key in zip(outputs, ["logits", "action"]):
                reference = np.array(row[key])
                error = float(np.max(np.abs(actual[0] - reference)))
                errors.append(error)
                np.testing.assert_allclose(actual[0], reference, atol=0.002, rtol=0.001)
            assert np.argmax(outputs[0][0]) == np.argmax(row["logits"])
    evidence = dict(
        reference="laya-mlx fc1df62828a3fedf4d8229fdac1cbd85f1cdf337 FP32 CPU",
        questions=sum(len(f["rows"]) for f in fixtures),
        maxAbsoluteLogitError=max(errors),
        fixtures=fixtures,
    )
    (path / "parity.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in evidence.items() if k != "fixtures"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    verify(parser.parse_args().model)
