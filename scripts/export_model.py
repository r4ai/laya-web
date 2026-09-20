"""Create a self-hostable browser checkpoint. Run with `pnpm model:export`."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from architecture import DecisionModel, load_weights

SOURCE = "convaiinnovations/laya-multilingual"
REVISION = "052592a15d198d9ad47da779604259b10b47b7aa"


def export_graph(model, path, hidden_size):
    args = (
        torch.zeros(1, 8, hidden_size),
        torch.tensor([2, 4]),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([0]),
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            str(path),
            opset_version=17,
            dynamo=False,
            input_names=["embeddings", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "action_logits"],
            dynamic_axes={
                "embeddings": {1: "sequence"},
                "marker_pos": {0: "options"},
                "marker_mask": {0: "options"},
                "logits": {1: "options"},
            },
        )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    onnx.save_model(
        graph,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export(output, source=SOURCE, revision=REVISION):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    local = Path(source)
    if not local.is_dir():
        local = Path(
            snapshot_download(
                source,
                revision=revision,
                allow_patterns=[
                    "model.safetensors",
                    "encoder/config.json",
                    "rl_agent_config.json",
                    "tokenizer/*",
                ],
            )
        )
    c = json.loads((local / "encoder/config.json").read_text())
    a = json.loads((local / "rl_agent_config.json").read_text())
    if not 4 < a["head_max_len"] < a["max_len"] <= c["max_position_embeddings"]:
        raise ValueError("Invalid checkpoint context limits")
    model = DecisionModel(c, a).eval()
    embedding = load_weights(model, load_file(str(local / "model.safetensors")))
    output.mkdir(parents=True)
    try:
        embedding.half().numpy().astype("<f2").tofile(output / "embeddings.f16.bin")
        del embedding
        export_graph(model, output / "model.onnx", c["hidden_size"])
        shutil.copytree(local / "tokenizer", output / "tokenizer")
        config = dict(
            format="laya-web-v1",
            maxLength=a["max_len"],
            headMaxLength=a["head_max_len"],
            hiddenSize=c["hidden_size"],
            vocabSize=c["vocab_size"],
            temperature=a.get("temperature", [1, 1, 1]),
            temperatureByOptions=a.get("temperature_by_options", {}),
            source=source,
            revision=revision,
            precision="fp32; fp16 embedding storage",
        )
        paths = [
            "model.onnx",
            "model.onnx.data",
            "embeddings.f16.bin",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        ]
        config["files"] = {
            p: {"bytes": (output / p).stat().st_size, "sha256": digest(output / p)}
            for p in paths
        }
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(config, indent=2))
    except BaseException:
        shutil.rmtree(output)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", default=SOURCE, help="Original or MLX checkpoint directory / Hub id"
    )
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    export(args.output, args.source, args.revision)
