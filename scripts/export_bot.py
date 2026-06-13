#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNTIME_FILES = [
    "botzone_entry.py",
    "mahjong_agent/__init__.py",
    "mahjong_agent/botzone/__init__.py", "mahjong_agent/botzone/protocol.py",
    "mahjong_agent/botzone/legality.py",
    "mahjong_agent/engine/__init__.py", "mahjong_agent/engine/actions.py", "mahjong_agent/engine/tiles.py",
    "mahjong_agent/engine/env.py",
    "mahjong_agent/features/__init__.py", "mahjong_agent/features/encoder.py",
    "mahjong_agent/features/token_encoder.py",
    "mahjong_agent/models/__init__.py", "mahjong_agent/models/hybrid_transformer.py",
    "mahjong_agent/models/token_transformer.py", "mahjong_agent/models/factory.py",
    "mahjong_agent/policies/__init__.py", "mahjong_agent/policies/analysis.py",
    "mahjong_agent/policies/baseline.py", "mahjong_agent/policies/model.py",
    "mahjong_agent/rules/__init__.py", "mahjong_agent/rules/backend.py",
    "mahjong_agent/rules/legality.py",
    "mahjong_agent/training/__init__.py", "mahjong_agent/training/checkpoint.py",
    "mahjong_agent/training/rollout.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/ppo_model.pt")
    parser.add_argument("--output", default="artifacts/botzone_submission.zip")
    parser.add_argument("--storage-model", default="artifacts/botzone_model.pt")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.storage_model) or ".", exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("__main__.py", "from botzone_entry import main\nmain()\n")
        for path in RUNTIME_FILES:
            archive.write(path)
    if os.path.exists(args.model):
        try:
            import torch
            payload = torch.load(args.model, map_location="cpu")
            payload["optimizer"] = None
            payload["scheduler"] = None
            torch.save(payload, args.storage_model, _use_new_zipfile_serialization=False)
        except ImportError:
            shutil.copyfile(args.model, args.storage_model)
    print("wrote source=%s storage_model=%s" % (args.output, args.storage_model))


if __name__ == "__main__":
    main()
