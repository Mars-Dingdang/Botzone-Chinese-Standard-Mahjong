#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FILES = ["botzone_entry.py", "mahjong_agent"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/ppo_model.pt")
    parser.add_argument("--output", default="artifacts/botzone_submission.zip")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write("botzone_entry.py")
        for root, dirs, files in os.walk("mahjong_agent"):
            dirs[:] = [name for name in dirs if name != "__pycache__"]
            for filename in files:
                if filename.endswith(".py"):
                    path = os.path.join(root, filename)
                    archive.write(path)
        if os.path.exists(args.model):
            archive.write(args.model, "model.pt")
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
