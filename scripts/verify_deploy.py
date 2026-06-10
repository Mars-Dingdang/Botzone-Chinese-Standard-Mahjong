#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/botzone_model.pt")
    parser.add_argument("--archive", default="artifacts/botzone_submission.zip")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if "__main__.py" not in names:
            raise RuntimeError("archive is missing root __main__.py")
        if any(name.endswith(".pt") for name in names):
            raise RuntimeError("model must be uploaded through Botzone user storage")
    payload = {
        "requests": [
            "0 1 0",
            "1 0 0 0 0 W1 W2 W3 W4 W5 W6 W7 W8 W9 T1 T2 T3 T4",
            "2 T5",
        ],
        "responses": ["PASS", "PASS"],
    }
    started = time.time()
    environment = dict(os.environ)
    environment.update({"MAHJONG_MODEL": args.model, "MAHJONG_REQUIRE_MODEL": "1"})
    output = subprocess.check_output(
        [sys.executable, args.archive], input=json.dumps(payload).encode("utf-8"),
        env=environment)
    decision_latency = time.time() - started
    response = json.loads(output.decode("utf-8"))
    if not response.get("response", "").startswith(("PLAY ", "GANG ", "BUGANG ", "HU")):
        raise RuntimeError("unexpected smoke-test response: %r" % response)
    from mahjong_agent.botzone.legality import response_to_action, validate_action
    from mahjong_agent.botzone.protocol import ProtocolState
    state = ProtocolState()
    for index, request in enumerate(payload["requests"]):
        state.apply(request, payload["responses"][index - 1] if index else None)
    valid, reason = validate_action(state, response_to_action(state, response["response"]))
    if not valid:
        raise RuntimeError("generated action failed strict validation: %s" % reason)
    own_discard_payload = {
        "requests": [
            "0 2 1",
            "1 0 0 0 0 T6 T6 T6 B2 B2 B2 B5 B5 B5 F4 F4 F4 W1",
            "2 F4",
            "3 2 PLAY F4",
        ],
        "responses": ["PASS", "PASS", "PLAY F4"],
    }
    own_discard_output = subprocess.check_output(
        [sys.executable, args.archive],
        input=json.dumps(own_discard_payload).encode("utf-8"), env=environment)
    own_discard_response = json.loads(own_discard_output.decode("utf-8"))
    if own_discard_response.get("response") != "PASS":
        raise RuntimeError("bot attempted to claim its own discard: %r" % own_discard_response)
    import torch
    from mahjong_agent.training.checkpoint import load_model_from_checkpoint
    model, metadata = load_model_from_checkpoint(args.model)
    print("archive verified; model loaded; decision_latency=%.3fs torch=%s metadata=%r" % (
        decision_latency, torch.__version__, metadata))


if __name__ == "__main__":
    main()
