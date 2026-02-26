#!/usr/bin/env python3
import os
import importlib.util
from pathlib import Path


def load_build_ui():
    root = Path(__file__).resolve().parent
    mod_path = root / "viz" / "json_argument_extractor_gui.py"
    spec = importlib.util.spec_from_file_location("json_argument_extractor_gui", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.build_ui


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7864"))
    app = load_build_ui()()
    app.launch(server_name="127.0.0.1", server_port=port)
