#!/usr/bin/env python3
import os

from viz.soccer_app_gui import build_ui


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7863"))
    build_ui().launch(server_name="127.0.0.1", server_port=port)

