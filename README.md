# MHmodel ComfyUI Workflow API

Exported ComfyUI workflow runner for Magic Hour model experiments.

## What is in this repo

- `workflow_api.py` searches for a local `ComfyUI` checkout, adds it to `sys.path`, loads optional `extra_model_paths.yaml`, and executes a generated workflow.

## Requirements

- Python 3.10+
- A working local ComfyUI installation
- PyTorch and the ComfyUI dependencies required by the workflow
- Required model checkpoints installed under the local ComfyUI model directories

## Usage

```bash
python workflow_api.py
```

Run it from a directory inside or near your ComfyUI checkout so the script can find the `ComfyUI` folder. If your model paths live outside ComfyUI, provide an `extra_model_paths.yaml` file in a parent directory.

## Notes

This repository does not include model weights or generated outputs. Keep those outside git or under ignored `models/`, `checkpoints/`, `output/`, or `outputs/` directories.
