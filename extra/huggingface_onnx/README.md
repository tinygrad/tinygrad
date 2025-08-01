# HuggingFace ONNX

Tool for discovering, downloading, and running ONNX models from HuggingFace.

## Huggingface Manager (discovering and downloading)

```bash
# Download top 50 models
python huggingface_manager.py --limit 50 --download

# Just collect metadata (no download)
python huggingface_manager.py --limit 100

# Sort by likes instead of downloads
python huggingface_manager.py --limit 20 --sort likes --download
```

**Note**: Models are downloaded to the `models/` directory by default.

### Options

| Option | Description |
|--------|-------------|
| `--limit N` | **Required.** Number of top models to discover |
| `--download` | Download models and extract graph inputs |
| `--sort CRITERIA` | Sort by: `downloads`, `likes`, `created`, `modified` (default: `downloads`) |
| `--output FILE` | Output YAML filename (default: `huggingface_repos.yaml`) |

### Output Format

```yaml
repositories:
  "model-name":
    url: "https://huggingface.co/model-name"
    download_path: "/path/to/models/..."  # when --download used
    files:
      - file: "model.onnx"
        size: "90.91MB"
        graph_inputs:  # when --download used
          input_ids:
            shape: [1, "sequence"]
            dtype: "int64"
            is_optional: false
            is_sequence: false
total_size: "2.45GB"
created_at: "2024-01-15T10:30:00Z"
```

## Run Models (running)

Use `run_models.py` to validate and analyze downloaded models:

```bash
# Check ONNX operation support in all models
python run_models.py huggingface_repos.yaml --check_ops

# Validate model correctness against ONNX Runtime
python run_models.py huggingface_repos.yaml --validate

# Debug specific repository (downloads and validates all ONNX models)
python run_models.py --debug "sentence-transformers/all-MiniLM-L6-v2"

# Debug specific model file
python run_models.py --debug "openai-community/gpt2/onnx/decoder_model.onnx"
```

### Options

| Option | Description |
|--------|-------------|
| `input` | **Required.** Path to YAML file from huggingface_manager.py |
| `--check_ops` | Check support for ONNX operations in models from YAML file |
| `--validate` | Validate correctness of models from YAML file |
| `--debug REPO/MODEL` | Debug specific repo or model without YAML file |
| `--truncate N` | Truncate ONNX model for debugging (use with --debug) |

## Extra Dependencies

```bash
pip install huggingface_hub pyyaml requests onnx onnxruntime
```