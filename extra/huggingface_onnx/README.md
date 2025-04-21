# HuggingFace ONNX

Helper scripts to scrape and validate models tagged as `ONNX` in HuggingFace

## Files

1. Retrieve current Top N model's metadata (`collect_metadata.py`)
2. Download the models and metadata containing the size of the models and OP coverage (`download_models.py`)
3. Independent script to validate model correctness (`run.py`)

### Collect Metadata
run `python extra/huggingface_onnx/collect_metadata.py --limit 100`  
This retrieves the metadata for the top 100 HuggingFace repos and produces a yaml file at `extra/huggingface_onnx/huggingface.yaml` 
Optionally use `--output` to specify a new file name

Example:
`python extra/huggingface_onnx/collect_metadata.py --limit 5 --output test.yaml`  
Output:
```yaml
repositories:
  FacebookAI/xlm-roberta-large:
    url: https://huggingface.co/FacebookAI/xlm-roberta-large
    files:
    - file: onnx/model.onnx
      size: 0.55MB
    - file: onnx/model.onnx_data
      size: 2235.36MB
  ...
stats:
  model_ops: null
  total_op_counter: null
  unsupported_ops: null
  diverse_models: null
  total_size: 6.18GB
```

### Download Models
run `python extra/huggingface_onnx/download_models.py extra/huggingface_onnx/huggingface.yaml`  
This downloads the models specified in `huggingface.yaml` to `extra/huggingface_onnx/models/`
This also updates the `null` values in the stats of the previous yaml

Example (using the `test.yaml` from Collect Metadata):  
`PYTHONPATH=. python extra/huggingface_onnx/download_models.py extra/huggingface_onnx/test.yaml`  
Output:
```yaml
repositories:
  FacebookAI/xlm-roberta-large:
    url: https://huggingface.co/FacebookAI/xlm-roberta-large
    download_path: path/to/tinygrad/extra/huggingface_onnx/models/models--FacebookAI--xlm-roberta-large/snapshots/c23d21b0620b635a76227c604d44e43a9f0ee389
    files:
    - file: onnx/model.onnx
      size: 0.55MB
    - file: onnx/model.onnx_data
      size: 2235.36MB
  ...
stats:
  model_ops:
    FacebookAI/xlm-roberta-large/onnx/model.onnx:
      Constant: 567
      Add: 341
      ...
    sentence-transformers/all-MiniLM-L6-v2/onnx/model_O2.onnx:
      MatMul: 48
      Add: 26
      ...
    ...
  total_op_counter:
    Constant: 1944
    Add: 1591
    MatMul: 1060
    ...
  unsupported_ops: []
  diverse_models:
  - FacebookAI/xlm-roberta-large/onnx/model.onnx
  - sentence-transformers/all-mpnet-base-v2/onnx/model_O2.onnx
  - sentence-transformers/all-MiniLM-L6-v2/onnx/model_O3.onnx
  total_size: 6.50GB
```
Total size of the download is `6.50GB`
`diverse_models` is selected by trying to maximize the op coverage.  
You can optionally increase the number of diverse_models by using `--diversity` to set your desired amount. However, if the remaining models' operations are subsets of the currently covered ops, the actual number of diverse_models may be less than specified.

### Run
run `PYTHONPATH=. python extra/huggingface_onnx/run.py FacebookAI/xlm-roberta-large/onnx/model.onnx sentence-transformers/all-mpnet-base-v2/onnx/model_O2.onnx`    
This runs and validates both `FacebookAI/xlm-roberta-large/onnx/model.onnx` and `sentence-transformers/all-mpnet-base-v2/onnx/model_O2.onnx`  

This also independently downloads the onnx model if you haven't done so before.  
You may also pass in only the `model_id` like `pszemraj/long-t5-tglobal-base-sci-simplify` to test all the models in a repo.  
