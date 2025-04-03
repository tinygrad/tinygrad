# HuggingFace ONNX

Helper scripts to scrape and validate HuggingFace for models tagged as `ONNX`

## Steps

1. Retrieve current Top N model's metadata (`collect_metadata.py`)
2. Download the ONNX models specified by the metadata (`download_models.py`)
3. Validate the ONNX models using the metadata (`run_models.py`)

### Collect Metadata
run `python extra/huggingface_onnx/collect_metadata.py --limit 100`  
This retrieves the metadata for the top 100 HuggingFace repos and produces a yaml file at `extra/huggingface_onnx/huggingface_repos.yaml` 
Optionally use `--output` to specify a new file name
The output yaml file will have `download_path` set as `null` as the model is not downloaded

Example:
`python extra/huggingface_onnx/collect_metadata.py --limit 5 --output test.yaml`  
Output:
```yaml
repositories:
  FacebookAI/xlm-roberta-large:
    url: https://huggingface.co/FacebookAI/xlm-roberta-large
    download_path: null
    files:
    - file: onnx/model.onnx
      size: 0.55MB
    - file: onnx/model.onnx_data
      size: 2235.36MB
  ...
total_size: 6.18GB
created_at: '2025-04-03T08:10:57Z'
```

### Download Models
run `python extra/huggingface_onnx/download_models.py extra/huggingface_onnx/huggingface_repos.yaml`  
This downloads the models specified in `huggingface_repos.yaml` to `extra/huggingface_onnx/models/`.

Example (using the `test.yaml` from Collect Metadata):  
`python extra/huggingface_onnx/download_models.py extra/huggingface_onnx/test.yaml`  
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
total_size: 6.18GB
created_at: '2025-04-03T08:10:57Z'
```
Total size of the download is `6.18GB`

### Run Models
run `PYTHONPATH=. python extra/huggingface_onnx/run_models.py extra/huggingface_onnx/huggingface_repos.yaml --validate`  
This does a correctness run through all the models. Make sure to have ran `download_models.py` on the yaml before this step.  
Optionally use  
`PYTHONPATH=. python extra/huggingface_onnx/run_models.py extra/huggingface_onnx/huggingface_repos.yaml --check_ops`  
to check the op distribution of the models and  
`PYTHONPATH=. python extra/huggingface_onnx/run_models.py extra/huggingface_onnx/huggingface_repos.yaml --debug FacebookAI/xlm-roberta-large/onnx/model.onnx`   
to debug run validation on a single model.  
When using `--debug` you may also use `--truncate` to test intermediate node outputs of a model.
