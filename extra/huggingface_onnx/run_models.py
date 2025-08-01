import onnx, yaml, tempfile, time, argparse, json
from huggingface_hub import snapshot_download
from pathlib import Path
from typing import Any
from tinygrad.frontend.onnx import OnnxRunner
from tinygrad.tensor import Tensor
from extra.onnx_helpers import validate, get_example_inputs

def get_config(root_path: Path) -> dict[str, Any]:
  ret = {}
  for path in root_path.rglob("*config.json"):
    config = json.load(path.open())
    if isinstance(config, dict):
      ret.update(config)
  return ret

def get_tolerances(file_name: str) -> tuple[float, float]:
  # TODO very high rtol atol
  if "fp16" in file_name: return 9e-2, 9e-2
  if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 3e-2

def run_huggingface_validate(onnx_model_path: str | Path, config: dict[str, Any], rtol: float, atol: float):
  onnx_runner = OnnxRunner(onnx_model_path)
  inputs = get_example_inputs(onnx_runner.graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def run_huggingface_model(onnx_model_path: str | Path,
                         custom_inputs: dict[str, Any] | None = None,
                         config: dict[str, Any] | None = None,
                         rtol: float | None = None,
                         atol: float | None = None,
                         validate_against_ort: bool = True) -> dict[str, Tensor]:
  onnx_model_path = Path(onnx_model_path)

  # Load config if not provided
  if config is None:
    model_dir = onnx_model_path.parent
    config = get_config(model_dir)

  # Get tolerances if not provided
  if rtol is None or atol is None:
    auto_rtol, auto_atol = get_tolerances(onnx_model_path.name)
    rtol = rtol or auto_rtol
    atol = atol or auto_atol

  # Create ONNX runner
  onnx_runner = OnnxRunner(onnx_model_path)

  # Use custom inputs or generate them
  if custom_inputs is not None:
    # Convert numpy arrays to Tensors if needed
    inputs = {}
    for name, value in custom_inputs.items():
      if isinstance(value, Tensor):
        inputs[name] = value
      else:
        inputs[name] = Tensor(value, requires_grad=False)
  else:
    # Auto-generate inputs
    inputs = get_example_inputs(onnx_runner.graph_inputs, config)

  # Run the model
  outputs = onnx_runner(inputs)

  # Validate against ORT if requested
  if validate_against_ort:
    try:
      validate(onnx_model_path, {k: v.numpy() if isinstance(v, Tensor) else v for k, v in inputs.items()},
               rtol=rtol, atol=atol)
      print(f"Validation passed (rtol={rtol}, atol={atol})")
    except Exception as e:
      print(f"WARNING: Validation failed: {e}")

  return outputs

def download_and_test_model(repo_id: str,
                           model_file: str | None = None,
                           custom_inputs: dict[str, Any] | None = None,
                           download_dir: str | Path | None = None) -> dict[str, Tensor]:
  if download_dir is None:
    download_dir = Path(__file__).parent / "models"
  else:
    download_dir = Path(download_dir)

  # Download model files
  if model_file:
    root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[model_file], cache_dir=download_dir))
    model_path = root_path / model_file
  else:
    root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=["*.onnx"], cache_dir=download_dir))
    # Find the first .onnx file
    onnx_files = list(root_path.rglob("*.onnx"))
    if not onnx_files:
      raise ValueError(f"No .onnx files found in {repo_id}")
    model_path = onnx_files[0]

  # Download config files
  snapshot_download(repo_id=repo_id, allow_patterns=["*config.json"], cache_dir=download_dir)

  # Load config
  config = get_config(root_path)

  print(f"Testing {repo_id} model: {model_path.relative_to(root_path)}")

  # Run the model
  return run_huggingface_model(model_path, custom_inputs=custom_inputs, config=config)

def validate_repos(models:dict[str, tuple[Path, Path]]):
  print(f"** Validating {len(models)} models **")
  for model_id, (root_path, relative_path) in models.items():
    print(f"validating model {model_id}")
    model_path = root_path / relative_path
    onnx_file_name = model_path.stem
    config = get_config(root_path)
    rtol, atol = get_tolerances(onnx_file_name)
    st = time.time()
    run_huggingface_validate(model_path, config, rtol, atol)
    et = time.time() - st
    print(f"passed, took {et:.2f}s")

def debug_run(model_path, truncate, config, rtol, atol):
  if truncate != -1:
    model = onnx.load(model_path)
    nodes_up_to_limit = list(model.graph.node)[:truncate + 1]
    new_output_values = [onnx.helper.make_empty_tensor_value_info(output_name) for output_name in nodes_up_to_limit[-1].output]
    model.graph.ClearField("node")
    model.graph.node.extend(nodes_up_to_limit)
    model.graph.ClearField("output")
    model.graph.output.extend(new_output_values)
    with tempfile.NamedTemporaryFile(suffix=model_path.suffix) as tmp:
      onnx.save(model, tmp.name)
      run_huggingface_validate(tmp.name, config, rtol, atol)
  else:
    run_huggingface_validate(model_path, config, rtol, atol)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Huggingface ONNX Model Validator")
  parser.add_argument("input", type=str, help="Path to the input YAML configuration file containing model information.")
  parser.add_argument("--validate", action="store_true", default=False,
                      help="Validate correctness of models from the YAML file")
  parser.add_argument("--debug", type=str, default="",
                      help="""Validates without explicitly needing a YAML or models pre-installed.
                      provide repo id (e.g. "minishlab/potion-base-8M") to validate all onnx models inside the repo
                      provide onnx model path (e.g. "minishlab/potion-base-8M/onnx/model.onnx") to validate only that one model
                      """)
  parser.add_argument("--truncate", type=int, default=-1, help="Truncate the ONNX model so intermediate results can be validated")
  args = parser.parse_args()

  if not (args.validate or args.debug):
    parser.error("Please provide either --validate or --debug.")
  if args.truncate != -1 and not args.debug:
    parser.error("--truncate and --debug should be used together for debugging")

  if args.validate:
    with open(args.input, 'r') as f:
      data = yaml.safe_load(f)
      assert all(repo["download_path"] is not None for repo in data["repositories"].values()), "please run `download_models.py` for this yaml"
      model_paths = {
        model_id + "/" + model["file"]: (Path(repo["download_path"]), Path(model["file"]))
        for model_id, repo in data["repositories"].items()
        for model in repo["files"]
        if model["file"].endswith(".onnx")
      }

    validate_repos(model_paths)

  if args.debug:
    from huggingface_hub import snapshot_download
    download_dir = Path(__file__).parent / "models"
    path:list[str] = args.debug.split("/")
    if len(path) == 2:
      # repo id
      # validates all onnx models inside repo
      repo_id = "/".join(path)
      root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=["*.onnx", "*.onnx_data"], cache_dir=download_dir))
      snapshot_download(repo_id=repo_id, allow_patterns=["*config.json"], cache_dir=download_dir)
      config = get_config(root_path)
      for onnx_model in root_path.rglob("*.onnx"):
        rtol, atol = get_tolerances(onnx_model.name)
        print(f"validating {onnx_model.relative_to(root_path)} with truncate={args.truncate}, {rtol=}, {atol=}")
        debug_run(onnx_model, -1, config, rtol, atol)
    else:
      # model id
      # only validate the specified onnx model
      onnx_model = path[-1]
      assert path[-1].endswith(".onnx")
      repo_id, relative_path = "/".join(path[:2]), "/".join(path[2:])
      root_path = Path(snapshot_download(repo_id=repo_id, allow_patterns=[relative_path], cache_dir=download_dir))
      snapshot_download(repo_id=repo_id, allow_patterns=["*config.json"], cache_dir=download_dir)
      config = get_config(root_path)
      rtol, atol = get_tolerances(onnx_model)
      print(f"validating {relative_path} with truncate={args.truncate}, {rtol=}, {atol=}")
      debug_run(root_path / relative_path, args.truncate, config, rtol, atol)