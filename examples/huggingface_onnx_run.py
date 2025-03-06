import onnx, yaml, tempfile, time, collections, gc, pprint
from pathlib import Path
from tinygrad.helpers import getenv
from extra.onnx import OnnxRunner, get_onnx_ops
from extra.onnx_helpers import validate, get_example_inputs
from huggingface_onnx_update import download_repo_onnx_models, download_repo_configs, get_config

def run_huggingface_validate(onnx_model_path, config, rtol, atol):
  onnx_model = onnx.load(onnx_model_path)
  onnx_runner = OnnxRunner(onnx_model)
  inputs = get_example_inputs(onnx_runner.graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def get_tolerances(file_name):
  # TODO very high rtol atol
  if "fp16" in file_name: return 9e-2, 9e-2
  if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 4e-3

def validate_repos(models:dict[str, tuple[Path, Path]]):
  print(f"** Validating {len(model_paths)} repos **")
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

def retrieve_op_stats(models:dict[str, tuple[Path, Path]]) -> dict:
  ret = {}
  op_counter = collections.Counter()
  unsupported_ops = collections.defaultdict(set)
  supported_ops = get_onnx_ops()
  print(f"** Retrieving stats from {len(model_paths)} models **")
  for model_id, (root_path, relative_path) in models.items():
    print(f"examining {model_id}")
    model_path = root_path / relative_path
    onnx_runner = OnnxRunner(onnx.load(model_path))
    for node in onnx_runner.graph_nodes:
      op_counter[node.op] += 1
      if node.op not in supported_ops:
        unsupported_ops[node.op].add(model_id)
    del onnx_runner
  ret["unsupported_ops"] = {k:list(v) for k, v in unsupported_ops.items()}
  ret["op_counter"] = op_counter.most_common()
  return ret

if __name__ == "__main__":
  do_check = getenv("CHECK_OPS")
  do_validate = getenv("VALIDATE")
  debug_model_path = getenv("MODELPATH", "")
  truncate = getenv("TRUNCATE", -1)
  assert do_check or do_validate or debug_model_path, \
    """
    Please provide either environment variable `VALIDATE`, `CHECK_OPS`, or `MODELPATH`:
    - 'CHECK_OPS=1' (to check ops for repos in "huggingface_repos.yaml" and print out a report)
    - 'VALIDATE=1' (to validate correctness for repos in "huggingface_repos.yaml")
    - 'MODELPATH=google-bert/bert-base-uncased/model.onnx' (to debug run validation on a single model)
      - optionally use 'TRUNCATE=50' with 'MODELPATH' to validate intermediate results"""

  # for running
  if do_check or do_validate:
    with open('huggingface_repos.yaml', 'r') as f:
      data = yaml.safe_load(f)
      model_paths = {
        model_id + "/" + model["model"]: (Path(repo["path"]), Path(model["model"]))
        for model_id, repo in data["repositories"].items()
        for model in repo["models"]
      }

    if do_check:
      pprint.pprint(retrieve_op_stats(model_paths))

    if do_validate:
      validate_repos(model_paths)

  # for debugging
  # `model_path` is `model.id + relative_path` ("google-bert/bert-base-uncased/model.onnx")
  if debug_model_path:
    print(f"debugging with {debug_model_path=}")
    model_id, relative_path = debug_model_path.split("/", 2)[:2], debug_model_path.split("/", 2)[2]
    onnx_file_name = model_id[-1]
    rtol, atol = get_tolerances(onnx_file_name)
    model_id = "/".join(model_id)
    root_path = download_repo_onnx_models(model_id)
    download_repo_configs(model_id)
    config = get_config(root_path)
    onnx_model = root_path / relative_path
    if truncate != -1:
      # truncates the onnx model so intermediate results can be validated
      model = onnx.load(onnx_model)
      nodes_up_to_limit = list(model.graph.node)[:truncate+1]
      new_output_values = [onnx.helper.make_empty_tensor_value_info(output_name) for output_name in nodes_up_to_limit[-1].output]
      model.graph.ClearField("node")
      model.graph.node.extend(nodes_up_to_limit)
      model.graph.ClearField("output")
      model.graph.output.extend(new_output_values)
      with tempfile.NamedTemporaryFile(suffix=onnx_model.suffix) as tmp:
        onnx.save(model, tmp.name)
        run_huggingface_validate(tmp.name, config, rtol, atol)
    else:
      run_huggingface_validate(onnx_model, config, rtol, atol)