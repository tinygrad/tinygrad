import onnx, yaml, tempfile, time, collections, pprint, argparse
from pathlib import Path
from extra.onnx import OnnxRunner, get_onnx_ops
from extra.onnx_helpers import validate, get_example_inputs
from examples.huggingface_onnx_download import get_config

def run_huggingface_validate(onnx_model_path, config, rtol, atol):
  onnx_model = onnx.load(onnx_model_path)
  onnx_runner = OnnxRunner(onnx_model)
  inputs = get_example_inputs(onnx_runner.graph_inputs, config)
  validate(onnx_model_path, inputs, rtol=rtol, atol=atol)

def get_tolerances(file_name): # -> rtol, atol
  # TODO very high rtol atol
  if "fp16" in file_name: return 9e-2, 9e-2
  if any(q in file_name for q in ["int8", "uint8", "quantized"]): return 4, 4
  return 4e-3, 3e-2

def validate_repos(models:dict[str, tuple[Path, Path]]):
  print(f"** Validating {len(model_paths)} models **")
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
  parser = argparse.ArgumentParser(description="Huggingface ONNX Model Validator and Ops Checker")
  parser.add_argument("--check_ops", action="store_true", default=False,
                      help="Check support for ONNX operations in models from the YAML file")
  parser.add_argument("--validate", action="store_true", default=False,
                      help="Validate correctness of models from the YAML file")
  parser.add_argument("--debug", type=str, default="",
                      help="""Validates without explicitly needing a YAML or models pre-installed.
                      provide repo id (e.g. "minishlab/potion-base-8M") to validate all onnx models inside the repo
                      provide onnx model path (e.g. "minishlab/potion-base-8M/onnx/model.onnx") to validate only that one model
                      """)
  parser.add_argument("--truncate", type=int, default=-1, help="Truncate the ONNX model so intermediate results can be validated")
  parser.add_argument("--yaml", type=str, default="huggingface_repos.yaml", help="Specify the YAML file to use")
  args = parser.parse_args()

  if not (args.check_ops or args.validate or args.debug):
    parser.error("Please provide either --validate, --check_ops, or --debug.")
  if args.truncate != -1 and not args.debug:
    parser.error("--truncate and --debug should be used together for debugging")

  if args.check_ops or args.validate:
    with open(args.yaml, 'r') as f:
      data = yaml.safe_load(f)
      model_paths = {
        model_id + "/" + model["file"]: (Path(repo["path"]), Path(model["file"]))
        for model_id, repo in data["repositories"].items()
        for model in repo["files"]
        if model["file"].endswith(".onnx")
      }

    if args.check_ops:
      pprint.pprint(retrieve_op_stats(model_paths))

    if args.validate:
      validate_repos(model_paths)

  if args.debug:
    from examples.huggingface_onnx_download import download_repo_onnx_models, download_repo_configs
    path:list[str] = args.debug.split("/")
    if len(path) == 2:
      # repo id
      # validates all onnx models inside repo
      repo_id = "/".join(path)
      root_path = download_repo_onnx_models(repo_id)
      download_repo_configs(repo_id)
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
      root_path = download_repo_onnx_models(repo_id, specific_model=relative_path)
      download_repo_configs(repo_id)
      config = get_config(root_path)
      rtol, atol = get_tolerances(onnx_model)
      print(f"validating {relative_path} with truncate={args.truncate}, {rtol=}, {atol=}")
      debug_run(root_path / relative_path, args.truncate, config, rtol, atol)