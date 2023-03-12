import pathlib
supported_runtimes = [x.stem[len("ops_"):] for x in pathlib.Path(__file__).parent.iterdir() if x.stem.startswith("ops_")]