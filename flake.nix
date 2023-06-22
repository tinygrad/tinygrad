{
  description =
    "tinygrad: For something between PyTorch and karpathy/micrograd. Maintained by tiny corp.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils }@I:
    flake-utils.lib.eachDefaultSystem (system:
      let
        P = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        B = builtins;
        L = P.lib;

        pythonVersion = "310";
        pythonPackageStr = "python${pythonVersion}";
        pythonNixpkgs = P."${pythonPackageStr}Packages";

        tinygrad = pythonNixpkgs.buildPythonPackage {
          pname = "tinygrad";
          version = "v0.6.0";
          src = ./.;

          propagatedBuildInputs = with pythonNixpkgs; [
            networkx
            numpy
            pillow
            pyopencl
            pyyaml
            requests
            tqdm
          ];

          # TODO(b7r6): enable when all tests pass...
          doCheck = false;
          pythonImportsCheck = [ "tinygrad" ];
        };

        commonBuildInputs = with P; [ git nixfmt ];
        commonPythonPackages = pypkgs:
          with pypkgs; [
            # test and development
            flake8
            mypy
            onnx
            onnxruntime
            opencv4
            pylint
            pytest
            pytest-xdist
            pytorch-bin
            safetensors

            # for LLaMA
            sentencepiece

            # package
            tinygrad
          ];

        shellHook = ''
          export __NIX_PS1__="tinygrad";
          source "${P.bash-completion}/etc/profile.d/bash_completion.sh";
          source "${P.git}/share/bash-completion/completions/git";
          SCRIPT_DIR="${B.toString ./.}"
          export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
        '';

      in {
        packages = rec {
          inherit tinygrad;
          default = tinygrad;
        };

        devShells = rec {
          cpu = P.mkShell rec {
            name = tinygrad.pname;
            python = P."${pythonPackageStr}".withPackages
              (pypkgs: (commonPythonPackages pypkgs));
            buildInputs = commonBuildInputs ++ [ python ];
            inherit shellHook;
          };

          cuda = P.mkShell rec {
            name = tinygrad.pname;
            cudaVersion = "11_6";
            cudaPackagesStr = "cudaPackages_${cudaVersion}";
            python = P."${pythonPackageStr}".withPackages (pypkgs:
              ((commonPythonPackages pypkgs) ++ (with pypkgs; [ pycuda ])));
            buildInputs = commonBuildInputs ++ [
              python
              P."${cudaPackagesStr}".cudatoolkit
              P."${cudaPackagesStr}".cuda_nvcc
            ];
            inherit shellHook;
          };

          llvm = P.mkShell rec {
            name = tinygrad.pname;
            python = P."${pythonPackageStr}".withPackages (pypkgs:
              ((commonPythonPackages pypkgs) ++ (with pypkgs; [ llvmlite ])));
            buildInputs = commonBuildInputs
              ++ [ python P.llvmPackages_13.clang ];
            inherit shellHook;
          };

          triton = P.mkShell rec {
            name = tinygrad.pname;
            python = P."${pythonPackageStr}".withPackages
              (pypkgs: (commonPythonPackages pypkgs));
            buildInputs = commonBuildInputs ++ [ python P.triton ];
            inherit shellHook;
          };

          default = cpu;
        };
      });
}
