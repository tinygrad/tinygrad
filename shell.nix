{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "tinygrad-nix";

    targetPkgs = _: [
      pkgs.clang
      pkgs.micromamba
    ];

    profile = ''
      set -e
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      eval "$(micromamba shell hook --shell=bash | sed 's/complete / # complete/g')"
      micromamba create --yes -q -n tinygrad
      micromamba activate tinygrad
      micromamba install --yes -f conda-requirements.txt -c conda-forge
      set +e
    '';


  };
in fhs.env
