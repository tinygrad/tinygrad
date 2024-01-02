{
  description = "You like pytorch? You like micrograd? You love tinygrad! ❤️ ";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem = { config, self', inputs', pkgs, system, ... }: {
        packages.default = pkgs.python3Packages.callPackage ./default.nix {
          version = inputs.self.lastModifiedDate;
        };
        devShells.default = pkgs.mkShell {
          buildInputs = [
            self'.packages.default
            pkgs.ruff
          ];
        };
      };
    };
}
