{
  description = "Sepsis mortality explainable prediction project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      flake-parts,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem =
        { pkgs, ... }:
        let
          python = pkgs.python314;

          py = import ./nix/python.nix {
            inherit
              pkgs
              python
              uv2nix
              pyproject-nix
              pyproject-build-systems
              ;
          };

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT=$(git rev-parse --show-toplevel)
          '';
        in
        {
          packages = {
            inherit (py.packages) default;
          };

          devShells = {
            default = pkgs.mkShell {
              inherit shellHook;
              env = env // {
                UV_PYTHON = "${py.env.editable}/bin/python";
              };
              packages = [
                py.env.editable
              ]
              ++ (with pkgs; [
                uv
                ruff
                nixd
                nixfmt-rfc-style
              ]);
            };

            uv = pkgs.mkShell {
              inherit shellHook;
              env = env // {
                UV_PYTHON = "${python}/bin/python";
              };
              packages = [
                python
                pkgs.uv
              ];
            };
          };
        };
    };
}
