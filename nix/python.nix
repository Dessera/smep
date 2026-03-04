{
  pkgs,
  python,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}:
let
  util = pkgs.callPackage pyproject-nix.build.util { };

  workspace = uv2nix.lib.workspace.loadWorkspace {
    workspaceRoot = ../.;
  };

  userOverlay = final: prev: {
    numba = prev.numba.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.tbb_2022 ];
    });
  };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {
      inherit python;
    }).overrideScope
      (
        pkgs.lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          userOverlay
        ]
      );

  editableOverlay = workspace.mkEditablePyprojectOverlay {
    root = "$REPO_ROOT";
  };

  editablePythonSet = pythonSet.overrideScope (
    pkgs.lib.composeManyExtensions [
      editableOverlay
      userOverlay
      (final: prev: {
        smep = prev.smep.overrideAttrs (old: {
          src = pkgs.lib.fileset.toSource {
            root = old.src;
            fileset = pkgs.lib.fileset.unions [
              (old.src + "/pyproject.toml")
              (old.src + "/README.md")
              (old.src + "/src/smep")
            ];
          };
          nativeBuildInputs =
            old.nativeBuildInputs
            ++ (final.resolveBuildSystem {
              editables = [ ];
            });
        });
      })
    ]
  );
in
rec {
  packages = {
    default = util.mkApplication {
      venv = env.default;
      package = pythonSet.smep;
    };
  };

  env = {
    default = pythonSet.mkVirtualEnv "smep-env" workspace.deps.all;
    editable = editablePythonSet.mkVirtualEnv "smep-dev-env" workspace.deps.all;
  };
}
