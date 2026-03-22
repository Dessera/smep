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
    torch = prev.torch.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
    });
    torchvision = prev.torchvision.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
      postFixup = ''
        addAutoPatchelfSearchPath "${final.torch}"
      '';
    });
    nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
    });
    nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
    });
    nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
    });
    nvidia-nvshmem-cu12 = prev.nvidia-nvshmem-cu12.overrideAttrs (old: {
      buildInputs = (old.buildInputs or [ ]) ++ cudaLibs;
    });
  };

  overlay = workspace.mkPyprojectOverlay {
    sourcePreference = "wheel";
  };

  cudaLibs = [
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.nccl
    pkgs.cudaPackages.libcublas
    pkgs.cudaPackages.libcusparse
    pkgs.cudaPackages.libcusparse_lt
    pkgs.cudaPackages.libcusolver
    pkgs.cudaPackages.libcurand
    pkgs.cudaPackages.libcufile
    pkgs.cudaPackages.libnvshmem
    pkgs.cudaPackages.cuda_gdb
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cuda_cudart
    pkgs.cudaPackages.cudatoolkit
    pkgs.linuxPackages.nvidia_x11
    pkgs.rdma-core
    pkgs.openmpi
  ];

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
