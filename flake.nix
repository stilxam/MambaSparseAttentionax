{
  description = "EQUINOX KVAX Flake";

  inputs = {
    unstable-nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, unstable-nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import unstable-nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        python = pkgs.python313;

        jax-triton = pkgs.python313Packages.buildPythonPackage {
          pname = "jax-triton";
          version = "0.3.0";
          pyproject = true;

          src = pkgs.fetchFromGitHub {
            owner = "jax-ml";
            repo = "jax-triton";
            rev = "a298a7884054d3fc4bf94e1cb3d2a3baa907ea6b";
            hash = "sha256-z853dxGWg3vLklYqSvapRLnhwwkPBRqtSAzGmE9rLns=";
          };

          nativeBuildInputs = with pkgs.python313Packages; [
            setuptools
            setuptools-scm
            wheel
          ];

          propagatedBuildInputs = with pkgs.python313Packages; [
            triton
            absl-py
            jax
            jaxlib
          ];

          doCheck = false;
        };


        mainPythonPackages = ps: with ps; [
          cython
          pytest

          jax-triton


          jax
          jaxlib
          jaxtyping
          equinox
          wadler-lindig

          optax
          orbax-checkpoint

          numpy
          numba

          polars
          pyarrow

          matplotlib
          plotly
          seaborn

          notebook
          click
	  tqdm
	  streamlit

        ];

        pythonEnv = python.withPackages mainPythonPackages;

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.zed-editor
            pkgs.cudaPackages.cuda_nvrtc
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.cudatoolkit
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.cudaPackages.cuda_nvrtc
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cudatoolkit
              pkgs.stdenv.cc.cc.lib
            ]}:$LD_LIBRARY_PATH

            # For JAX to find PTXAS if needed (sometimes required for XLA)
            export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH
	    export TF_CPP_MIN_LOG_LEVEL=2

            echo "Environment loaded. JAX/CUDA paths configured."
            echo "LD_LIBRARY_PATH set to include cuda_nvrtc and cudnn."
          '';
        };
      });
}
