# This shell.nix combines with a .envrc to create or use a venv in the
# directory it's in.
#
# To use rename as shell.nix.

{ pkgs ? import <nixpkgs> {} }:

let
  # set python version
  pyver = "3.9";
  py = pkgs.python39;
  pypkgs = pkgs.python39Packages;
in
pkgs.stdenv.mkDerivation rec {
  name = "python-virtualenv-shell";
  env = pkgs.buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    # It is essential **not** to add `pip` here as
    # it would prevent proper virtualenv creation
    # by forcing the wrong pip installation path.

    pkgs.readline

    py
    pypkgs.virtualenv
    pypkgs.setuptools

    pypkgs.jax
    pypkgs.jaxlib
    pypkgs.matplotlib

    pypkgs.zstandard
    pypkgs.pyqt5
    pypkgs.hy
    pypkgs.pynvim
  ];

  shellHook = ''
    # set SOURCE_DATE_EPOCH so that we can use python wheels
    SOURCE_DATE_EPOCH=$(date +%s)

    # set up the venv
    export venv=".venv/''$MACHTYPE/${pyver}"
    test -f ''${venv}/bin/activate || ( echo "Installing new venv ''${venv}"; virtualenv ''${venv}; )
    source ''${venv}/bin/activate

    export LD_LIBRARY_PATH="''${venv}/lib"
    export PYTHONHASHSEED=$(date +%s)
  '';
}
