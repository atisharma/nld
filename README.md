# nld

This is a library of some nice nonlinear systems on the GPU.
It uses [JAX](https://jax.readthedocs.io/en/latest/) to put the computations on the GPU. It's written in [Hy](https://hylang.org/).


## Installation

```bash
pip install .
```

## Usage

The library provides tools for analyzing nonlinear dynamical systems using GPU acceleration.


## Development

Contributions are welcome. Please feel free to submit a pull request.

## FAQ

**Q**: Why are you using this stupid language (Hy) and not python?
**A**: Because I like it. Also, it's easier to write in a functional style in a lisp, and JAX needs pure functions. If it bothers you, use `hy2py` to produce the equivalent python code.


## License

GPL v3.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/atisharma/nld)
