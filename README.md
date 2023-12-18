# GEMBED: GEnerative Modelling and emBEDing spaces

## Installation
To install the package run the following instruction on the machine you will use for training/inference!

First clone/download the repository on to your machine and `cd gembed` to the root project directory.

We recommend installing the package in a new conda/virtual environment: 
```bash
conda create -n gembed python=3.10 pip=23.3 
```

Activate the conda environment:
```bash
conda activate gembed
```

Next, install package dependencies
```bash
pip install -r requirements.txt
```

Finally, install package itself:
```bash
pip install -e .
```

You're done! 

## Uninstall
The easiest way to remove the package, is by removing the virtual environment: 
```bash
conda remove -n gembed --all
```