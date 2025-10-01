## Latent neural ensembles detection

Detect neural ensembles, groups of neurons that are active together, from spike data using fastICA. 

### Overview
```
latent_ensembles_detection/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── detect.py # run fastICA, find principal neurons in each ensemble
│   ├── utils.py # utilities for loading and saving data, and pre-processing spikes
│   └── plots.py # plotting functions
├── notebooks/ # clean notebook demonstrating usage
└──  examples/ # example runner script
```

### Quick start
Recommended: create a virtual environment first
```
# from repo root
python3 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .\.venv\Scripts\activate        # Windows PowerShell

# install package in editable mode 
pip install -e .

# install requirements
pip install -r requirements.txt
```
### Contact
Maintainer: Sara Molas Medina. Open issues or PRs for questions, bugs, or feature requests.

### Reference:
Replication of the method by: 
Lopes-dos-Santos V et al (2014). Detecting cell assemblies in large neuronal populations. J Neurosci Methods. 220(2):149-66. doi: 10.1016/j.jneumeth.2013.04.010.
