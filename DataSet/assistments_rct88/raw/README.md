# Expansion on Exploring Common Trends in Online Educational Experiments

[![OSF DOI](https://img.shields.io/badge/OSF-10.17605%2Fosf.io%2Fm2jqe-blue)][doi]
[![Docker](https://img.shields.io/docker/automated/ahaim5357/10.17605-osf.io-m2jqe)][container]

[*Expansion on Exploring Common Trends in Online Educational Experiments*][doi] contains the *50 Experiments+2022* dataset, providing additional information to the *50 Experiments-2021* dataset released with [*Exploring Common Trends in Online Educational Experiments*][odoi]. The expansion contains the raw problem logs associated with the provided experiments and raw student actions prior to the experiment period.

This project serves as a environment setup for integrating with both datasets by downloading them from their OSF projects ([original][oosf] and [expansion][doi]) and merging them into the same subfolder structure. This also downloads the accompanying documentation and licenses.

This was presented at the [*2022 Conference on Digital Experimentation @ MIT (CODE@MIT)*][code].

## License

The content of this Open Science Foundation project is licensed under a [Creative Commons Attribute 4.0 International License][cl]. As the dataset is pulled from ASSISTments, this project is compliant under the [ASSISTments Public Data License v0.1][dl].

The software of this Open Science Foundation project is licensed under the [MIT License][sl]. This is to prevent distribution issues when using the source in other projects. You can read more about this [here][ccsoftware].

## Environment Setup Script

### Method 1: Docker

The [Docker Container][container] can be run using:

```bash
docker run -v ${PWD}:/app/env ahaim5357/10.17605-osf.io-m2jqe:mitcode2022
```

You can also clone this repository, then run the following [Docker][docker] commands:

```bash
docker build -t <image_name> .
docker run -v ${PWD}:/app/env <image_name>
```

Where `image_name` can be specified to whatever identifier the user desires.

This will automatically download the datasets, dataset documentation, and content/dataset license.

#### Environment Variables

There are three environment variables:

* `M2JQE_RAW_ZIP` (default false): When true, will download the raw project zip onto the local machine if it's not already present. The project will be pulled regardless for setup.
* `M2JQE_DOCS` (default true): When true, will download the dataset documentation and the content/dataset license.
* `M2JQE_MULTIPROCESS` (default false): When true, will execute the setup of the original and expansion dataset in two separate subprocesses.

These can be added to the docker script via `-e <ENV_VAR>=<VALUE>`:

```bash
# Download the raw zip, no docs, and setup using two separate subprocesses on the local machine
docker run -v ${PWD}:/app/data -e M2JQE_RAW_ZIP=true -e M2JQE_DOCS=false -e M2JQE_MULTIPROCESS=true <image_name>
```

### Method 2: Python Environment

The compiler script is written in Python 3.8.10. You can install the required libraries using the `requirements.txt` provided:

```bash
pip install -r requirements.txt
```

> You may need to prefix the `pip` command with either `python -m` for Unix systems or `py -m` for Windows systems if `pip` was not properly installed onto the path.

Then navigate to the folder in your terminal and run `env_setup.py`.

For Unix Systems (Linux, MacOS):

```bash
python3 ./env_setup.py
```

For Windows Systems:

```pwsh
py ./env_setup.py
```

#### Command Line Arguments

There are four command line arguments that can be specified after the script:

* `-h`/`--help`: Prints the command and what arguments it can take.
* `-r`/`--raw-zip`: Downloads the raw project zip onto the local machine if it's not already present.
* `-n`/`--no-docs`: Will not download the supplemental documents with the environment setup.
* `-m`/`--multiprocess`: Will execute the setup of the original and expansion dataset in two separate subprocesses.

For Unix Systems (Linux, MacOS):

```bash
# Download the raw zip, no docs, and setup using two separate subprocesses on the local machine
python3 ./env_setup.py -r -n -m
```

For Windows Systems:

```pwsh
# Download the raw zip, no docs, and setup using two separate subprocesses on the local machine
py ./env_setup.py -r -n -m
```

## Citation

### *50 Experiments-2021*

```
@inproceedings{50experimentsoriginal,
  author       = {Ethan Prihar and
                  Manaal Syed and
                  Korinn Ostrow and
                  Stacy Shaw and
                  Adam Sales and
                  Neil Heffernan},
  title        = {{Exploring Common Trends in Online Educational 
                   Experiments}},
  booktitle    = {{Proceedings of the 15th International Conference 
                   on Educational Data Mining}},
  year         = 2022,
  pages        = {27--38},
  publisher    = {International Educational Data Mining Society},
  month        = jul,
  venue        = {Durham, United Kingdom},
  doi          = {10.5281/zenodo.6853041},
  url          = {https://doi.org/10.5281/zenodo.6853041}
}
```

### *50 Experiments+2022*

```
Haim, A., Prihar, E., Shaw, S. T., Sales, A., & Heffernan, N. T., III. (2022, October 20). Expansion on Exploring Common Trends in Online Educational Experiments. Presented at the 2022 Conference on Digital Experimentation @ MIT (CODE@MIT '22), October 20-21, 2022, Boston, MA, US. Accessible at https://doi.org/10.17605/osf.io/m2jqe
```


[docker]: https://www.docker.com/
[container]: https://hub.docker.com/repository/docker/ahaim5357/10.17605-osf.io-m2jqe
[doi]: https://doi.org/10.17605/osf.io/m2jqe
[odoi]: https://doi.org/10.5281/zenodo.6853041
[oosf]: https://doi.org/10.17605/osf.io/59shv
[code]: https://ide.mit.edu/events/2022-conference-on-digital-experimentation-mit-codemit/

[cl]: https://osf.io/qykhb
[dl]: https://osf.io/s7ufj
[sl]: ./LICENSE
[ccsoftware]: https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software
