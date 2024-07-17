Data Quality dashboard
=========================================================
The repository comprises the source code of the data quality dashboard. The dashboard enables data engineers and business users to automatically monitor data quality and use this insight to make data-driven decisions. The data quality dashboard has a modular design, i.e., new tools can be easily integrated using REST APIs. Specifically, the dashboard has the following requirements:

* Data Ingestion: dealing with CSV files and SQL databases, e.g., PostgreSQL, MySQL, and SQL Server.

* Data profiling: The dashboard provides an overview of the data, such as the number of records, data types, data patterns, and automatic generation of functional dependencies.

* Data Cleaning:
    * Auttomatically detecting errors
    * For ML-based detectors, the dashboard asks users to provide a set of labels
    * Automatically generating repair candidates

* Data Visualization: It provides graphical representations of data quality metrics, such as charts, graphs, and tables.

* DataSheets, similar to model cards, which show the lineage of the data, from its source to its destination, including any transformations or modifications that occurred during the process.

* Human-in-the-loop
    * Users may be asked to provide labels or judge the repair accuracy
    * domain knowledge may be provided via specific dirty values, rules, patterns, or constraints (used for data validation)

* iterative cleaning:
    * the dashboard performs consecutive cleaning operations with the objective of enhancing the performance of an ML model. If a user provides the type of the ML task (e.g., binary classification or regression) and the name of the attributes serving as labels, this module embarks on a cycle of cleaning processes, employing a variety of cleaning tools to automatically identify and apply the detection and repair tools that broadly improve the target model's predictive performance.


## Setup

Clone with submodules
```shell script
git clone https://git.sagresearch.de/kompaki/data-quality-dashboard.git --recurse-submodules
```

### Prerequisites

* Python3 
* Java (for the automatic generation of functional dependencies)

### Install requirements

```
pipx install poetry
poetry install
```
### Data Folder

For the use of default datasets that also have a ground truth it is required to create a new folder:

```
mkdir data
mkdir data/raw
```

Inside the folder called raw each there should be a folder for each dataset to be used, with the folder named after the dataset. Each dataset folder contains a `clean.csv` and a `dirty.csv`.

### Install error detection and repair methods

##### RAHA and BARAN

To install these methods, you can do so in two different ways:

Option 1: through pip3
```
pip3 install raha
```
Option 2: through the setup.py script which exists in the raha main directory
```
python3 setup.py install
```

##### Katara

For this method, we do not need to install packages, but we need to download the knowledge base:
Download the knowledge base ([link](https://bit.ly/3hPkpWX)) and unzip the file. The files of the knowledge base should 
be placed in the following path.
```
cd detectors/katara/knowedge-base
```


## Usage

### Running the dashboard and API

In order to run both the dashboard and API in one instance the startup script run_all.py can be used.

```
poetry run python3 run_dashboard.py
```

Alternativley both API and dashboard can be run separately with the following commands in order to get separated info about the processes in the terminals.

Dashboard:
```
poetry run python3 -m main.app
```

API:
```
poetry run uvicorn API.fastAPImain:app --reload
```
