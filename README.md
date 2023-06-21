# MyFM and OGB Framework

This repository contains a framework that facilitates the loading of graphs from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/) and training them using the [myFM](https://github.com/tohtsky/myFM) library.

## Installation

There are two ways to install the required dependencies:

- Recreate the environment using the provided `.yml` file and [Anaconda](https://www.anaconda.com/download). This is the recommended way to ensure you have all the necessary packages.
- Manually install the [OGB](https://ogb.stanford.edu/docs/home/) and [myFM](https://github.com/tohtsky/myFM) libraries. Guides on how to do this can be found on the respective sites.

## Usage

Follow these steps to utilize this framework:

1. Modify the `config.ini` file:
   - Choose the desired graph.
   - Activate the desired data parts.
   - Choose hyperparameters.

2. Run `main.py`.

3. Folders and data will be created automatically

All run data will be automatically saved in an Excel file for convenient examination and comparison.

## Implemented Data Parts

The framework currently supports the following data parts:

- Node Embeddings
- NodeIDs
- Vectors:
  - Neighborhood (Binary) (Normalized)
  - Common Neighborhood (Binary) (Normalized)
- Values:
  - Total Common Neighbors
  - Adamic Adar
  - Resource Allocation
  - Jaccard

Enjoy training your models!