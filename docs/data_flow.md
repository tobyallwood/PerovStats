# Data Flow

## Input
- PerovStats takes .spm files as an input, as well as a config file (`default_config.yaml` by default) containing values for variables including the expected location of the `.spm` files to process.

## Processing steps
The program will:
1. Loads the `.spm` file and convert to an image mask
2. Optionally performs a fourier transform to isolate the topograhy of the perovskite material
3. Analyses this new image and generate a mask outlining the edges of grains
4. Counts and generates data about both individual grains and averages of all grains in the scan
5. Exports this data to a `.csv` file

## Output
All output data is saved to `/output/` (editable in the config) under a sub-folder with the same name as the original `.spm` file.

This folder contains:
- A copy of the config settings used to generate the data
- The post-fourier transform scan both in `.jpg` and `.npy` form
- The boolean mask in both `.jpg` and `.npy` form
- The original image used
- A `.csv` file containing the statistics of the image and individual grains
