# Scalable Option Learning in High-Throughput Environments

## Installation

Please follow the original Sample Factory 2 installation instructions. 

## Usage

To run any experiment, use the `sweep.py` script in the `slurm` folder. This takes as argument a config `.yaml` file specifying all hyperparameters to sweep over. You can add values in a list and it will do a grid search over everything.

The script can be run in dry mode to run locally, and also to submit jobs to the cluster. The dry mode runs with the last hyperparameter config in the grid search. Additionally, there is the `--debug` flag which runs single-threaded locally, so you can use debuggers like `pdb`.

To run single-threaded locally for debugging:

```
python sweep.py --expfile exp_configs/{sweep_file}.yaml --dry --debug
```

To run locally in multi-thread mode (for example, for speed tests---you might need a 32G machine for this):

```
python sweep.py --expfile exp_configs/{sweep_file}.yaml --dry 
```

To submit the full sweep to the cluster:

```
python sweep.py --expfile exp_configs/{sweep_file}.yaml --wandb_proj {wandb_project_name} --partition {partition_name} --num_cpus {num_cpus} --seeds 5 --days 3
```

## Dependencies

Make sure to use python 3.12, not 3.13 (that is not compatible with `gymnasium_robotics`). Otherwise the dependencies are pretty much those of Sample Factory. 

## Notes

- The repo includes a patched version of the NLE (`nle_patched`) where the changes in luck due to lunar phases are disabled. Very annoying for reproducibility otherwise. 


## Acknowledgements

Our algorithm is built upon the excellent Sample Factory codebase, check it out! 
