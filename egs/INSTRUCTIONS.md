From the egs directory, run the ../src/run.py script using some number of inputs from setup.py

	python ../src/run.py tests 1

This allows for computation to be ran on compute clusters, e.g. grid engine

	qsub -N test_name ~/submit.sh ../src/run.py tests 1

And array jobs can be run, where the task_id is taken from the SGE_TASK_ID environment variable (can be changed in ../src/run.py)

	qsub -N test_name -t 1-6:1 ~/submit.sh ../src/run.py tests

