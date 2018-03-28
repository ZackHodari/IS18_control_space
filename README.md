# IS18_control_space
Code relating to paper submitted for review at INTERSPEECH 2018

Makes use of [modNN](https://github.com/ZackHodari/modNN), a TensorFlow interface that allows for input, output, and model handlers to be combined together as modules.

- - - -

Models are trained using [run.py](src/run.py) using the *run_task* function which takes a config (python dictionary) as input.

Example config included at [setup.py](egs/setup.py), two formats possible;
* Sequential computational graph (SimpleModel): one input, one output, sequential NN modules
* Customisable computational graph (GraphModel): requires handlers to be given names and an adjacency list to be defined in the config.

