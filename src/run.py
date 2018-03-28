from models import SimpleModel, GraphModel
from helper import load_from_file, save_to_file, average_logs, print_log, get_task
import sys
import os
from __init__ import _PROJECT_NAME

_num_epochs = 40
_interactive = __name__ == '__main__'

os.environ['project_DIR'] = os.path.join(os.path.abspath(__file__).split('/'+_PROJECT_NAME)[0], _PROJECT_NAME)
os.environ['SOURCE_DIR'] = os.path.join(os.environ['HOME'], 'data')
os.environ['DATA_DIR'] = os.path.join(os.environ['project_DIR'], 'data')

"""
USAGE
Through Son of Grid Engine:

- Or, single job:
qsub -N <job_name> ~/submit.sh ../src/run.py <task_set> <task_id>

- Or, array job:
qsub \
  -N <job_name> \
  -t <begin_id>-<end_id>:1 \
  -tc <concurrent_jobs> \
  ~/submit.sh ../src/run.py <task_set>

- Through python command line:
python ../src/run.py <task_set> <task_id>
"""


# Creates the model and either loads the existing model and log data, or trains the model and returns the log data
def run_task(task, fold=None, verbose=True):
    # set output path and ensure a directory exists for this path
    output_path = os.path.join(os.environ['project_DIR'], 'results', task['name'])
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # only print out log data during training when this is False
    cross_validation = fold is not None

    # initialise the model
    if 'graph' in task:
        # use the given graph structure
        model = GraphModel(task['name'], task['data_provider'],
                           task['input_handlers'], task['module_handlers'], task['output_handlers'], task['graph'],
                           use_cross_validation=cross_validation, cross_validation_fold=fold,
                           add_summaries=True, verbose=not cross_validation)
    else:
        # no graph structure given, create simple chain graph
        model = SimpleModel(task['name'], task['data_provider'],
                            task['input_handler'], task['module_handlers'], task['output_handler'],
                            use_cross_validation=cross_validation, cross_validation_fold=fold,
                            add_summaries=True, verbose=not cross_validation)

    # report the model built
    if verbose:
        print(model)

    # reload or train the model
    saved_model = os.path.join(os.environ['project_DIR'], 'results', model.experiment_name, 'model', 'trained_model.ckpt.index')
    if os.path.isfile(saved_model):
        # load the model and the log file
        model.restore_model()
        log_data = load_from_file(os.path.join(os.environ['project_DIR'], 'results', model.experiment_name, 'results.log'), '.pkl')
    else:
        # train the model
        log_data = model.train(num_epochs=task.get('num_epochs', _num_epochs))

    return model, log_data


# runs the task 5 times for each cross validation fold, and combines the results
def cross_validate_task(task):
    if os.path.isfile(os.path.join('results', task['name'], 'results.log.pkl')):
        raise ValueError('Experiment {} already has a results.log file, please ensure this cross validation '
                         'experiment is not overwriting a previous experiment'.format(task['name']))

    log_data_list = []
    task_name = task['name']
    for fold in range(1, 5+1):
        # modify the experiment_name
        task['name'] = os.path.join(task_name, 'fold_{}'.format(fold))

        # train/load the model and log data
        print('training/loading model for fold {}\n'.format(fold))
        model, log_data = run_task(task, fold=fold, verbose=fold == 1)
        log_data_list.append(log_data)

    # average the model performance over the 5 cross validation folds
    avg_log_data = average_logs(log_data_list)
    save_to_file(avg_log_data, os.path.join(os.environ['project_DIR'], 'results', task_name, 'results.log'), '.pkl')

    print_log(task_name)
    return model, avg_log_data


if _interactive:
    task = get_task()

    if task.get('use_cross_validation', True) is True:
        cross_validate_task(task)
    else:
        run_task(task)



