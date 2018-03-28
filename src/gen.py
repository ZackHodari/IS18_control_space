from run import run_task
from load import IEMOCAPData, BlizzardData, BlizzardTestData
from helper import save_to_file, get_task
import os
from __init__ import _PROJECT_NAME

_num_epochs = 40
_interactive = __name__ == '__main__'

os.environ['project_DIR'] = os.path.join(os.path.abspath(__file__).split('/'+_PROJECT_NAME)[0], _PROJECT_NAME)
os.environ['SOURCE_DIR'] = os.path.join(os.environ['project_DIR'], 'data')
os.environ['DATA_DIR'] = os.path.join(os.environ['project_DIR'], 'data')

"""
USAGE
Through Son of Grid Engine:

- Or, single job:
qsub -N <job_name> ~/submit.sh ../src/gen.py <task_set> <task_id>

- Or, array job:
qsub \
  -N <job_name> \
  -t <begin_id>-<end_id>:1 \
  -tc <concurrent_jobs> \
  ~/submit.sh ../src/gen.py <task_set>

- Through python command line:
python ../src/gen.py <task_set> <task_id>
"""


def gen_features(model, data_provider, feature_names='cat'):
    if isinstance(feature_names, str):
        feature_names = [feature_names]

    feature_dir = os.path.join(os.environ['project_DIR'], 'results', model.experiment_name, 'features')
    if not os.path.isdir(feature_dir):
        os.mkdir(feature_dir)

    supported = ['cat', 'dim', 'eGeMAPS']
    feature_names = filter(lambda f: f in supported, feature_names)
    if len(feature_names) == 0:
        raise ValueError(
            'feature type(s) not recognised or not supported\ngot {}'.format(' '.join(feature_names)))

    features = []
    for feature_name in feature_names:
        if feature_name == 'cat':
            features.append(model.output_handlers['Categorical-4'].predictions)
        if feature_name == 'dim':
            features.append(model.output_handlers['Dimensional-3'].predictions)
        if feature_name == 'eGeMAPS':
            features.append(model.input_handlers['eGeMAPS-88'].inputs)

    for feature_name in feature_names:
        if not os.path.isdir(os.path.join(feature_dir, feature_name)):
            os.mkdir(os.path.join(feature_dir, feature_name))

    # for all batches in the data_provider, perform inference for the given features and save to files
    print('Generating for features (using {}): {}'.format(data_provider.__class__.__name__, ' '.join(feature_names)))
    for batch in data_provider.all_data():
        # This would need to be changed to use whatever arbitrary input type the model specifies
        feed_dict = {
            model.input_handlers['eGeMAPS-88'].inputs: batch.input_func,
            model.input_handlers['eGeMAPS-88'].is_training: False
        }

        batch_features = model.sess.run(features, feed_dict=feed_dict)

        for feature_name, batch_feature in zip(feature_names, batch_features):
            for utt_name, feature_val in zip(batch.utt_name, batch_feature):
                save_to_file(feature_val.reshape(-1), os.path.join(feature_dir, feature_name, utt_name), '.'+feature_name)


if _interactive:
    task = get_task()

    model, _ = run_task(task)

    gen_features(model, BlizzardData({'load_lld_features': False, 'load_emo_labels': False}), task['features'])
    gen_features(model, BlizzardTestData({'load_lld_features': False, 'load_emo_labels': False}), task['features'])



