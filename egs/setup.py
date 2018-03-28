import tensorflow as tf
from src.load import DemoProvider, IEMOCAPData, BlizzardData
from src.input_handlers import Waves, eGeMAPSFunctionals
from src.module_handlers import FCModule
from src.output_handlers import WaveClasses, EmoCategoricalBasic4, EmoCategoricalHappySad, EmoDimensional, EmoProfileBasic4


tasks = {
    'tests': [
        {   # 1
            'name': 'tests/SimpleModel',
            'use_cross_validation': False,
            'num_epochs': 5,
            'data_provider': DemoProvider,
            'input_handler': Waves,
            'module_handlers': [FCModule(layer_dims=[20], nonlinearity=tf.nn.relu)],
            'output_handler': WaveClasses
        },
        {   # 2
            'name': 'tests/GraphModel',
            'use_cross_validation': False,
            'num_epochs': 5,
            'data_provider': DemoProvider,
            'input_handlers': {'input-1': Waves},
            'module_handlers': {'module-1': FCModule(layer_dims=[20], nonlinearity=tf.nn.relu)},
            'output_handlers': {'output-1': WaveClasses},
            'graph': [('input-1', ['module-1']), ('module-1', ['output-1'])]
        },
        {   # 3
            'name': 'tests/emotion_recognition',
            'use_cross_validation': False,
            'num_epochs': 5,
            'data_provider': IEMOCAPData,
            'input_handler': eGeMAPSFunctionals,
            'module_handlers': [FCModule(layer_dims=[20], nonlinearity=tf.nn.sigmoid)],
            'output_handler': EmoCategoricalHappySad,
        },
        {   # 4
            'name': 'tests/cross_validation',
            'use_cross_validation': True,
            'num_epochs': 5,
            'data_provider': IEMOCAPData,
            'input_handler': eGeMAPSFunctionals,
            'module_handlers': [FCModule(layer_dims=[20], nonlinearity=tf.nn.sigmoid)],
            'output_handler': EmoCategoricalHappySad,
        },
    ],

    'gen_features': [
        {   # 1
            'name': 'generation/categorical_width-200',
            'data_provider': IEMOCAPData,
            'input_handlers': {'eGeMAPS-88': eGeMAPSFunctionals},
            'module_handlers':  {
                'NN': FCModule(layer_dims=[200], nonlinearity=tf.nn.sigmoid),
                'NN-cat': FCModule(layer_dims=[20], nonlinearity=tf.nn.sigmoid),
                'NN-dim': FCModule(layer_dims=[20], nonlinearity=tf.nn.sigmoid),
            },
            'output_handlers': {
                'Categorical-4': EmoCategoricalBasic4,
                'Dimensional-3': EmoDimensional
            },
            'graph': [
                ('eGeMAPS-88', ['NN']),
                ('NN', ['NN-cat', 'NN-dim']),
                ('NN-cat', ['Categorical-4']),
                ('NN-dim', ['Dimensional-3'])
            ],
            'features': ['cat', 'dim']
        },
    ],
}



