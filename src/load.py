import numpy as np
import os
import re
from helper import load_from_file, save_to_file
from matplotlib.mlab import specgram
from scipy.io.wavfile import read as read_wavfile

DEFAULT_SEED = 1234567890
CATEGORIES = np.array(['Anger', 'Disgust', 'Excited', 'Fear', 'Frustration',
                       'Happiness', 'Neutral', 'Other', 'Sadness', 'Surprise'])
categories = np.array(['ang', 'dis', 'exc', 'fea', 'fru', 'hap', 'neu', 'sad', 'sur', 'xxx', 'oth'])

missing_data = ['Ses03M_impro03_M001', 'Ses03F_impro07_M030']
miss_dim_labels = ['Ses03M_impro03_M001', 'Ses03F_impro07_M030', 'Ses01M_impro04_M025', 'Ses01M_script02_2_M051',
                   'Ses01M_script02_2_M052', 'Ses01M_script02_2_M053', 'Ses01M_script02_2_M054', 'Ses01M_script03_1_F026',
                   'Ses02M_script02_2_F046', 'Ses03F_impro07_M035', 'Ses03M_impro02_M000', 'Ses03M_impro04_F031',
                   'Ses03M_impro07_F023', 'Ses03M_impro07_M000', 'Ses03M_impro07_M001', 'Ses03M_impro07_M002',
                   'Ses03M_impro07_M003', 'Ses03M_impro07_M004', 'Ses03M_impro07_M005', 'Ses03M_impro07_M006',
                   'Ses03M_impro07_M007', 'Ses03M_impro07_M008', 'Ses03M_impro07_M009', 'Ses03M_impro07_M010',
                   'Ses03M_impro07_M011', 'Ses03M_impro07_M012', 'Ses03M_impro07_M013', 'Ses03M_impro07_M014',
                   'Ses03M_impro07_M015', 'Ses03M_impro07_M016', 'Ses03M_impro07_M017', 'Ses03M_impro07_M018',
                   'Ses03M_impro07_M019', 'Ses03M_impro07_M020', 'Ses03M_impro07_M021', 'Ses03M_impro07_M022',
                   'Ses03M_impro07_M023', 'Ses03M_impro07_M024', 'Ses03M_impro07_M025', 'Ses03M_impro07_M026',
                   'Ses04M_script03_2_M009']
missing_data.extend(miss_dim_labels)

IEMOCAP_source = os.path.join(os.environ['SOURCE_DIR'], 'IEMOCAP')
IEMOCAP_path = os.path.join(os.environ['DATA_DIR'], 'IEMOCAP')

Blizzard_source = os.path.join(os.environ['SOURCE_DIR'], 'Blizzard2017')
Blizzard_path = os.path.join(os.environ['DATA_DIR'], 'Blizzard2017')

BlizzardTest_source = os.path.join(os.environ['SOURCE_DIR'], 'Blizzard2017Test')
BlizzardTest_path = os.path.join(os.environ['DATA_DIR'], 'Blizzard2017Test')

NUM_FUNCTIONALS = 88
NUM_LLDS = 23


class _AbstractProvider(object):
    """
    Abstract data provider, uses a config that is passed onto the AbstractSample class which loads the datum

    :param file_paths: list - file paths which samples will use to load the data
    :param data_config: dictionary - configuration specifying what to load for each sample
    :param batch_size: integer - how big each mini-batch should be
    :param shuffle_data: boolean - shuffle the training data
    :param rng_seed: integer (optional) - custom seed for rng used for shuffling
    """

    def __init__(self, file_paths, data_config, batch_size, shuffle_data, rng_seed=DEFAULT_SEED, data_splitter=None):
        assert isinstance(file_paths, list) or isinstance(file_paths, np.ndarray), (
            'file_paths must be a list'
            '\nGot {}'.format(type(file_paths)))
        assert len(file_paths) > 0, (
            'file_paths must be non-empty'
            '\nGot {}'.format(len(file_paths)))
        assert isinstance(data_config, dict), (
            'data_config must be a dictionary'
            '\nGot {}'.format(type(data_config)))
        assert isinstance(rng_seed, int), (
            'rng_seed must be an integer'
            '\nGot {} ({})'.format(rng_seed, type(rng_seed)))
        assert isinstance(batch_size, int), (
            'batch_size must be an integer'
            '\nGot {}'.format(type(batch_size)))
        assert batch_size > 0, (
            'batch_size must be greater than zero'
            '\nGot {}'.format(batch_size))
        assert isinstance(shuffle_data, bool), (
            'shuffle_data must be a boolean'
            '\nGot a {}'.format(type(shuffle_data)))

        self.file_paths_all = np.array(file_paths)
        self.rng_seed       = rng_seed
        self.rng            = np.random.RandomState(rng_seed)

        if data_splitter is None:
            self.train_valid_test_split()
        else:
            # use a custom splitting function instead
            data_splitter(self)

        self._current_order = np.arange(self.file_paths_train.shape[0])
        self.data_config    = data_config
        self.batch_size     = batch_size
        self.shuffle_data   = shuffle_data

    # Splits the file paths into 3 lists for training, validation and testing
    def train_valid_test_split(self, valid_size=0.15, test_size=0.15):
        num_samples = self.file_paths_all.shape[0]
        indices = self.rng.permutation(num_samples)

        test_index = int(num_samples*test_size)
        valid_index = int(num_samples*valid_size) + test_index

        # data = [test_data, valid_data, train_data]
        self.file_paths_test  = self.file_paths_all[indices[0:           test_index]]
        self.file_paths_valid = self.file_paths_all[indices[test_index:  valid_index]]
        self.file_paths_train = self.file_paths_all[indices[valid_index: None]]

    # Resets the data provider to the initial state, resetting the rng and fixing the order of the training data
    def reset(self):
        self.rng = np.random.RandomState(self.rng_seed)
        inv_perm = np.argsort(self._current_order)

        self._current_order = self._current_order[inv_perm]
        self.file_paths_train = self.file_paths_train[inv_perm]

    # Randomly shuffles the training data
    def shuffle(self):
        perm = self.rng.permutation(self.file_paths_train.shape[0])
        self._current_order = self._current_order[perm]
        self.file_paths_train = self.file_paths_train[perm]

    # Given a list of file paths, yield mini-batches (_AbstractBatch) of self.batch_size samples (AbstractSample)
    def yield_batches(self, file_paths):
        num_batches = int((file_paths.shape[0] - 1) // self.batch_size + 1)

        for batch_num in range(num_batches):
            batch_slice = slice(self.batch_size * batch_num,
                                self.batch_size * (batch_num + 1))

            yield self.load_data(file_paths[batch_slice])

    def load_data(self, file_paths):
        # e.g. return _AbstractBatch(file_paths, self.data_config)
        raise NotImplementedError('This should be implemented in subclasses for specific datasets')

    # A generator for batches from the test data
    @property
    def train_data(self):
        if self.shuffle_data:
            self.shuffle()

        return self.yield_batches(self.file_paths_train)

    @property
    def valid_data(self):
        return self.yield_batches(self.file_paths_valid)

    @property
    def test_data(self):
        return self.yield_batches(self.file_paths_test)

    def all_data(self):
        return self.yield_batches(self.file_paths_all)

    def __iter__(self):
        return self.train_data

    @property
    def num_batches(self):
        return self.file_paths_train.shape[0] // self.batch_size

    @property
    def num_batches_train(self):
        return self.num_batches

    @property
    def num_batches_valid(self):
        return self.file_paths_valid.shape[0] // self.batch_size

    @property
    def num_batches_test(self):
        return self.file_paths_test.shape[0] // self.batch_size


class _AbstractBatch(list):
    """
    Abstract mini-batch list, contains a list of samples and allows for access to sample attributes in array form

    :param file_paths: list - file paths which samples will use to load the data
    """
    
    def __init__(self, samples):
        super(_AbstractBatch, self).__init__(samples)

    # get attr from the list of samples
    def __getattr__(self, attr):
        attr_vals = list(map(lambda sample: sample.__getattribute__(attr), self))

        # try to convert values into a numpy array
        try:
            attr_list = np.array(attr_vals)
        except:
            attr_list = attr_vals

        self.__setattr__(attr, attr_list)  # save value for future fetches
        return attr_list


# ----------------------- #
#          Demo           #
# ----------------------- #


class DemoProvider(_AbstractProvider):
    def __init__(self, data_config, batch_size=50, shuffle_data=True):
        file_paths = range(data_config.get('num_samples', 100))
        super(DemoProvider, self).__init__(file_paths, data_config, batch_size, shuffle_data)

    def load_data(self, file_paths):
        return DemoBatch(file_paths, self.data_config)


class DemoBatch(_AbstractBatch):
    def __init__(self, file_paths, data_config):
        super(DemoBatch, self).__init__([DemoSample(file_path, data_config) for file_path in file_paths])


class DemoSample(object):
    funcs = [np.sin, np.cos, np.tan]

    def __init__(self, file_path, data_config):
        self.id = file_path % 3
        N = data_config.get('num', 100)

        self.x = np.linspace(0, 2*np.pi, N)
        self.y = self.funcs[self.id](self.x)


# ----------------------- #
#         IEMOCAP         #
# ----------------------- #


class IEMOCAPData(_AbstractProvider):
    """
    Populates list of utterance paths from the preprocessed directory
    Prepares the data for retrieval when training/testing a model (i.e. batches the data)
    """

    def __init__(self, data_config={}, batch_size=50, shuffle_data=True, rng_seed=DEFAULT_SEED):
        # Get data paths and split into 3 sets: train, valid, test
        if 'emotion_mask' in data_config and not np.all(data_config['emotion_mask']):
            # filters out utterances using the emotion_mask
            file_paths = self.get_file_paths(self.filter_file_names(data_config['emotion_mask']))
        else:
            file_paths = self.get_file_paths()

        if data_config.get('use_cross_validation', False) is True:
            data_splitter = self.cross_validation_split(data_config['cross_validation_fold'])
        else:
            data_splitter = None

        super(IEMOCAPData, self).__init__(file_paths, data_config, batch_size, shuffle_data, rng_seed, data_splitter)

    # return a list of utt_paths to visualise, if no utts specified then all utts are returned
    @staticmethod
    def get_file_paths(utt_names=[]):
        base_path = os.path.join(IEMOCAP_path, 'preprocessed')

        if not os.path.isdir(base_path):
            raise OSError('No preprocessed directory found, please run IEMOCAPProcessor first')

        if len(utt_names) == 0:
            return np.array([
                os.path.join(base_path, session, sess_name, utt_name)
                for session in filter(lambda s: s[:7] == 'Session', os.listdir(base_path))
                for sess_name in filter(lambda s: s[:3] == 'Ses', os.listdir(os.path.join(base_path, session)))
                for utt_name in filter(lambda s: s[:3] == 'Ses', os.listdir(os.path.join(base_path, session, sess_name)))
            ])
        else:
            return np.array([
                os.path.join(base_path, 'Session{}'.format(utt_name[4]), utt_name[:-5], utt_name)
                for utt_name in utt_names
            ])

    @staticmethod
    def filter_file_names(emotion_mask=np.ones(categories.shape).astype(np.bool)):
        file_names = []
        for emotion in categories[emotion_mask]:
            file_names.extend(load_from_file(
                os.path.join(os.environ['project_DIR'], 'resources', 'emotion_file_id_lists', emotion), '.scp'))
        return file_names

    # Splits the utterance names into 3 lists for training, validation and testing
    def cross_validation_split(self, fold):
        def func(_self):
            file_paths_set = set(_self.file_paths_all)
            dir = os.path.join(os.environ['project_DIR'], 'resources', 'cross_validation_splits', 'IEMOCAP')

            file_names_train_fold = load_from_file(os.path.join(dir, 'train', 'fold_{}'.format(fold)), '.scp')
            file_paths_train_fold = _self.get_file_paths(file_names_train_fold)
            _self.file_paths_train = np.array(list(file_paths_set.intersection(file_paths_train_fold)))

            file_names_valid_fold = load_from_file(os.path.join(dir, 'valid', 'fold_{}'.format(fold)), '.scp')
            file_paths_valid_fold = _self.get_file_paths(file_names_valid_fold)
            _self.file_paths_valid = np.array(list(file_paths_set.intersection(file_paths_valid_fold)))

            file_names_test_fold = load_from_file(os.path.join(dir, 'test', 'fold_{}'.format(fold)), '.scp')
            file_paths_test_fold = _self.get_file_paths(file_names_test_fold)
            _self.file_paths_test = np.array(list(file_paths_set.intersection(file_paths_test_fold)))

        return func

    def load_data(self, file_paths):
        return IEMOCAPBatch(file_paths, **self.data_config)


class IEMOCAPBatch(_AbstractBatch):

    def __init__(self, file_paths,
                 emotion_mask=np.ones(categories.shape).astype(np.bool),
                 emotion_mask_profile=np.ones(CATEGORIES.shape).astype(np.bool),
                 **kwargs):
        super(IEMOCAPBatch, self).__init__([
            IEMOCAPUtterance(file_path, emotion_mask=emotion_mask, emotion_mask_profile=emotion_mask_profile, **kwargs)
            for file_path in file_paths
        ])
        self.emotion_mask = emotion_mask
        self.emotion_mask_profile = emotion_mask_profile

    @property
    def target_one_of_k_cat_masked(self):
        # filters the categorical output vector so only emotions in the emotion_mask are included
        return self.target_one_of_k_cat[:, self.emotion_mask]

    @property
    def target_categorical_masked(self):
        return self.target_categorical[:, self.emotion_mask_profile]


class IEMOCAPUtterance(object):
    """
    Load data from various files associated with a particular utterance
    Can specify various arguments to choose how/what to load

    kwargs:
     - load_wavefile
         Boolean (default: False)
         Load the raw wavefile
     - load_func_features
         Boolean (default: True)
         Load the eGeMAPS functionals
     - load_lld_features
         Boolean (default: True)
         Load the eGeMAPS low-level descriptors
     - load_phone_alignment
         Boolean (default: False)
         Load the phone alignment
     - load_transcription
         Boolean (default: False)
         Load the word transcription
     - load_emo_labels
         Boolean (default: True)
         Load the emotion targets

     - gen_spectrogram
         Boolean (default: False)
         Generate the spectrogram of the raw wavefile
     - fft_width
         Integer (default: 256)
         Fft width for windows used to generate spectrogram
     - fft_overlap
         Integer (default: 128)
         Number of overlapping samples for windows used to generate spectrogram
     - include_frametime
         Boolean (default: False)
         Include the frametime of the eGeMAPS lld time-series features
     - flatten
         Boolean (default: False)
         Flatten the eGeMAPS lld time-series features

    Attributes:
     - sess_num
         Session number, from 1 to 5
     - sess_name
         Session name, e.g. 'Ses03F_impro02'
     - utt_name
         Utterance name, e.g. 'Ses03F_impro02_F014'

     - sample_rate
         sample rate of wavfile
     - input_wave
         wavfile
     - input_func
         eGeMAPS functionals
     - input_llds
         eGeMAPS low-level descriptors (time-series)
     - input_phone
         phone transcription, from forced alignment
     - input_words
         word transcription, from forced alignment

     - target_cat_consensus
         3 letter emotion label, set as 'xxx' if there is no majority decision between annotators
     - target_one_of_k_cat
         target_cat_consensus as a 1-hot vector encoding
     - target_cat_anno
         dictionary of annotator's and subject's categorical label
         One of: Anger, Disgust, Excited, Fear, Frustration, Happiness, Neutral, Other, Sadness, Surprise
     - target_categorical
         "Multi-hot" encoding of target_cat_anno, each annotator contributes a vector of their categorical labels
         Annotator label vectors are summed together and normalised to sum to 1
     - target_dimensional_raw
         3 dimensional emotion label, an average of annotator labels (not including self-evaluation)
     - target_dimensional
         target_dimensional_raw normalised to range [0, 1]
     - target_dim_anno_raw
         dicitonary of annotator's and subject's dimensional label
     - target_dim_anno
         target_dim_anno_raw normalised to range [0, 1]
    """

    def __init__(self, utt_path,
                 load_wavefile=False, gen_spectrogram=False, fft_width=256, fft_overlap=128,
                 load_func_features=True,
                 load_lld_features=True, include_frametime=False, flatten=False,
                 load_phone_alignment=False,
                 load_transcription=False,
                 load_emo_labels=True,
                 emotion_mask=np.ones(categories.shape).astype(np.bool),
                 emotion_mask_profile=np.ones(CATEGORIES.shape).astype(np.bool),
                 **kwargs):
        self.utt_path = utt_path
        self.session, self.sess_name, self.utt_name = self.utt_path.split('/')[-3:]

        if load_wavefile or gen_spectrogram:
            data = load_from_file(os.path.join(self.utt_path, 'wavefile'))
            self.__dict__.update(data)

            if gen_spectrogram:
                fft_width = fft_width
                fft_overlap = fft_overlap
                self.spectrogram = specgram(self.input_wave, NFFT=fft_width, Fs=self.sample_rate, noverlap=fft_overlap)[0]

        if load_func_features:
            data = load_from_file(os.path.join(self.utt_path, 'func_features'))
            self.__dict__.update(data)

        if load_lld_features:
            data = load_from_file(os.path.join(self.utt_path, 'lld_features'))

            if not include_frametime:  # if we dont want to include the frame time
                data['input_llds'] = data['input_llds'][:, 1:]
                data['input_llds_raw'] = data['input_llds_raw'][:, 1:]

            if flatten:  # if we want to flatten the time-series data
                data['input_llds'] = data['input_llds'].flatten()
                data['input_llds_raw'] = data['input_llds_raw'].flatten()

            self.__dict__.update(data)

        if load_phone_alignment:
            data = load_from_file(os.path.join(self.utt_path, 'phone_alignment'))
            self.__dict__.update(data)

        if load_transcription:
            data = load_from_file(os.path.join(self.utt_path, 'transcription'))
            self.__dict__.update(data)

        if load_emo_labels:
            data = load_from_file(os.path.join(self.utt_path, 'emo_labels'))
            self.__dict__.update(data)
            self.emotion_mask = emotion_mask
            self.emotion_mask_profile = emotion_mask_profile

    @property
    def target_one_of_k_cat_masked(self):
        return self.target_one_of_k_cat[self.emotion_mask]

    @property
    def target_categorical_masked(self):
        return self.target_categorical[self.emotion_mask_profile]


# Used to pre-process the IEMOCAP data to reduce load times
class IEMOCAPProcessor(object):

    def __init__(self):

        # Get all tuples of session number, session name, utterance name for access of files per utterance
        self.utt_names = np.array([
            (session, sess_name, utt_name[:-4])
            for session in filter(lambda s: s[:7] == 'Session', os.listdir(IEMOCAP_source))
            for sess_name in filter(lambda s: s[:3] == 'Ses',
                                    os.listdir(os.path.join(IEMOCAP_source, session, 'sentences', 'wav')))
            for utt_name in filter(lambda s: s[:3] == 'Ses',
                                   os.listdir(os.path.join(IEMOCAP_source, session, 'sentences', 'wav', sess_name)))
            if utt_name[-4:] == '.wav' and utt_name[:-4] not in missing_data
        ])

    def process_utts(self, utt_names=None):
        if utt_names is None:
            utt_names = self.utt_names

        for utt_info in utt_names:
            wave_path, func_path, llds_path, phone_path, words_path, target_labels_path = self.get_filepaths(*utt_info)
            target_path = os.path.join(IEMOCAP_source, 'preprocessed', *utt_info)

            if not os.path.exists(target_path):  # ensure directory already exists
                os.makedirs(target_path)

            self.preprocess_wavefile(wave_path, target_path)
            self.preprocess_func_features(func_path, target_path)
            self.preprocess_lld_features(llds_path, target_path)
            self.preprocess_phone_alignment(phone_path, target_path)
            self.preprocess_transcription(words_path, target_path)
            self.preprocess_emo_labels(target_labels_path, target_path, utt_info[-1])

    def speaker_stats_wave(self, session, sess_name, utt_name):
        gender = utt_name[-4]
        speaker = '{}-{}-wavefile'.format(session, gender)

        if speaker not in self.__dict__:
            input_waves = []

            for _session, _sess_name, _utt_name in self.utt_names:
                # if the utterance is for the right speaker
                if _session == session and _utt_name[-4] == gender:
                    utt_path = os.path.join(IEMOCAP_source, _session, 'sentences', 'wav',
                                            _sess_name, '{}.wav'.format(_utt_name))
                    input_waves.extend(read_wavfile(utt_path)[1])

            self.__dict__[speaker] = np.mean(input_waves, axis=0), np.std(input_waves, axis=0)

        return self.__dict__[speaker]

    def speaker_stats_funcs(self, session, sess_name, utt_name):
        gender = utt_name[-4]
        speaker = '{}-{}-func_features'.format(session, gender)

        if speaker not in self.__dict__:
            input_funcs = []

            for _session, _sess_name, _utt_name in self.utt_names:
                # if the utterance is for the right speaker
                if _session == session and _utt_name[-4] == gender:
                    utt_path = os.path.join(IEMOCAP_source, session, 'sentences', 'eGeMAPS',
                                            _sess_name, '{}.eGeMAPS.func.csv'.format(_utt_name))
                    input_funcs.append(np.loadtxt(utt_path, delimiter=';', usecols=range(2, 90), skiprows=1))

            self.__dict__[speaker] = np.mean(input_funcs, axis=0), np.std(input_funcs, axis=0)

        return self.__dict__[speaker]

    def speaker_stats_llds(self, session, sess_name, utt_name):
        gender = utt_name[-4]
        speaker = '{}-{}-lld_features'.format(session, gender)

        if speaker not in self.__dict__:
            input_lldss = []

            for _session, _sess_name, _utt_name in self.utt_names:
                # if the utterance is for the right speaker
                if _session == session and _utt_name[-4] == gender:
                    utt_path = os.path.join(IEMOCAP_source, session, 'sentences', 'eGeMAPS',
                                            _sess_name, '{}.eGeMAPS.lld.csv'.format(_utt_name))
                    input_lldss.extend(np.loadtxt(utt_path, delimiter=';', skiprows=1, usecols=range(1, 25)))

            self.__dict__[speaker] = np.mean(input_lldss, axis=0), np.std(input_lldss, axis=0)

        return self.__dict__[speaker]

    def preprocess_wavefile(self, source_path, target_path, **kwargs):
        # Load the wavefile
        sample_rate, input_wave = read_wavfile(source_path)

        # Perform speaker normalisation on the wavefile, (x - mean) / std
        session, sess_name, utt_name = target_path.split('/')[-3:]
        mean, std = self.speaker_stats_wave(session, sess_name, utt_name)
        input_wave_norm = (input_wave - mean) / std

        data = {'sample_rate': sample_rate, 'input_wave_raw': input_wave, 'input_wave': input_wave_norm}
        path = os.path.join(target_path, 'wavefile')
        save_to_file(data, path)

    def preprocess_func_features(self, source_path, target_path, **kwargs):
        # Load the eGeMAPS functionals
        input_func = np.loadtxt(source_path, delimiter=';', usecols=range(2, 90), skiprows=1)

        # Perform speaker normalisation on each functional (ie find 88 mean and variance values per speaker)
        session, sess_name, utt_name = target_path.split('/')[-3:]
        mean, std = self.speaker_stats_funcs(session, sess_name, utt_name)
        input_func_norm = (input_func - mean) / std

        data = {'input_func_raw': input_func, 'input_func': input_func_norm}
        path = os.path.join(target_path, 'func_features')
        save_to_file(data, path)

    def preprocess_lld_features(self, source_path, target_path, **kwargs):
        # Load the eGeMAPS low level descriptor time series
        input_llds = np.loadtxt(source_path, delimiter=';', skiprows=1, usecols=range(1, 25))

        # Perform speaker normalisation on each LLD (ie find 23 mean and variance values per speaker)
        session, sess_name, utt_name = target_path.split('/')[-3:]
        mean, std = self.speaker_stats_llds(session, sess_name, utt_name)
        input_llds_norm = (input_llds - mean) / std

        data = {'input_llds_raw': input_llds, 'input_llds': input_llds_norm}
        path = os.path.join(target_path, 'lld_features')
        save_to_file(data, path)

    def preprocess_phone_alignment(self, source_path, target_path, **kwargs):
        # Load the phone segmentation of the utterance
        phone_raw = open(source_path, 'r').read()
        phone_clean = re.sub('[ \t\r\f\v]+', ' ', phone_raw).strip()  # remove whitespace
        phone_lines = phone_clean.split('\n')[1:]  # separate lines and remove first line
        input_phone = map(lambda line: line.strip().split(' ')[-1].split('_'), phone_lines)  # select last column

        data = {'input_phone': input_phone}
        path = os.path.join(target_path, 'phone_alignment')
        save_to_file(data, path)

    def preprocess_transcription(self, source_path, target_path, **kwargs):
        # Load the word transcription of the utterance
        words_raw = open(source_path, 'r').read()
        words_clean = re.sub('[ \t\r\f\v]+', ' ', words_raw).strip()  # remove whitespace
        word_lines = words_clean.split('\n')[1:-1]  # separate lines and remove first and last lines
        word_list_raw = map(lambda line: line.strip().split(' ')[-1], word_lines)  # select the last column
        word_list_clean = map(lambda word: re.sub('\(\d+\)', '', word), word_list_raw)  # remove (\d) regex
        input_words = filter(lambda word: word not in ['<s>', '</s>'], word_list_clean)  # remove start/end symbols

        data = {'input_words': input_words}
        path = os.path.join(target_path, 'transcription')
        save_to_file(data, path)

    def preprocess_emo_labels(self, source_path, target_path, utt_name, **kwargs):
        # Load the emotion labels
        with open(source_path, 'r') as f:
            # search for the summary line containing the current utterance
            line = ''
            while utt_name not in line:
                line = f.readline()

            # split by the whitespace separating the 4 elements on the summary line
            vals = re.split('[\t\r\f\v]', line.strip())
            # select the third and fourth items on the summary line, ie the categorical and dimensional labels
            target_cat_consensus, target_dimensional = vals[2], eval(vals[3])

            assert target_cat_consensus in categories, (
                'target_cat_consensus not one of 11 predefined consensus categories'
                'got {}'.format(target_cat_consensus))
            assert len(target_dimensional) == 3, (
                'target_dimensional does not contain 3 values'
                'target_dimensional: {}'.format(target_dimensional))

            # continue until we reach the line break between utterances
            target_cat_anno, target_dim_anno = {}, {}
            line = f.readline()
            while not re.match('^\s*$', line):  # while the line is not just whitespace

                annotator, vals = line.split(':')  # separate the annotator from the values

                vals_list = vals.strip().split(';')[:-1]  # split on delimiter and remove last column (comments)
                vals_list = map(str.strip, vals_list)  # strip whitespace from around category/dimension values

                if re.match('^C-', annotator):  # categorical evaluators
                    # remove any extra values caused by spurious semi-colons
                    target_cat_anno[annotator[2:]] = filter(lambda val: val in CATEGORIES, vals_list)

                elif re.match('^A-', annotator):  # dimensional evaluators
                    # remove the string descriptor from the values by splitting on the whitespace between them
                    target_dim_anno[annotator[2:]] = map(
                        lambda dim_val: float(re.split('\s+', dim_val.strip())[-1]), vals_list)

                line = f.readline()  # prepare the next line in the file for the while statement

        # Create one-hot encoding of categorical consensus from annotators
        target_one_of_k_cat = (target_cat_consensus == categories).astype(int)

        # Normalise dimensional labels to be in range [0, 1]
        target_dimensional_norm = (np.array(target_dimensional) - 1.) / 4.
        target_dim_anno_norm = {annotator: (np.array(vals) - 1.) / 4. for annotator, vals in
                                target_dim_anno.iteritems()}

        # Create multi-hot encoding of categorical annotations
        target_categorical = np.sum(  # sum all contributions from all annotators
            # for all annotators, create one-hot vector for all labels v,
            # weighting contribution of v by number of vals (e.g. using np.mean)
            # only use experienced annotators (i.e. not the self annotations
            map(lambda (k, vals): np.mean(map(lambda v: CATEGORIES == v, vals), axis=0),
                filter(lambda (annotator, v): 'E' in annotator, target_cat_anno.iteritems())),
            axis=0)
        # normalise categorical annotation encoding to sum to 1
        target_categorical = target_categorical / np.sum(target_categorical)

        data = {'target_cat_consensus': target_cat_consensus, 'target_one_of_k_cat': target_one_of_k_cat,
                'target_cat_anno': target_cat_anno, 'target_categorical': target_categorical,
                'target_dimensional_raw': target_dimensional, 'target_dimensional': target_dimensional_norm,
                'target_dim_anno_raw': target_dim_anno, 'target_dim_anno': target_dim_anno_norm}
        path = os.path.join(target_path, 'emo_labels')
        save_to_file(data, path)

    # Given the session number (1 - 5), the session name, and the utterance name, retrieve the relevant filenames
    def get_filepaths(self, session, sess_name, utt_name):
        sess_path = os.path.join(IEMOCAP_source, session)

        wave_path = os.path.join(sess_path, 'sentences', 'wav', sess_name, '{}.wav'.format(utt_name))
        func_path = os.path.join(sess_path, 'sentences', 'eGeMAPS', sess_name, '{}.eGeMAPS.func.csv'.format(utt_name))
        llds_path = os.path.join(sess_path, 'sentences', 'eGeMAPS', sess_name, '{}.eGeMAPS.lld.csv'.format(utt_name))
        phone_path = os.path.join(sess_path, 'sentences', 'ForcedAlignment', sess_name, '{}.syseg'.format(utt_name))
        words_path = os.path.join(sess_path, 'sentences', 'ForcedAlignment', sess_name, '{}.wdseg'.format(utt_name))
        target_labels_path = os.path.join(sess_path, 'dialog', 'EmoEvaluation', '{}.txt'.format(sess_name))

        assert os.path.isfile(wave_path), ('input_wave does not exist at {}'.format(wave_path))
        assert os.path.isfile(func_path), ('input_func does not exist  at {}'.format(func_path))
        assert os.path.isfile(llds_path), ('input_llds does not exist  at {}'.format(llds_path))
        assert os.path.isfile(phone_path), ('input_phone does not exist  at {}'.format(phone_path))
        assert os.path.isfile(words_path), ('input_words does not exist  at {}'.format(words_path))
        assert os.path.isfile(target_labels_path), ('target_labels does not exist  at {}'.format(target_labels_path))

        return wave_path, func_path, llds_path, phone_path, words_path, target_labels_path

    def get_wave_path(self, session, sess_name, utt_name, sess_path=None):
        if sess_path is None:
            sess_path = os.path.join(IEMOCAP_source, session)


# ----------------------- #
#        Blizzard         #
# ----------------------- #


class BlizzardData(_AbstractProvider):
    """
    Populates list of utterance paths from the blizzard directory
    """

    def __init__(self, data_config={}, batch_size=50, shuffle_data=False, rng_seed=DEFAULT_SEED):
        if 'emotion_mask' in data_config:
            assert list(data_config['emotion_mask']).count(True) == 4, (
                'Blizzard data is only labelled with 4 emotions; angry, happy, neutral, sad'
                '\nGot {}'.format(categories[data_config['emotion_mask']]))

        file_paths = self.get_file_paths(self.get_utt_names())
        data_splitter = self.merlin_data_split()

        super(BlizzardData, self).__init__(file_paths, data_config, batch_size, shuffle_data, rng_seed, data_splitter)

    # return a list of utt_paths to visualise, if no utts specified then all utts are returned
    def get_file_paths(self, utt_names=[]):
        base_path = os.path.join(Blizzard_path, 'preprocessed')

        if not os.path.isdir(base_path):
            raise OSError('No preprocessed directory found, please run BlizzardProcessor first')

        return np.array([
            os.path.join(base_path, utt_name)  # if utt_names == [], then don't filter
            for utt_name in filter(lambda s: len(utt_names) == 0 or s in utt_names, os.listdir(base_path))
        ])

    def get_utt_names(self):
        return load_from_file(os.path.join(os.environ['project_DIR'], 'resources', 'file_id_lists', 'Blizzard', 'file_id_list'), '.scp')

    def merlin_data_split(self):
        def func(_self):
            base_path = os.path.join(os.environ['project_DIR'], 'resources', 'file_id_lists', 'Blizzard')

            file_names_train = load_from_file(os.path.join(base_path, 'train'), '.scp')
            _self.file_paths_train = _self.get_file_paths(file_names_train)

            file_names_valid = load_from_file(os.path.join(base_path, 'valid'), '.scp')
            _self.file_paths_valid = _self.get_file_paths(file_names_valid)

            file_names_test = load_from_file(os.path.join(base_path, 'test'), '.scp')
            _self.file_paths_test = _self.get_file_paths(file_names_test)

        return func

    def load_data(self, file_paths):
        return BlizzardBatch(file_paths, **self.data_config)


class BlizzardTestData(_AbstractProvider):

    def __init__(self, data_config={}, batch_size=50, shuffle_data=False, rng_seed=DEFAULT_SEED):
        file_paths = self.get_file_paths(self.get_utt_names())
        data_splitter = self.all_train_data()

        super(BlizzardTestData, self).__init__(file_paths, data_config, batch_size, shuffle_data, rng_seed, data_splitter)

    # return a list of utt_paths to visualise, if no utts specified then all utts are returned
    def get_file_paths(self, utt_names=[]):
        base_path = os.path.join(BlizzardTest_path, 'preprocessed')

        if not os.path.isdir(base_path):
            raise OSError('No preprocessed directory found, please run BlizzardProcessor first')

        return np.array([
            os.path.join(base_path, utt_name)  # if utt_names == [], then don't filter
            for utt_name in filter(lambda s: len(utt_names) == 0 or s in utt_names, os.listdir(base_path))
        ])

    def get_utt_names(self):
        return load_from_file(os.path.join(os.environ['project_DIR'], 'resources', 'file_id_lists', 'BlizzardTest', 'file_id_list'), '.scp')

    def all_train_data(self):
        def func(_self):
            base_path = os.path.join(os.environ['project_DIR'], 'resources', 'file_id_lists', 'BlizzardTest')

            file_names_train = load_from_file(os.path.join(base_path, 'file_id_list'), '.scp')
            _self.file_paths_train = _self.get_file_paths(file_names_train)

            _self.file_paths_valid = _self.get_file_paths([])
            _self.file_paths_test = _self.get_file_paths([])

        return func

    def load_data(self, file_paths):
        return BlizzardBatch(file_paths, **self.data_config)


class BlizzardBatch(_AbstractBatch):

    def __init__(self, file_paths, **kwargs):
        super(BlizzardBatch, self).__init__([BlizzardUtterance(file_path, **kwargs) for file_path in file_paths])

    @property
    def target_one_of_k_cat_masked(self):
        # filters the categorical output vector so only emotions in the emotion_mask are included
        return self.target_one_of_k_cat


class BlizzardUtterance(object):

    def __init__(self, utt_path,
                 load_wavefile=False, gen_spectrogram=False, fft_width=256, fft_overlap=128,
                 load_func_features=True,
                 load_lld_features=False, include_frametime=False, flatten=False,
                 load_emo_labels=True,
                 **kwargs):
        self.utt_path = utt_path
        self.utt_name = self.utt_path.split('/')[-1]

        if load_wavefile:
            data = load_from_file(os.path.join(self.utt_path, 'wavefile'))
            self.__dict__.update(data)

            if gen_spectrogram:
                self.spectrogram = specgram(self.input_wave, NFFT=fft_width, Fs=self.sample_rate, noverlap=fft_overlap)[0]

        if load_func_features:
            data = load_from_file(os.path.join(self.utt_path, 'func_features'))
            self.__dict__.update(data)

        if load_lld_features:
            data = load_from_file(os.path.join(self.utt_path, 'lld_features'))

            if not include_frametime:  # if we dont want to include the frame time
                data['input_llds'] = data['input_llds'][:, 1:]
                data['input_llds_raw'] = data['input_llds_raw'][:, 1:]

            if flatten:  # if we want to flatten the time-series data
                data['input_llds'] = data['input_llds'].flatten()
                data['input_llds_raw'] = data['input_llds_raw'].flatten()

            self.__dict__.update(data)

        if load_emo_labels:
            data = load_from_file(os.path.join(self.utt_path, 'emo_labels'))
            self.__dict__.update(data)

    @property
    def target_one_of_k_cat_masked(self):
        return self.target_one_of_k_cat


class BlizzardProcessor(object):

    def __init__(self, wav_dir='wav_16000', source=Blizzard_source, target=Blizzard_path):
        self.wav_dir = wav_dir
        self.source = source
        self.target = target
        self.utt_names = np.array(map(lambda utt_name: utt_name[:-4],
                                      filter(lambda utt_name: utt_name[-4:] == '.wav' and utt_name[:-4] not in missing_data,
                                             os.listdir(os.path.join(self.source, self.wav_dir)))))

    def process_utts(self, utt_names=None):
        # allow user to specify certain subset of utterances to preprocess
        if utt_names is not None:
            prev_utt_names = self.utt_names
            self.utt_names = utt_names

        target_path = os.path.join(self.target, 'preprocessed')

        print('beginning pre-processing for Blizzard 2017 data')
        print('data will be saved to {}\n'.format(target_path))
        self.preprocess_wavefiles(target_path)
        self.preprocess_func_features(target_path)
        # self.preprocess_lld_features(target_path)

        # set utt_names back to the full list
        if utt_names is not None:
            self.utt_names = prev_utt_names

    def preprocess_wavefiles(self, target_path):

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Calculating wave normalisation statistics')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # load all wavefiles
        num_samples = 0
        val_sum, sqr_sum = 0., 0.
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, self.wav_dir, '{}.wav'.format(utt_name))
            _, input_wave = read_wavfile(utt_path)
            if input_wave.ndim == 2:
                input_wave = np.mean(input_wave, axis=1)
            num_samples += input_wave.shape[0]

            val_sum += np.sum(input_wave)
            sqr_sum += np.sum(input_wave ** 2)

        # calculate summary statistics
        mean = val_sum / num_samples
        std = np.sqrt((sqr_sum / num_samples) - (mean ** 2))
        print('mean {}\nstd {}\nnum_samples {}'.format(mean, std, num_samples))

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Processing for wave files finished')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # save raw and normalised wavefile for each utterance
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, self.wav_dir, '{}.wav'.format(utt_name))
            sample_rate, input_wave = read_wavfile(utt_path)
            if input_wave.ndim == 2:
                input_wave = np.mean(input_wave, axis=1)

            # Perform speaker normalisation on the wavefile, (x - mean) / std
            input_wave_norm = (input_wave - mean) / std

            if not os.path.exists(os.path.join(target_path, utt_name)):  # ensure directory already exists
                os.makedirs(os.path.join(target_path, utt_name))
            data = {'sample_rate': sample_rate,
                    'input_wave_raw': input_wave,
                    'input_wave': input_wave_norm}
            path = os.path.join(target_path, utt_name, 'wavefile')
            save_to_file(data, path)

    def preprocess_func_features(self, target_path):

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Calculating func normalisation statistics')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # load all functional features
        num_samples = 0
        val_sum, sqr_sum = np.zeros(88), np.zeros(88)
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, 'eGeMAPS', '{}.eGeMAPS.func.csv'.format(utt_name))
            input_func = np.loadtxt(utt_path, delimiter=';', usecols=range(2, 90), skiprows=1)
            num_samples += 1

            val_sum += input_func
            sqr_sum += input_func ** 2

        # calculate summary statistics
        mean = val_sum / num_samples
        std = np.sqrt((sqr_sum / num_samples) - (mean ** 2))
        print('mean {}\nstd {}\nnum_samples {}'.format(mean, std, num_samples))

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Processing for func files finished')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # save raw and normalised functionals for each utterance
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, 'eGeMAPS', '{}.eGeMAPS.func.csv'.format(utt_name))
            input_func = np.loadtxt(utt_path, delimiter=';', usecols=range(2, 90), skiprows=1)

            # Perform speaker normalisation on each functional (ie find 88 mean and variance values per speaker)
            input_func_norm = (input_func - mean) / std

            if not os.path.exists(os.path.join(target_path, utt_name)):  # ensure directory already exists
                os.makedirs(os.path.join(target_path, utt_name))
            data = {'input_func_raw': input_func, 'input_func': input_func_norm}
            path = os.path.join(target_path, utt_name, 'func_features')
            save_to_file(data, path)

    def preprocess_lld_features(self, target_path):

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Calculating LLDs normalisation statistics')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # load all LLDs
        num_samples = 0
        val_sum, sqr_sum = np.zeros(24), np.zeros(24)
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, 'eGeMAPS', '{}.eGeMAPS.lld.csv'.format(utt_name))
            input_llds = np.loadtxt(utt_path, delimiter=';', skiprows=1, usecols=range(1, 25))
            num_samples += input_llds.shape[0]

            val_sum += np.sum(input_llds, axis=0)
            sqr_sum += np.sum(input_llds ** 2, axis=0)

        # calculate summary statistics
        mean = val_sum / num_samples
        std = np.sqrt((sqr_sum / num_samples) - (mean ** 2))

        # the first item is time steps, this should not be normalised
        mean[0], std[0] = 0., 1.
        print('mean {}\nstd {}\nnum_samples {}'.format(mean, std, num_samples))

        print('\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        print('Processing for LLDs files finished')
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')

        # save raw and normalised LLDs for each utterance
        for utt_name in self.utt_names:
            utt_path = os.path.join(self.source, 'eGeMAPS', '{}.eGeMAPS.lld.csv'.format(utt_name))
            input_llds = np.loadtxt(utt_path, delimiter=';', skiprows=1, usecols=range(1, 25))

            # Perform speaker normalisation on each LLD (ie find 23 mean and variance values per speaker)
            input_llds_norm = (input_llds - mean) / std

            if not os.path.exists(os.path.join(target_path, utt_name)):  # ensure directory already exists
                os.makedirs(os.path.join(target_path, utt_name))
            data = {'input_llds_raw': input_llds, 'input_llds': input_llds_norm}
            path = os.path.join(target_path, utt_name, 'lld_features')
            save_to_file(data, path)



