import gc
import threading
from math import ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.utils import Sequence
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Dropout
from sklearn.model_selection import KFold

import const
from features import FeatherFeatureIDFromDF


class InstallmentPaymentsTimeseriesDAE(FeatherFeatureIDFromDF):

    @staticmethod
    def input():
        return const.in_inp

    @staticmethod
    def categorical_features():
        return []

    @staticmethod
    def create_features_from_dataframe(inp: pd.DataFrame, random_state: int):
        inp['AMT_PAYMENT'] = inp.AMT_PAYMENT.fillna(0)
        inp['DAYS_ENTRY_PAYMENT'] = inp.DAYS_ENTRY_PAYMENT.fillna(0)

        inp['PAID_LATE_IN_DAYS'] = inp['DAYS_ENTRY_PAYMENT'] - inp['DAYS_INSTALMENT']
        inp['PAID_OVER_AMOUNT'] = inp['AMT_PAYMENT'] - inp['AMT_INSTALMENT']

        inp = inp.sort_values(
            ['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']
        ).reset_index(drop=True)

        inp['RG_PAID_LATE_IN_DAYS'] = rank_gauss(inp.PAID_LATE_IN_DAYS.values)
        inp['RG_PAID_OVER_AMOUNT'] = rank_gauss(inp.PAID_OVER_AMOUNT.values)

        pivotdf = pd.DataFrame()
        print("Pivotting...")
        for c in ['RG_PAID_LATE_IN_DAYS', 'RG_PAID_OVER_AMOUNT']:
            partdf = pd.pivot_table(inp, index=['SK_ID_CURR', 'SK_ID_PREV'],
                                    values=c,
                                    columns=['NUM_INSTALMENT_NUMBER'])

            del partdf.columns.name
            partdf.columns = [c + "_" + str(col) for col in partdf.columns]
            partdf = partdf.reset_index().drop(columns=['SK_ID_PREV'])

            part_agg = partdf.groupby('SK_ID_CURR').agg(['min', 'max', 'mean'])
            part_agg.columns = [col[0] + '_' + col[1].upper() for col in part_agg.columns]
            part_agg.isnull().sum() / len(part_agg)

            drop_cols = part_agg.columns[((part_agg.isnull().sum() / len(part_agg)) > 0.8)]
            part_agg = part_agg.drop(columns=drop_cols).fillna(0)
            pivotdf = pd.concat([pivotdf, part_agg], axis=1)

        encoded = dae_encode(pivotdf)
        feats = pd.DataFrame(encoded, index=pivotdf.index,
                             columns=["INP_TS_DAE{}".format(i) for i in range(encoded.shape[1])])
        return feats.reset_index()


def rank_gauss(x, title=None):
    # Trying to implement rankGauss in python, here are my steps
    # 1) Get the index of the series
    # 2) sort the series
    # 3) standardize the series between -1 and 1
    # 4) apply erfinv to the standardized series
    # 5) create a new series using the index
    # Am i missing something ??
    # I subtract mean afterwards. And do not touch 1/0 (binary columns).
    # The basic idea of this "RankGauss" was to apply rank trafo and them shape them like gaussians.
    # Thats the basic idea. You can try your own variation of this.

    if (title != None):
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))
        fig.suptitle(title)
        axs[0][0].hist(x)

    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    if (title != None):
        print('1)', max(temp), min(temp))
        axs[0][1].hist(temp)
    rank_x = temp.argsort() / N
    if (title != None):
        print('2)', max(rank_x), min(rank_x))
        axs[0][2].hist(rank_x)
    rank_x -= rank_x.mean()
    if (title != None):
        print('3)', max(rank_x), min(rank_x))
        axs[1][0].hist(rank_x)
    rank_x *= 2
    if (title != None):
        print('4)', max(rank_x), min(rank_x))
        axs[1][1].hist(rank_x)
    efi_x = erfinv(rank_x)
    if (title != None):
        print('5)', max(efi_x), min(efi_x))
        axs[1][2].hist(efi_x)
    efi_x -= efi_x.mean()
    if (title != None):
        print('6)', max(efi_x), min(efi_x))
        axs[2][0].hist(efi_x)
        plt.show()

    return efi_x


# i'm doing this cause i don't know if some keras backend have threading problems...
class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        self._read_ready.release()


class DAESequence(Sequence):

    def __init__(self, df, batch_size=128, random_cols=.15,
                 random_rows=1, use_cache=False, use_lock=False, verbose=True):
        self.df = df.values.copy()  # ndarray baby
        self.batch_size = int(batch_size)
        self.len_data = df.shape[0]
        self.len_input_columns = df.shape[1]
        if (random_cols <= 0):
            self.random_cols = 0
        elif (random_cols >= 1):
            self.random_cols = self.len_input_columns
        else:
            self.random_cols = int(random_cols * self.len_input_columns)
        if (self.random_cols > self.len_input_columns):
            self.random_cols = self.len_input_columns
        self.random_rows = random_rows
        self.cache = None
        self.use_cache = use_cache
        self.use_lock = use_lock
        self.verbose = verbose

        self.lock = ReadWriteLock()
        self.on_epoch_end()

    def on_epoch_end(self):
        if (not self.use_cache):
            return
        if (self.use_lock):
            self.lock.acquire_write()
        if (self.verbose):
            print("Doing Cache")
        self.cache = {}
        for i in range(0, self.__len__()):
            self.cache[i] = self.__getitem__(i, True)
        if (self.use_lock):
            self.lock.release_write()
        gc.collect()
        if (self.verbose):
            print("Done")

    def __len__(self):
        return int(ceil(self.len_data / float(self.batch_size)))

    def __getitem__(self, idx, doing_cache=False):
        if (not doing_cache and self.cache is not None and not (self.random_cols <= 0 or self.random_rows <= 0)):
            if (idx in self.cache.keys()):
                if (self.use_lock):
                    self.lock.acquire_read()
                ret0, ret1 = self.cache[idx][0], self.cache[idx][1]
                if (self.use_lock):
                    self.lock.release_read()
                if (not doing_cache and self.verbose):
                    print('DAESequence Cache ', idx)
                return ret0, ret1

        idx_end = min(idx + self.batch_size, self.len_data)
        cur_len = idx_end - idx
        rows_to_sample = int(self.random_rows * cur_len)
        input_x = self.df[idx: idx_end]
        if (self.random_cols <= 0 or self.random_rows <= 0 or rows_to_sample <= 0):
            return input_x, input_x  # not dae
        # here start the magic
        random_rows = np.random.randint(low=0, high=self.len_data -
                                                    rows_to_sample, size=rows_to_sample)
        random_rows[random_rows > idx] += cur_len  # just to don't select twice the current rows
        cols_to_shuffle = np.random.randint(
            low=0, high=self.len_input_columns, size=self.random_cols)
        noise_x = input_x.copy()
        noise_x[0:rows_to_sample, cols_to_shuffle] = self.df[random_rows[:, None], cols_to_shuffle]
        if (not doing_cache and self.verbose):
            print('DAESequence ', idx)

        return noise_x, input_x


def dae_encode(data):
    print("Create Model")
    len_input_columns, len_data = data.shape[1], data.shape[0]

    NUM_GPUS = 1
    layer1 = int(len_input_columns * .5)
    bottleneck_layer = int(len_input_columns * .2)
    layer3 = int(len_input_columns * .5)

    stddev0 = np.sqrt(2 / (len_input_columns + layer1))
    stddev1 = np.sqrt(1 / (layer1 + bottleneck_layer))
    stddev2 = np.sqrt(1 / (bottleneck_layer + layer3))
    stddev3 = np.sqrt(1 / (layer3 + len_input_columns))

    kernel_initializer_0 = keras.initializers.RandomNormal(
        mean=-4.74564e-05, stddev=stddev0, seed=None
    )

    kernel_initializer_1 = keras.initializers.RandomNormal(
        mean=8.51905e-06, stddev=stddev1, seed=None
    )

    kernel_initializer_2 = keras.initializers.RandomNormal(
        mean=8.51905e-06, stddev=stddev2, seed=None
    )

    kernel_initializer_3 = keras.initializers.RandomNormal(
        mean=-1.80977e-05, stddev=stddev3, seed=None
    )

    num_fold = 5
    folds = KFold(n_splits=num_fold, shuffle=True, random_state=0)
    encoded = np.zeros((len_data, bottleneck_layer))

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        print("Fold, {}".format(n_fold))

        trn_data = data.iloc[trn_idx, :].copy()
        val_data = data.iloc[val_idx, :].copy()

        print("Input len=", len_input_columns, len_data)
        model_dae = Sequential()
        model_dae.add(Dense(units=layer1, activation='relu', dtype='float32',
                            name='Hidden1', input_shape=(len_input_columns,),
                            kernel_initializer=kernel_initializer_0))
        model_dae.add(Dense(units=bottleneck_layer, activation='relu', dtype='float32',
                            name='Bottleneck', kernel_initializer=kernel_initializer_1))
        model_dae.add(Dense(units=layer3, activation='relu', dtype='float32',
                            name='Hidden3', kernel_initializer=kernel_initializer_2))
        model_dae.add(Dense(units=len_input_columns, activation='linear', dtype='float32',
                            name='Output', kernel_initializer=kernel_initializer_3))
        # decay -> Oscar Takeshita comment
        model_opt = keras.optimizers.SGD(lr=0.1, decay=1 - 0.995, momentum=0, nesterov=False)

        try:
            print('Loading model from file')
            model_dae = keras.models.load_model('DAE.keras.model.h5')
        except Exception as e:
            print("Can't load previous fitting parameters and model", repr(e))
        if (NUM_GPUS > 1):
            try:
                multi_gpu_model = keras.utils.multi_gpu_model(model_dae, gpus=NUM_GPUS)
                multi_gpu_model.compile(loss='mean_squared_error', optimizer=model_opt)
                print("MULTI GPU MODEL")
                print(multi_gpu_model.summary())
            except Exception as e:
                print("Can't run multi gpu, error=", repr(e))
                model_dae.compile(loss='mean_squared_error', optimizer=model_opt)
                NUM_GPUS = 0
        else:
            model_dae.compile(loss='mean_squared_error', optimizer=model_opt)

        print("BASE MODEL")
        print(model_dae.summary())
        batch_size = 256
        multi_process_workers = 1

        model_dae.fit_generator(
            DAESequence(trn_data, batch_size=batch_size, verbose=False),
            steps_per_epoch=int(ceil(trn_data.shape[0] / batch_size)),
            epochs=100,
            workers=multi_process_workers,
            use_multiprocessing=True if multi_process_workers > 1 else False,
            validation_data=(val_data, val_data),
            verbose=1, callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0.00001, patience=10, verbose=0, mode='auto'
                )
            ])

        bottleneck_layer_model = Model(inputs=model_dae.input,
                                       outputs=model_dae.get_layer('Bottleneck').output)

        encoded[val_idx, :] = bottleneck_layer_model.predict(val_data)

    return encoded


if __name__ == "__main__":
    feat_path = InstallmentPaymentsTimeseriesDAE(const.FEATDIR).create_feature(0)
