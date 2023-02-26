import pandas as pd
import numpy as np
import pickle
import re
import os
import warnings
from datetime import datetime
from dataclasses import dataclass
from abc import abstractmethod, ABCMeta
from .Constants import SystemPaths
# 機械学習用ライブラリの読み込み
import optuna.integration.lightgbm as lgb
import optuna.logging as oplog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# モデルの学習・テストを行う抽象クラス
class _AbstractModel(metaclass=ABCMeta):
    @abstractmethod
    def train_with_tuning(self):
        # 学習・チューニング用メソッド
        pass
    @abstractmethod
    def test(self):
        # テスト用メソッド
        pass
    @abstractmethod
    def predict(self, data):
        # 推論用メソッド
        pass
    @abstractmethod
    def feature_importance(self):
        # 特徴量の重要度を計算するためのメソッド
        pass

# モデル生成を行う抽象クラス
class _AbstractFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, df, test_size, valid_size):
        # 具象クラスで定義したインスタンスを生成するためのメソッド
        pass
    @abstractmethod
    def save(self, manager, version):
        # 生成したインスタンスを保存するためのメソッド
        pass
    @abstractmethod
    def load(self, filepath):
        # 保存したインスタンスを読み込むためのメソッド
        pass

@dataclass
class _AIManager:
    def save(self, target, filepath):
        """
        データ保存用メソッド

        Parameters
        ----------
        target : object
            保存するオブジェクト
        filepath : str
            保存先
        """
        dirname = os.path.dirname(filepath)

        # 保存先のディレクトリがない場合
        if not os.path.isdir(dirname):
            # ディレクトリ作成
            os.makedirs(dirname)

        with open(filepath, 'wb') as fout:
            pickle.dump(target, fout)

    @classmethod
    def load(cls, filepath):
        """
        データ読み込み用メソッド

        Parameters
        ----------
        filepath : str
            読み込み先

        Returns
        -------
        target : object
            保存したデータ
        """
        if os.path.exists(filepath):
            with open(filepath, 'rb') as fin:
                target = pickle.load(fin)
        else:
            raise Exception(f'Error: {filepath} does not exist.')

        return target

@dataclass(init=False)
class _LGBManager(_AIManager):
    train_dataset: lgb.Dataset
    validation_dataset: lgb.Dataset
    test_dataset: lgb.Dataset
    model: lgb.LightGBMTuner = None

    def __init__(self, train, valid, test, model=None):
        """
        初期化

        Parameters
        ----------
        train : dict
            学習用データセット
        valid : dict
            検証用データセット
        test : dict
            テスト用データセット
        model : lgb.LightGBMTuner or None
            モデル
        """
        self.train_dataset = lgb.Dataset(**train, free_raw_data=False)
        self.validation_dataset = lgb.Dataset(**valid, reference=self.train_dataset, free_raw_data=False)
        self.test_dataset = lgb.Dataset(**test, free_raw_data=False)
        self.model = model

    def save(self, target, filepath):
        """
        データ保存用メソッド

        Parameters
        ----------
        target : object
            保存対象のオブジェクト
        filepath : str
            保存先
        """
        output = {}
        # クラス変数のみ抽出
        pattern = re.compile('^__')
        _func = lambda key: not bool(pattern.match(key)) and hasattr(target, key) and not callable(getattr(target, key))
        items = filter(_func, dir(target))

        with warnings.catch_warnings():
            # 一時的に警告を無視する
            warnings.filterwarnings('ignore')

            for key in items:
                value = getattr(self, key)

                if isinstance(value, lgb.Dataset):
                    # データの取得
                    dataset = value.set_reference(lgb.Dataset(None)).construct()
                    _data = dataset.get_data()
                    _label = dataset.get_label()
                    _group = dataset.get_group()
                    value = {'data': _data, 'label': _label, 'group': _group}
                output[key] = value

        super().save(output, filepath)

    @classmethod
    def create_inputs(cls, df):
        """
        DataFrameから入力データを作成する

        Parameters
        ----------
        df : pd.DataFrame
            入力データ

        Returns
        -------
        inputs : dict
            出力データ
        """
        label_name = 'rank'
        judge = df[label_name] <= 3
        target = df.copy()
        target.loc[ judge, label_name] = 1
        target.loc[~judge, label_name] = 0
        target = target[df[label_name] <= 6]
        # データの取得
        target.sort_index(inplace=True)
        _data = target.drop([label_name], axis=1)
        _label = target[label_name]
        _group = target.index.value_counts().sort_index()
        # 出力データの生成
        inputs = {'data': _data, 'label': _label, 'group': _group}

        return inputs

    @classmethod
    def load(cls, filepath):
        """
        データを読み込みインスタンスを作成する

        Parameters
        ----------
        filepath : str
            読み込み先

        Returns
        -------
        instance : _LGBManager
            _LGBManagerのインスタンス
        """
        target = super().load(filepath)
        dummy_data = {'data': None}
        inputs = {
            'train': target.get('train_dataset', dummy_data),
            'valid': target.get('validation_dataset', dummy_data),
            'test':  target.get('test_dataset', dummy_data),
            'model': target.get('model', None),
        }
        instance = cls(**inputs)

        return instance

# モデルの定義
class _LightGBM(_AbstractModel):
    """
    LightGBM : 決定木をベースとした手法による機械学習手法

    Attributes
    ----------
    __manager : _LGBManager
        _LGBManagerのインスタンス
    __callback : object or None
        __del__時に呼び出すコールバック
    """
    def __init__(self, manager, callback=None):
        """
        初期化

        Parameters
        ----------
        manager : _LGBManager
            _LGBManagerのインスタンス
        callback : object or None
            __del__時に呼び出すコールバック
        """
        self.__manager = manager
        self.__callback = callback

    def __del__(self):
        """
        破棄処理
        """
        if callable(self.__callback):
            self.__callback(self)

    def train_with_tuning(self, seed=3, optuna_seed=7):
        """
        ハイパーパラメータのチューニングも行いつつ学習する

        Parameters
        ----------
        seed : int
            チューニング時の乱数のシード
        optuna_seed : int
            ハイパーパラメータ探索時の乱数のシード

        Returns
        -------
        best_params : pd.DataFrame
            データから予測した最適なパラメータ
        """
        # 固定するパラメータの設定
        params = {
            'objective': 'binary',         # 目的関数
            'metric':    'binary_logloss', # 評価指標
            'boosting':  'gbdt',           # 勾配ブースティングの種類
            'verbose':   -1,               # ログの出力を抑制
            'seed':      seed,             # 乱数のシード
        }
        callbacks = [
            lgb.early_stopping(stopping_rounds=16, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        # ロギングの設定変更
        oplog.set_verbosity(oplog.WARNING)
        # ハイパーパラメータ探索付きの学習を実施
        with warnings.catch_warnings():
            # 一時的に警告を無視する
            warnings.filterwarnings('ignore')

            self.__manager.model = lgb.train(
                params, self.__manager.train_dataset,
                valid_sets=[self.__manager.train_dataset, self.__manager.validation_dataset],
                valid_names=['train', 'valid'],
                optuna_seed=optuna_seed,
                callbacks=callbacks,
            )

        best_params = pd.DataFrame({
            key: [val]
            for key, val in self.__manager.model.params.items()
        }).T.set_axis(['best_params'], axis='columns')

        return best_params

    def test(self):
        """
        テスト（評価）

        Returns
        -------
        result : pd.DataFrame
            評価結果
        """
        # テスト用データを取得
        test_dataset = self.__manager.test_dataset.construct()
        x_test = test_dataset.get_data()
        y_test = test_dataset.get_label()
        # 推論
        _, y_pred = self.predict(x_test)
        # 分類性能を計算
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['other', 'Top3'])
        # 整形
        result = pd.DataFrame(report).T

        return result

    def predict(self, data):
        """
        推論

        Parameters
        ----------
        data : numpy.array or pd.DataFrame
            入力データ

        Returns
        -------
        probability : np.array
            予測確率
        prediction : np.array
            予測ラベル
        """
        probability = self.__manager.model.predict(data, num_iteration=self.__manager.model.best_iteration)
        # 最も近い整数に丸める
        prediction = np.rint(probability).astype(int)

        return probability, prediction

    def feature_importance(self):
        """
        特徴量の重要度の計算

        Returns
        -------
        result : pd.DataFrame
            特徴量の重要度
        """
        features = self.__manager.model.feature_name()
        importance = self.__manager.model.feature_importance()
        result = pd.DataFrame({
            'features': features,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return result

class LGBFactory(_AbstractFactory):
    """
    LGBFactory : lightGBM Factory

    Attributes
    ----------
    __managers : dict
        _LGBManagerのインスタンス一覧
    """
    def __init__(self):
        """
        初期化
        """
        self.__managers = {}

    def __update_instance_list(self, instance):
        hash_key = id(instance)

        if hash_key in self.__managers.keys():
            # 保持していた対応関係を削除
            del self.__managers[hash_key]

    def create(self, df, test_size=0.3, valid_size=0.2, seed=1):
        """
        具象クラス生成用のメソッド

        Parameters
        ----------
        df : pd.DataFrame
            特徴量エンジニアリング後のデータ
        test_size : float
            テストデータの割合
        valid_size : float
            検証用データの割合
        seed : int
            乱数のシード

        Returns
        -------
        instance : _LightGBM
            _LightGBMのインスタンス
        """
        _train_valid_indices, _test_indices = train_test_split(
            df.index.unique(), test_size=test_size, random_state=seed, shuffle=False
        )
        _train_indices, _valid_indices = train_test_split(
            _train_valid_indices, test_size=valid_size, random_state=seed+1, shuffle=False
        )
        _get_target_data = lambda _df, _indices: _df[_df.index.isin(_indices)]
        train_dataset = _get_target_data(df, _train_indices)
        validation_dataset = _get_target_data(df, _valid_indices)
        test_dataset = _get_target_data(df, _test_indices)
        # 訓練用データセット、検証用データセット、テスト用データセットの定義
        inputs = {
            'train': _LGBManager.create_inputs(train_dataset),
            'valid': _LGBManager.create_inputs(validation_dataset),
            'test':  _LGBManager.create_inputs(test_dataset),
        }
        # データセット作成
        manager = _LGBManager(**inputs)
        # 具象クラスのインスタンス作成
        instance = _LightGBM(manager, self.__update_instance_list)
        # instanceとmanagerを紐づけて管理
        self.__managers.update({id(instance): manager})

        return instance

    def save(self, instance, version='model'):
        """
        モデル保存用メソッド

        Parameters
        ----------
        instance : _LightGBM
            _LightGBMのインスタンス

        Returns
        -------
        relative_filepath : str
            出力先のファイルパス
        """
        now = datetime.now()
        dirname = now.strftime('%Y%m%d')
        filename = f'base{version}.pkl'
        filepath = os.path.join(SystemPaths.MODEL_DIR, dirname, filename)
        hash_key = id(instance)
        manager = self.__managers[hash_key]
        # 保存
        manager.save(manager, filepath)
        # 相対パスを返却
        relative_filepath = os.path.join(os.path.basename(SystemPaths.MODEL_DIR), dirname, filename)

        return relative_filepath

    def load(self, filepath):
        """
        モデル読み込み用メソッド

        Parameters
        ----------
        filepath : str
            読み込み先

        Returns
        -------
        instance : _LightGBM
            _LightGBMのインスタンス
        """
        manager = _LGBManager.load(filepath)
        instance = _LightGBM(manager)
        # instanceとmanagerを紐づけて管理
        self.__managers.update({id(instance): manager})

        return instance
