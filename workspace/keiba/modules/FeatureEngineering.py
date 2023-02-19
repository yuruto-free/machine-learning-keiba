import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .Constants import SystemPaths

class FeatureEngineering:
    """
    FeatureEngineering : 特徴量エンジニアリング用クラス

    Attributes
    ----------
    features : pd.DataFrame
        特徴量
    """
    def __init__(self, df):
        """
        初期化

        Parameters
        ----------
        df : pd.DataFrame
            入力データ
        """
        self.features = df

    def drop(self, columns):
        """
        指定されたcolumnを削除

        Parameters
        ----------
        columns : list
            削除対象のcolumn

        Returns
        -------
        self : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        self.features.drop(columns=columns, inplace=True)

        return self

    def dumminize(self, columns):
        """
        指定されたcolumnをダミー変数化

        Parameters
        ----------
        columns : list
            ダミー変数化の対象となるcolumn

        Returns
        -------
        self : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        if len(columns) > 0:
            dummy_df = pd.get_dummies(self.features[columns])
            extracted_df = self.features.loc[:, ~self.features.columns.isin(columns)].copy()
            self.features = pd.concat([extracted_df, dummy_df], axis=1)

        return self

    def encode_jockey_id(self):
        """
        jockey id をラベルエンコーディング

        Returns
        -------
        self : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        # jockey idを変換
        name = 'jockey_id'
        encoder = LabelEncoder()
        original = self.features[name].values
        transformed = encoder.fit_transform(original)
        df = pd.DataFrame({
            name: original,
            'encoded_id': transformed,
        })
        # 対応関係を保存
        df.to_pickle(SystemPaths.ENCODED_JOCKEYID_PATH)
        # 変換結果を格納
        self.features[name] = transformed
        self.features[name] = self.features[name].astype('category')

        return self

    def encode_horse_id(self):
        """
        horse id をラベルエンコーディング

        Returns
        -------
        self : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        # 集計対象のcolumnの定義
        columns = self.features.columns
        _peds = columns[columns.str.match('^peds_.*')].to_list()
        horse_ids = ['horse_id'] + _peds
        # 対象データの取得
        data = self.features[horse_ids].stack().reset_index(drop=True).unique()
        # ラベルエンコーディングの実施
        encoder = LabelEncoder()
        transformed = encoder.fit_transform(data)
        # 対応関係を保存
        df = pd.DataFrame({
            'horse_id': data,
            'encoded_id': transformed,
        })
        df.to_pickle(SystemPaths.ENCODED_HORSEID_PATH)
        # 変換結果を格納
        self.features[horse_ids] = self.features[horse_ids].apply(lambda series: encoder.transform(series))
        self.features[horse_ids] = self.features[horse_ids].astype('category')

        return self

    def label_encoding(self, columns):
        """
        指定されたcolumnをラベルエンコーディング

        Parameters
        ----------
        columns : list
            ラベルエンコーディングの対象となるcolumn

        Returns
        -------
        self : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        encoder = LabelEncoder()

        for name in columns:
            self.features[name] = encoder.fit_transform(self.features[name])
            self.features[name] = self.features[name].astype('category')

        return self
