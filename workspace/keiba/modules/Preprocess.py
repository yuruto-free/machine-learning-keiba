import numpy as np
import pandas as pd
import re
from .Constants import LocalPaths, HorseResultsCols, ResultsCols, Master
from .DataManager import DataManager

class _Extractor:
    """
    _Extractor : データ抽出器

    Attributes
    ----------
    __name : str
        column name of DataFrame
    __pattern : re.compile
        正規表現パターン
    __callback : callable or None
        callback function
    """
    def __init__(self, name, pattern, callback=None):
        """
        初期化処理

        Parameters
        ----------
        name : str
            column name of DataFrame
        pattern : re.compile
            正規表現パターン
        callback : callable or None
            callback function
        """
        self.__name = name
        self.__pattern = pattern
        self.__callback = callback

    def extract(self, series):
        """
        抽出器

        Parameters
        ----------
        series : pd.Series
            変換対象

        Returns
        -------
        result : pd.DataFrame
            変換結果
        """
        result = series.str.extract(f'(?P<{self.__name}>{self.__pattern})')

        if self.__callback is not None:
            result = result[self.__name].map(self.__callback)

        return result

class Preprocess:
    """
    Preprocess : 前処理用クラス

    Attributes
    ----------
    __class_name : str
        クラス名
    __date_format : str
        日付のフォーマット
    __course_len_scaler : float
        course_lenのスケーリング値
    __max_relatives : int
        対象とする親等の最大値（例：2親等 -> 2）
    """
    def __init__(self):
        """
        初期化処理
        """
        # クラスメソッド用の変数
        self.__class_name = self.__class__.__name__
        self.__date_format = '%Y-%m-%d'
        self.__course_len_scaler = 100.0  # course_lenの正規化
        self.__max_relatives = 2          # 対象とする親等の最大値（例：2親等 -> 2）

    def __read_pickle(self, filepath, err_msg='function'):
        """
        pickleファイルの読み込み

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        df : pd.DataFrame
            読み込んだDataFrame
        """
        try:
            df = pd.read_pickle(filepath)
        except Exception as e:
            raise Exception(f'Error({self.__class_name}::{err_msg}) {e}')

        return df

    def preprocess_rawdata(self,
                           results_filepath=LocalPaths.RAW_RESULTS_PATH,
                           raceinfo_filepath=LocalPaths.RAW_RACEINFO_PATH,
                           payback_filepath=LocalPaths.RAW_PAYBACK_PATH,
                           horseresults_filepath=LocalPaths.RAW_HORSERESULTS_PATH,
                           peds_filepath=LocalPaths.RAW_PEDS_PATH,
                           since_=None):
        """
        生データの前処理

        Parameters
        ----------
        results_filepath : str
            レース結果テーブルのファイルパス
        raceinfo_filepath : str
            レース情報テーブルのファイルパス
        payback_filepath : str
            払い戻し結果テーブルのファイルパス
        horseresults_filepath : str
            馬の過去成績テーブルのファイルパス
        peds_filepath : str
            血統情報テーブルのファイルパス
        since_ : datetime or None
            集計開始年月日

        Returns
        -------
        manager : _DataManager
            データ管理用インスタンス
        """
        # 前処理実行
        results = self.__preprocess_results(results_filepath)
        raceinfo = self.__preprocess_raceinfo(raceinfo_filepath)
        paybacks = self.__preprocess_payback(payback_filepath)
        horseresults = self.__preprocess_horseresults(horseresults_filepath)
        peds = self.__preprocess_peds(peds_filepath)
        # 集計期間の更新
        if since_ is not None:
            # 集計開始年月日以降に開催されたレースを抽出
            valid_indices = raceinfo[raceinfo[HorseResultsCols.DATE] >= since_].index
            # フィルタリング
            _func = lambda _df: _df[_df.index.isin(valid_indices)]
            results = _func(results)
            raceinfo = _func(raceinfo)
            paybacks = _func(paybacks)

        # データ管理用のインスタンス作成
        manager = DataManager(results, raceinfo, paybacks, horseresults, peds)

        return manager

    def __preprocess_results(self, filepath):
        """
        レース結果テーブルに対する前処理を行う

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        results : pd.DataFrame
            前処理後のレース結果テーブル
        """
        # データ読み込み
        df = self.__read_pickle(filepath, err_msg='preprocess_results')
        # columnのスペースを削除
        df = df.set_axis(df.columns.map(lambda x: x.replace(' ', '')), axis='columns', copy=False)
        # RANKが数字以外のものは削除・降着判定されたものも正しく扱う
        df = df[df[ResultsCols.RANK].str.contains('^\d+', na=False)]
        df[ResultsCols.RANK] = df[ResultsCols.RANK].str.replace('\D+', '', regex=True).map(int)
        # 抽出する列を指定
        same_cols = [
            ResultsCols.FRAME_NUMBER,
            ResultsCols.HORSE_NUMBER,
            ResultsCols.WEIGHT,
            'horse_id',
            'jockey_id',
        ]
        # 性齢を性別と年齢に分割
        sex_age_df = df[ResultsCols.SEX_AGE].str.extract('(?P<sex>\D+)(?P<age>\d+)')
        sex_age_df['age'] = sex_age_df['age'].map(int)
        # 体重を馬体重と体重変化に分割
        weight_diff_df = df[ResultsCols.HORSE_WEIGHT].str.extract(
            '(?P<weight>\d+)\((?P<diff>[-+\d]+)\)'
        )
        weight_diff_df = weight_diff_df.fillna(weight_diff_df.mode().iloc[0]).applymap(float)
        # データ結合
        results = pd.concat([df[same_cols].copy(), sex_age_df, weight_diff_df], axis=1)
        # 頭数の設定
        index_counts = results.index.value_counts(sort=False)
        results['n_horse'] = 0
        results.loc[index_counts.index, 'n_horse'] = index_counts
        # 着順処理
        results['rank'] = df[ResultsCols.RANK]

        return results

    def __preprocess_raceinfo(self, filepath):
        """
        レース情報テーブルに対する前処理を行う

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        results : pd.DataFrame
            前処理後のレース情報テーブル
        """
        df = self.__read_pickle(filepath, err_msg='preprocess_raceinfo')

        extractors = [
            # PLACEの抽出
            _Extractor(HorseResultsCols.PLACE, '|'.join(Master.PLACES.keys()).replace('(', '\(').replace(')', '\)')),
            # WEATHERの抽出
            _Extractor(HorseResultsCols.WEATHER, '|'.join(Master.WEATHER)),
            # GROUND_STATEの抽出
            _Extractor(HorseResultsCols.GROUND_STATE, '|'.join(Master.GROUND_STATES.values())),
            # dateの抽出
            _Extractor(HorseResultsCols.DATE, '\d+年\d+月\d+',
                       callback=lambda x: re.sub(r'年|月', '-', x)),
            # RACE_TYPEの抽出
            _Extractor('race_type', '|'.join(Master.RACE_TYPES.keys()),
                       callback=lambda x: Master.RACE_TYPES[x]),
            # AROUNDの抽出
            _Extractor('around', '|'.join(Master.AROUND)),
            # RACE_CLASSの抽出
            _Extractor('race_class', '|'.join(Master.RACE_CLASSES)),
            # course_lenの抽出
            _Extractor('course_len', '\d+m',
                       callback=lambda x: float(x[:-1]) / self.__course_len_scaler),
        ]
        texts = df['texts']
        results = pd.concat([instance.extract(texts) for instance in extractors], axis=1)
        results.fillna({'race_class': 'unknown'}, inplace=True)
        results[HorseResultsCols.DATE] = pd.to_datetime(results[HorseResultsCols.DATE], format=self.__date_format)

        return results

    def __preprocess_payback(self, filepath):
        """
        払い戻しテーブルに対する前処理を行う

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        results : pd.DataFrame
            前処理後の払い戻しテーブル
        """
        df = self.__read_pickle(filepath, err_msg='preprocess_payback')
        # column名変更
        df = df.set_axis(['type', 'matched', 'price', 'popular'], axis='columns', copy=False)
        # popularの列を削除
        df.drop(columns='popular', inplace=True)
        # ワイドと複勝を分割
        judge = df['type'].isin(['ワイド', '複勝'])
        filtered_df = df.loc[~judge, :].apply(lambda x: x.str.replace(' ', ''))
        matched = df.loc[judge, 'matched'].str.replace(r'\s-\s', '-', regex=True).str.split()
        price = df.loc[judge, ['type', 'price']].apply(lambda x: x.str.split() if x.name != 'type' else x)
        modified_df = pd.concat([matched.explode(), price.explode('price')], axis=1)
        # バラバラにしたデータを結合・整形
        results = pd.concat([filtered_df, modified_df]).sort_index().fillna(0)
        results['price'] = results['price'].map(lambda x: int(str(x).replace(',', '')))

        return results

    def __preprocess_horseresults(self, filepath):
        """
        馬の過去成績テーブルに対する前処理を行う

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        results : pd.DataFrame
            前処理後の馬の過去成績テーブル
        """
        df = self.__read_pickle(filepath, err_msg='preprocess_horseresults')
        # columnのスペースを削除
        df = df.set_axis(df.columns.map(lambda x: x.replace(' ', '')), axis='columns', copy=False)
        # RANKが数字以外のものは削除・降着判定されたものも正しく扱う
        df = df[df[HorseResultsCols.RANK].str.contains('^\d+', na=False)]
        df[HorseResultsCols.RANK] = df[HorseResultsCols.RANK].str.replace('\D+', '', regex=True).map(int)
        # HORSE_WEIGHTが不適切なものは削除
        df = df[df[HorseResultsCols.HORSE_WEIGHT].str.contains('^\d+', na=False)]
        # GROUND_STATEがNaNとなっているものを削除
        df = df.dropna(subset=[HorseResultsCols.GROUND_STATE], how='any', axis=0)
        # PRIZEの値の補完
        df.fillna({HorseResultsCols.PRIZE: 0.0}, inplace=True)
        # 実数を整数に変換
        targets = [
            HorseResultsCols.R,
            HorseResultsCols.N_HORSES,
            HorseResultsCols.FRAME_NUMBER,
        ]
        df = df[df[targets].isna().sum(axis=1).map(lambda x: not bool(x))]
        df[targets] = df[targets].applymap(int)
        # 日付のフォーマット変更
        df[HorseResultsCols.DATE] = pd.to_datetime(df[HorseResultsCols.DATE], format=self.__date_format)
        # GROUND_STATEのフォーマット変更
        df[HorseResultsCols.GROUND_STATE] = df[HorseResultsCols.GROUND_STATE].map(
            lambda x: Master.GROUND_STATES[x]
        )
        # PLACEのフォーマット変更
        df[HorseResultsCols.PLACE] = df[HorseResultsCols.PLACE].str.replace('\d', '', regex=True)
        # 抽出する列を指定
        same_cols = [
            HorseResultsCols.DATE,
            HorseResultsCols.PLACE,
            HorseResultsCols.WEATHER,
            HorseResultsCols.R,
            HorseResultsCols.N_HORSES,
            HorseResultsCols.FRAME_NUMBER,
            HorseResultsCols.HORSE_NUMBER,
            HorseResultsCols.WEIGHT,
            HorseResultsCols.GROUND_STATE,
            HorseResultsCols.RANK,
            HorseResultsCols.RANK_DIFF,
            HorseResultsCols.LAST_TIME,
            HorseResultsCols.PRIZE,
        ]
        # 体重を馬体重と体重変化に分割
        weight_diff_df = df[HorseResultsCols.HORSE_WEIGHT].str.extract(
            '(?P<weight>\d+)\((?P<diff>[-+\d]+)\)'
        )
        weight_diff_df = weight_diff_df.fillna(weight_diff_df.mode().iloc[0]).applymap(float)
        # 距離をレースタイプと全長に分割
        race_type_distance_df = df[HorseResultsCols.COURSE_LEN].str.extract(
            '(?P<race_type>\w)(?P<course_len>\d+)'
        )
        race_type_distance_df['race_type'] = race_type_distance_df['race_type'].map(
            lambda x: Master.RACE_TYPES[x]
        )
        # コーナー算出
        corners = df[HorseResultsCols.CORNER].str.split('-').fillna('0')
        several_corner_df = pd.DataFrame({
            'first_corner': corners.map(lambda xs: int(xs[0])),
            'final_corner': corners.map(lambda xs: int(xs[-1])),
            'mean_corner':  corners.map(lambda xs: np.mean(list(map(int, xs))) if isinstance(xs, list) else float(xs)),
        })
        # データ結合
        results = pd.concat([
            df[same_cols].copy(), weight_diff_df, race_type_distance_df, several_corner_df
        ], axis=1)
        results['course_len'] = results['course_len'].map(lambda x: float(x) / self.__course_len_scaler)

        return results

    def __preprocess_peds(self, filepath):
        """
        血統テーブルに対する前処理を行う

        Parameters
        ----------
        filepath : str
            ファイルパス

        Returns
        -------
        results : pd.DataFrame
            前処理後の血統テーブル
        """
        df = self.__read_pickle(filepath, err_msg='preprocess_peds')
        # 対象とする「n親等」までの情報を抽出
        judge = lambda xs: np.nan if (isinstance(xs, float) and np.isnan(xs)) or xs[1] > self.__max_relatives else xs[0]
        results = df.applymap(judge).dropna(axis=1)

        return results
