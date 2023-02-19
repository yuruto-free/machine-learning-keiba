import numpy as np
import pandas as pd
import re
from dataclasses import dataclass
from typing import Any
from .Constants import HorseResultsCols, ResultsCols
from .FeatureEngineering import FeatureEngineering

@dataclass
class _ValidColumns:
    """
    DataManagerが持っている各データcolumn一覧
    """
    results: pd.DataFrame
    raceinfo: pd.DataFrame
    horseresults: pd.DataFrame

@dataclass
class _MergedData:
    """
    結合後のデータとcolumn一覧
    """
    df: pd.DataFrame
    table: pd.DataFrame

class DataManager:
    """
    DataManager : データ管理用クラス

    Attributes
    ----------
    __results : pd.DataFrame
        レース結果テーブル
    __raceinfo : pd.DataFrame
        レース情報テーブル
    __paybacks : pd.DataFrame
        払い戻し結果テーブル
    __horseresults : pd.DataFrame
        過去成績テーブル
    __peds : pd.DataFrame
        血統情報テーブル
    """
    def __init__(self, results, raceinfo, paybacks, horseresults, peds):
        """
        初期化処理
        """
        self.__results = results
        self.__raceinfo = raceinfo
        self.__paybacks = paybacks
        self.__horseresults = horseresults
        self.__peds = peds

    def __create_column_df(self, _cls, columns, description):
        """
        定数定義されたcolumn名とPythonでの変数定義のペアを生成する

        Parameters
        ----------
        _cls : object
            集計対象のクラス
        columns : list-like object
            該当するcolumn
        description : str
            概要説明
        """
        pattern = re.compile(r'^__')
        pairs = dict(map(lambda key: (_cls.__dict__[key], key),
                         filter(lambda val: not bool(pattern.match(val)), dir(_cls))))
        data = {'description': description}
        data.update({pairs.get(name, name): [name] for name in columns})
        df = pd.DataFrame(data)

        return df

    def get_columns(self):
        """
        有効なcolumn一覧の取得

        Returns
        -------
        instance : _ValidColumns
            _ValidColumnsのインスタンス
        """
        columns = {
            'results': self.__create_column_df(ResultsCols, self.__results.columns, 'レース結果テーブル'),
            'raceinfo': self.__create_column_df(HorseResultsCols, self.__raceinfo.columns, 'レース情報テーブル'),
            'horseresults': self.__create_column_df(HorseResultsCols, self.__horseresults.columns, '馬の過去成績テーブル'),
        }
        instance = _ValidColumns(**columns)

        return instance

    @property
    def paybacks(self):
        return self.__paybacks

    def __custom_aggregation(self, _df, target_columns, group_columns):
        """
        対象のhorse_idとdateに対する集計処理

        Parameters
        ----------
        _df : pd.Series
            集計時の評価データが含まれるDataFrame
        target_columns : list
            結合時のcolumn
        group_columns : list
            グループ化して集計する際に対象とするcolumn

        Returns
        -------
        results : pd.DataFrame
            集計結果のDataFrame
        """
        def _calc_mean_mode(base_series, target_df, columns):
            """
            平均値と最頻値を算出

            Parameters
            ----------
            base_series : pd.Series or None
                集計対象のSeries
            target_df : pd.DataFrame
                集計対象のDataFrame
            columns : list
                集計対象の列名のリスト

            Returns
            -------
            output : pd.Series
                出力結果
            """
            @dataclass
            class _CondParams:
                name: str
                condition: Any

            if base_series is None:
                # 集計時の制約条件なしの場合
                _params = _CondParams(
                    name='',
                    condition=map(bool, np.ones(len(target_df), dtype=np.int64)),
                )
            else:
                # 集計時の制約条件ありの場合
                _params = _CondParams(
                    name=base_series.name,
                    condition=target_df[base_series.name].isin(base_series.values),
                )
            # 集計方法の決定（True: 何もしない, False: mean）
            # pd.to_numericで無効値がない（すべて数値である）場合、平均値を算出
            agg_table = target_df[columns].apply(
                lambda _series: pd.to_numeric(_series, errors='coerce').isna().sum() > 0
            ).to_dict()
            # 集計対象の分類
            mean_info = {key: f'{_params.name}{key}_mean' for key, value in agg_table.items() if not value}
            mean_cols = mean_info.keys()
            # 集計処理
            ret_mean = target_df.loc[_params.condition, mean_cols].agg('mean').rename(mean_info)
            # 出力結果の生成
            output = ret_mean.dropna()

            if output.empty:
                output = pd.Series(dtype=object)

            return output

        # 集計対象の情報を取得
        name = HorseResultsCols.DATE
        date = _df[name].values[0]
        horse_id = _df['horse_id'].values
        # 過去成績テーブルのフィルタリング
        target_df = self.__horseresults[
            self.__horseresults.index.isin(horse_id) & (self.__horseresults[name] < date)
        ]
        # ========
        # 集計処理
        # ========
        if target_df.empty:
            results = pd.DataFrame({'dummy': [0]})
        else:
            outputs = {'dummy': 0}
            _kwargs = {'target_df': target_df, 'columns': target_columns}
            # 制約条件なしの場合の処理
            outputs.update(_calc_mean_mode(None, **_kwargs).to_dict())
            # group_columnsの各要素に対する処理
            _agg_series = _df[group_columns].agg(lambda _series: _calc_mean_mode(_series, **_kwargs))
            outputs.update(_agg_series.stack().reset_index(level=1, drop=True).to_dict())
            # 結果の集計
            results = pd.DataFrame({str(key): [value] for key, value in outputs.items()})

        return results.set_index('dummy')

    def merge(self, target_columns, group_columns):
        """
        データ結合

        Parameters
        ----------
        target_columns : list
            結合時のcolumn
        group_columns : list
            グループ化して集計する際に対象とするcolumn

        Returns
        -------
        result : _MergedData
            _MergedDataのインスタンス
                df: 結合後のデータ
                table: 列名とPythonコードの変数名との対応関係
        """
        # レース結果テーブルとレース情報テーブルの結合
        df = self.__results.merge(self.__raceinfo, left_index=True, right_index=True).reset_index()
        # 馬IDを基準に血統情報テーブルを結合（欠損値は最頻値で置換）
        df = df.merge(self.__peds.reset_index(drop=False), on='horse_id').fillna('mode')
        # 集計対象に存在する馬IDのみ抽出
        valid_horse_ids = self.__horseresults.index.unique()
        filtered_df = df[df['horse_id'].isin(valid_horse_ids)]
        # 集計
        _groupby_target = ['horse_id', HorseResultsCols.DATE]
        _extraction_cols = _groupby_target + group_columns
        aggregated_df = filtered_df.groupby(by=_groupby_target, group_keys=True)[_extraction_cols].apply(
            self.__custom_aggregation, target_columns=target_columns, group_columns=group_columns
        ).reset_index().drop(columns=['dummy'])
        # 集計結果を結合
        merged_df = df.merge(aggregated_df, on=_groupby_target, how='outer')
        filled_agg_df = pd.concat([
            merged_df[df.columns],
            merged_df.loc[:, merged_df.columns.str.match('.*mean$')].fillna(0),
        ], axis=1).set_index('race_id')
        # インスタンス生成
        result = _MergedData(**{
            'df': filled_agg_df,
            'table': self.__create_column_df(HorseResultsCols, filled_agg_df.columns, '結合後テーブル'),
        })

        return result

    def get_features(self, df, functions=None):
        """
        特徴量の取得

        Parameters
        ----------
        df : pd.DataFrame
            処理対象のDataFrame
        functions : dict or None
            呼び出す関数一覧
                key:   関数名（str）
                value: 関数の引数（dict）

        Returns
        -------
        instance : FeatureEngineering
            FeatureEngineeringのインスタンス
        """
        if functions is None:
            functions = {}

        # 初期化
        instance = FeatureEngineering(df)
        # 指定された関数を順に呼び出す
        for name, kwargs in functions.items():
            _func = getattr(instance, name, None)
            # 関数が存在するかつ呼び出し可能
            if hasattr(instance, name) and callable(_func):
                _func(**kwargs)

        return instance
