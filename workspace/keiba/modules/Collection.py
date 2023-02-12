import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from monthdelta import monthmod
import time
import requests
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm
import re
import os
import sys
from .Constants import UrlPaths, SystemPaths

class Collection:
    """
    Collection : データ収集用クラス

    Attributes
    ----------
    __class_name : str
        クラス名
    __wait_time : float
        待機時間
    """
    def __init__(self, wait_time=1.2):
        """
        初期化処理
        """
        # クラスメソッド用の変数
        self.__class_name = self.__class__.__name__
        self.__wait_time = wait_time
        np.random.seed(int(time.time()))

    def get_event_date(self, from_, to_):
        """
        開催日の一覧を取得

        Parameters
        ----------
        from_ : str
            取得開始年月(format: yyyy-mm)
        to_ : str
            取得終了年月(format: yyyy-mm)

        Returns
        -------
        race_dates : list
            レース日一覧
        """
        start_date = datetime.strptime(from_, '%Y-%m')
        end_date = datetime.strptime(to_, '%Y-%m')

        # 探索範囲が不適切な場合
        if end_date < start_date:
            raise Exception(f'Error({self.__class_name}::get_event_date): Invalid argument')

        # 年月の差分取得
        diff, _ = monthmod(start_date, end_date)
        race_dates = []

        for idx in tqdm(range(diff.months + 1)):
            # 取得年月の計算
            target = start_date + relativedelta(months=idx)
            year = target.year
            month = str(target.month).zfill(2)
            # 取得URLの設定&データ取得
            url = f'{UrlPaths.CALENDAR_URL}?year={year}&month={month}'
            df = pd.read_html(url)[0]
            # 日付一覧を取得
            days = [str(val) for val in sum(df.values.tolist(), [])]
            # レースがある日のみを取得し、年月日を計算
            events = [f'{year}{month}{val.split()[0].zfill(2)}' for val in days if len(val.split()) > 1]
            race_dates += events
            time.sleep(self.__wait_time + np.random.rand())

        return race_dates

    def get_race_ids(self, race_dates):
        """
        レースID一覧を取得

        Parameters
        ----------
        race_dates : list
            レース日一覧

        Returns
        -------
        race_ids : list
            レースID一覧
        """
        race_ids = []

        for date in tqdm(race_dates):
            url = f'{UrlPaths.RACE_LIST_URL}?kaisai_date={date}'
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            # 該当レース一覧を取得
            targets = soup.find_all('dd', attrs={'class': 'RaceList_Data'})
            for target in targets:
                items = target.find_all('span', attrs={'class': 'MyRace_List_Item'})
                out = [re.sub(r'\D', '', val.get('id')) for val in items]
                race_ids += out
            time.sleep(self.__wait_time + np.random.rand())

        return race_ids

    def __scrape_html(self, ids, base_url, output_dir, isSkip=True):
        """
        Webからデータを取得

        Parameters
        ----------
        ids : list
            取得対象のID
        base_url : str
            取得先のURL
            利用時のフォーマット：f'{base_url}/{target_id}'
        output_dir : str
            出力ディレクトリ
            利用時のフォーマット：os.path.join(output_dir, f'{target_id}.bin')
        isSkip : boolean
            既にファイルが存在する場合の対応
            True:  読み飛ばす
            False: Webから再取得する

        Returns
        -------
        html_filepaths : list
            htmlファイルパス
        """
        html_filepaths = []

        for target_id in tqdm(ids):
            html_filename = os.path.join(output_dir, f'{target_id}.bin')
            # ファイルが存在するかつ、スキップする場合
            if os.path.exists(html_filename) and isSkip:
                continue
            else:
                # ===================
                # Webからデータを取得
                # ===================
                time.sleep(self.__wait_time + np.random.rand())
                # スクレイピング実行
                url = f'{base_url}/{target_id}'
                response = requests.get(url)
                # バイナリデータをhtmlファイルに保存
                with open(html_filename, 'wb') as fout:
                    fout.write(response.content)
                html_filepaths += [html_filename]

        return html_filepaths

    def scrape_html_race(self, race_ids, isSkip=True):
        """
        レース結果のhtmlファイルを取得

        Parameters
        ----------
        race_ids : list
            レースID一覧

        Returns
        -------
        html_filepaths : list
            レース結果のhtmlファイルパス
        """
        html_filepaths = self.__scrape_html(
            race_ids, UrlPaths.RACE_URL, SystemPaths.HTML_RACE_DIR, isSkip=isSkip
        )

        return html_filepaths

    def scrape_html_horse(self, horse_ids, isSkip=True):
        """
        馬情報のhtmlファイルを取得

        Parameters
        ----------
        horse_ids : list
            馬ID一覧

        Returns
        -------
        html_filepaths : list
            馬情報のhtmlファイルパス
        """
        html_filepaths = self.__scrape_html(
            horse_ids, UrlPaths.HORSE_URL, SystemPaths.HTML_HORSE_DIR, isSkip=isSkip
        )

        return html_filepaths

    def scrape_html_ped(self, horse_ids, isSkip=True):
        """
        血統情報のhtmlファイルを取得

        Parameters
        ----------
        horse_ids : list
            馬ID一覧

        Returns
        -------
        html_filepaths : list
            血統情報のhtmlファイルパス
        """
        html_filepaths = self.__scrape_html(
            horse_ids, UrlPaths.PED_URL, SystemPaths.HTML_PED_DIR, isSkip=isSkip
        )

        return html_filepaths

    def scrape_html_horse_with_master(self, horse_ids, isSkip=True):
        """
        馬情報のhtmlファイルを取得・マスターファイルを更新

        Parameters
        ----------
        horse_ids : list
            馬ID一覧

        Returns
        -------
        html_filepaths : list
            馬情報のhtmlファイルパス
        """
        html_filepaths = self.scrape_html_horse(horse_ids, isSkip=isSkip)
        horse_ids = [os.path.splitext(os.path.basename(html_filename))[0] for html_filename in html_filepaths]
        # horse idのDataFrame作成
        horse_df = pd.DataFrame({'horse_id': horse_ids})
        # 現在時刻を取得
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename_master = SystemPaths.HORSE_RESULTS_PATH

        # ファイルが存在しない場合は作成
        if not os.path.exists(filename_master):
            pd.DataFrame(columns=['horse_id', 'updated_at']).to_pickle(filename_master)
        # マスターファイルの読み込み
        master = pd.read_pickle(filename_master)
        # マージ
        df = master.merge(horse_df, on='horse_id', how='outer')
        # 更新時刻を現在の時刻にアップデート
        df.loc[df['horse_id'].isin(horse_ids), 'updated_at'] = now
        # 結果を保存
        df.to_pickle(filename_master)

        return html_filepaths

    def __read_html_file(self, html_filename):
        """
        htmlファイルの読み込み

        Parameters
        ----------
        html_filename : str
            htmlファイル名

        Returns
        -------
        html : binary
            htmlファイル（バイナリ形式）
        target_id : str
            対象のID
        """
        if os.path.exists(html_filename):
            # データの読み込み
            with open(html_filename, 'rb') as fin:
                html = fin.read()
            target_id = os.path.splitext(os.path.basename(html_filename))[0]
        else:
            html, target_id = None, None

        return html, target_id

    def get_rawdata_results(self, html_filepaths):
        """
        レース結果テーブルの取得

        Parameters
        ----------
        html_filepaths : list
            レース結果のhtmlファイルパス

        Returns
        -------
        race_results : pandas.DataFrame
            全レース結果テーブル
        """
        horse_pattern = re.compile('^/horse')
        jockey_pattern = re.compile('^/jockey')
        races = {}

        for html_filename in tqdm(html_filepaths):
            html, race_id = self.__read_html_file(html_filename)
            # 無効なファイルパスは読み飛ばす
            if html is None:
                continue

            try:
                df = pd.read_html(html)[0]
                soup = BeautifulSoup(html, 'html.parser')
                # 馬IDと騎手IDを取得
                summary_table = soup.find('table', attrs={'class': 'race_table_01'})
                # 馬IDを取得
                atags = summary_table.find_all('a', attrs={'href': horse_pattern})
                horse_ids = [re.findall(r'\d+', atag['href'])[0] for atag in atags]
                # 騎手IDを取得
                atags = summary_table.find_all('a', attrs={'href': jockey_pattern})
                jockey_ids = [re.findall(r'\d+', atag['href'])[0] for atag in atags]
                df['horse_id'] = horse_ids
                df['jockey_id'] = jockey_ids
                df['race_id'] = race_id
                races[race_id] = df.set_index('race_id')
            # IndexError, AttributeErrorは読み飛ばす
            except (IndexError, AttributeError):
                continue
            # 接続切れ等のエラー処理
            except Exception as e:
                _, _, tb = sys.exc_info()
                print(f'Error({self.__class_name}::get_rawdata_results, {tb.tb_lineno})[{race_id}] {e}')
                break
            # Jupyterのエラー処理
            except:
                break
        # 結果集計
        race_results = pd.concat([val for val in races.values()])

        return race_results

    def get_rawdata_raceinfo(self, html_filepaths):
        """
        レース情報テーブルの取得

        Parameters
        ----------
        html_filepaths : list
            レース結果のhtmlファイルパス

        Returns
        -------
        all_race_info : pandas.DataFrame
            全レース情報テーブル
        """
        info = {}

        for html_filename in tqdm(html_filepaths):
            html, race_id = self.__read_html_file(html_filename)
            # 無効なファイルパスは読み飛ばす
            if html is None:
                continue

            try:
                soup = BeautifulSoup(html, 'html.parser')
                # 天候、レースの種類、コースの長さ、馬場の状態、日付などを取得
                texts = re.findall(f'\w+', ''.join([
                    item.text
                    for item in soup.find('div', attrs={'class': 'data_intro'}).find_all('p')[:2]
                ]))
                # DataFrameの生成
                data = {
                    'texts':   ['-'.join(texts)],
                    'race_id': [race_id],
                }
                df = pd.DataFrame(data)
                info[race_id] = df.set_index('race_id')
            # AttributeErrorは読み飛ばす
            except AttributeError:
                continue
            # 接続切れ等のエラー処理
            except Exception as e:
                _, _, tb = sys.exc_info()
                print(f'Error({self.__class_name}::get_rawdata_info, {tb.tb_lineno})[{race_id}] {e}')
                break
            # Jupyterのエラー処理
            except:
                break

        # 結果集計
        all_race_info = pd.concat([val for val in info.values()])

        return all_race_info

    def get_rawdata_payback(self, html_filepaths):
        """
        払い戻し結果テーブルの取得

        Parameters
        ----------
        html_filepaths : list
            レース結果のhtmlファイルパス

        Returns
        -------
        paybacks : pandas.DataFrame
            すべての払い戻し結果のテーブル
        """
        payouts = {}

        for html_filename in tqdm(html_filepaths):
            html, race_id = self.__read_html_file(html_filename)
            # 無効なファイルパスは読み飛ばす
            if html is None:
                continue

            try:
                dfs = pd.read_html(html)
                df = pd.concat([dfs[1], dfs[2]])
                df['race_id'] = race_id
                payouts[race_id] = df.set_index('race_id')
            # IndexError, AttributeErrorは読み飛ばす
            except (IndexError, AttributeError):
                continue
            # 接続切れ等のエラー処理
            except Exception as e:
                _, _, tb = sys.exc_info()
                print(f'Error({self.__class_name}::get_rawdata_payback, {tb.tb_lineno})[{race_id}] {e}')
                break
            # Jupyterのエラー処理
            except:
                break

        # 結果集計
        paybacks = pd.concat([val for val in payouts.values()])

        return paybacks

    def get_rawdata_horse(self, html_filepaths):
        """
        過去成績テーブルの取得

        Parameters
        ----------
        html_filepaths : list
            馬の過去成績データのhtmlファイルパス

        Returns
        -------
        horse_results : pandas.DataFrame
            すべての馬の過去成績テーブル
        """
        horses = {}

        for html_filename in tqdm(html_filepaths):
            html, horse_id = self.__read_html_file(html_filename)
            # 無効なファイルパスは読み飛ばす
            if html is None:
                continue

            try:
                dfs = pd.read_html(html)
                df = dfs[4] if dfs[3].columns[0] == '受賞歴' else dfs[3]
                df['horse_id'] = horse_id
                horses[horse_id] = df.set_index('horse_id')
            # IndexError, AttributeErrorは読み飛ばす
            except (IndexError, AttributeError):
                continue
            # 接続切れ等のエラー処理
            except Exception as e:
                _, _, tb = sys.exc_info()
                print(f'Error({self.__class_name}::get_rawdata_horse, {tb.tb_lineno})[{horse_id}] {e}')
                break
            # Jupyterのエラー処理
            except:
                break

        # 結果集計
        horse_results = pd.concat([val for val in horses.values()])

        return horse_results

    def get_rawdata_ped(self, html_filepaths):
        """
        血統情報テーブルの取得

        Parameters
        ----------
        html_filepaths : list
            馬の過去成績データのhtmlファイルパス

        Returns
        -------
        ped_results : pandas.DataFrame
            すべての血統情報テーブル
        """
        peds = {}
        sex_pattern = re.compile(r'b_ml|b_fml')
        horse_id_pattern = re.compile(r'^/horse/[0-9a-z]+/$')
        relatives = {
            '16': 1, # 両親（1親等）
            '8':  2, # 祖父母（2親等）
            '4':  3, # 曾祖父母（3親等）
            '2':  4, # 高祖父母（4親等）
            '1':  5, # 5世の祖（5親等）
        }

        for html_filename in tqdm(html_filepaths):
            html, horse_id = self.__read_html_file(html_filename)
            # 無効なファイルパスは読み飛ばす
            if html is None:
                continue

            try:
                soup = BeautifulSoup(html, 'html.parser')
                ped_table = soup.find('table', attrs={'class': 'blood_table'})
                tds = ped_table.find_all('td', attrs={'class': sex_pattern})
                atags = [td.find('a', attrs={'href': horse_id_pattern}) for td in tds]
                ped_horse_ids = [
                    (re.sub(r'^/horse/', '', atag['href'])[:-1], relatives[atag.parent.get('rowspan', '1')])
                    for atag in atags if hasattr(atag, 'href')
                ]
                peds[horse_id] = pd.DataFrame({f'{horse_id}': ped_horse_ids})
            # IndexError, AttributeErrorは読み飛ばす
            except (IndexError, AttributeError):
                continue
            # 接続切れ等のエラー処理
            except Exception as e:
                _, _, tb = sys.exc_info()
                print(f'Error({self.__class_name}::get_rawdata_horse, {tb.tb_lineno})[{horse_id}] {e}')
                break
            # Jupyterのエラー処理
            except:
                break

        # 結果集計
        ped_results = pd.concat([val for val in peds.values()], axis=1).T.add_prefix('peds_').rename_axis('horse_id')

        return ped_results


    def update_rawdata(self, filepath, new_df):
        """
        テーブルの更新

        Parameters
        ----------
        filepath : str
            テーブルの保存先
        new_df : pd.DataFrame
            保存するテーブル情報
        """
        # ファイルが存在する場合
        if os.path.exists(filepath):
            old_df = pd.read_pickle(filepath)
            # 重複を削除
            filtered_old = old_df[~old_df.index.isin(new_df.index)]
            df = pd.concat([filtered_old, new_df])
        else:
            df = new_df.copy()
        # 更新結果を保存
        df.to_pickle(filepath)

    def load_rawdata(self, filepath):
        """
        テーブルの読み込み

        Parameters
        ----------
        filepath : str
            テーブルの読み込み先
        df : pd.DataFrame
            読み込んだテーブル情報
        """
        # ファイルが存在しない場合
        if not os.path.exists(filepath):
            raise Exception(f'Error({self.__class_name}::load_rawdata): Does not exist {filepath}')
        df = pd.read_pickle(filepath)

        return df
