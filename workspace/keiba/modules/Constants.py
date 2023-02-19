from dataclasses import dataclass
from types import MappingProxyType
import os

# グローバル変数
_BASE_DIR = os.path.abspath('./')
_DATA_DIR = os.path.join(_BASE_DIR, 'data')
_RAW_DIR = os.path.join(_DATA_DIR, 'raw')
_HTML_DIR = os.path.join(_DATA_DIR, 'html')
_MASTER_DIR = os.path.join(_DATA_DIR, 'master')

@dataclass(frozen=True)
class LocalPaths:
    # レース結果
    RAW_RESULTS_PATH = os.path.join(_RAW_DIR, 'results.pkl')
    RAW_RACEINFO_PATH = os.path.join(_RAW_DIR, 'race_info.pkl')
    RAW_PAYBACK_PATH = os.path.join(_RAW_DIR, 'payback.pkl')
    # 馬の過去成績
    RAW_HORSERESULTS_PATH = os.path.join(_RAW_DIR, 'horse_results.pkl')
    # 血統情報
    RAW_PEDS_PATH = os.path.join(_RAW_DIR, 'ped_results.pkl')

@dataclass(frozen=True)
class SystemPaths:
    HTML_RACE_DIR = os.path.join(_HTML_DIR, 'race')
    HTML_HORSE_DIR = os.path.join(_HTML_DIR, 'horse')
    HTML_PED_DIR = os.path.join(_HTML_DIR, 'ped')
    HORSE_RESULTS_PATH = os.path.join(_MASTER_DIR, 'horse_results_updated_at.pkl')

@dataclass(frozen=True)
class ResultsCols:
    RANK = '着順'
    FRAME_NUMBER = '枠番'
    HORSE_NUMBER = '馬番'
    HORSE_NAME = '馬名'
    SEX_AGE = '性齢'
    WEIGHT = '斤量'
    JOCKEY = '騎手'
    TIME = 'タイム'
    ODDS = '単勝'
    POPULARITY = '人気'
    HORSE_WEIGHT = '馬体重'
    TRAINER = '調教師'

@dataclass(frozen=True)
class HorseResultsCols:
    DATE = '日付'
    PLACE = '開催'
    WEATHER = '天気'
    R = 'R'
    RACE_NAME = 'レース名'
    N_HORSES = '頭数'
    FRAME_NUMBER = '枠番'
    HORSE_NUMBER = '馬番'
    ODDS = 'オッズ'
    POPULARITY = '人気'
    RANK = '着順'
    JOCKEY = '騎手'
    WEIGHT = '斤量'
    COURSE_LEN = '距離'
    GROUND_STATE = '馬場'
    TIME = 'タイム'
    RANK_DIFF = '着差'
    CORNER = '通過'
    PACE = 'ペース'
    LAST_TIME = '上り'
    HORSE_WEIGHT = '馬体重'
    PRIZE = '賞金'

@dataclass(frozen=True)
class Master:
    PLACES = MappingProxyType({
        '札幌':'01',
        '函館':'02',
        '福島':'03',
        '新潟':'04',
        '東京':'05',
        '中山':'06',
        '中京':'07',
        '京都':'08',
        '阪神':'09',
        '小倉':'10',
        '門別':'30',
        '旭川':'34',
        '盛岡':'35',
        '水沢':'36',
        '浦和':'42',
        '船橋':'43',
        '大井':'44',
        '川崎':'45',
        '金沢':'46',
        '笠松':'47',
        '名古屋':'48',
        '園田':'50',
        '姫路':'51',
        '福山':'53',
        '高知':'54',
        '佐賀':'55',
        '荒尾':'56',
        '札幌(地)':'58',
    })
    RACE_TYPES = MappingProxyType({
        '芝': '芝',
        'ダ': 'ダート',
        '障': '障害',
    })
    WEATHER = ('晴', '曇', '小雨', '雨', '小雪', '雪')
    GROUND_STATES = MappingProxyType({
        '良': '良',
        '稍': '稍重',
        '重': '重',
        '不': '不良'
    })
    SEXES = ('牡', '牝', 'セ')
    AROUND = ('右', '左', '直線', '障害')
    RACE_CLASSES = ('新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス', 'オープン', '障害')

# グローバル変数
_DB_DOMAIN = 'https://db.netkeiba.com'
_TOP_URL = 'https://race.netkeiba.com/top'

@dataclass(frozen=True)
class UrlPaths:
    RACE_URL = f'{_DB_DOMAIN}/race'
    HORSE_URL = f'{_DB_DOMAIN}/horse'
    PED_URL = f'{_DB_DOMAIN}/horse/ped'
    CALENDAR_URL = f'{_TOP_URL}/calendar.html'
    RACE_LIST_URL = f'{_TOP_URL}/race_list_sub.html'
