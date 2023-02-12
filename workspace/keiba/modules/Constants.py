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
