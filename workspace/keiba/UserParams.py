from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    # 馬の過去成績を取得する際の対象
    TARGET_COLUMNS = ('LAST_TIME', 'RANK', 'RANK_DIFF', 'PRIZE', 'first_corner', 'final_corner', 'mean_corner')
    # 馬の過去成績を取得する際のグループ化して取得する対象
    GROUP_COLUMNS = ('race_type', 'course_len', 'PLACE')
    # カテゴリー変数化する対象
    CATEGORY_COLUMNS = ('PLACE', 'WEATHER', 'GROUND_STATE', 'sex', 'race_type', 'around', 'race_class')
