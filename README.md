# 機械学習を用いた競馬予想
## JupyterLabの設定
### 準備
1. 下記のコマンドを実行する。

    ```bash
    docker-compose build
    ```

1. 下記のコマンドを実行し、`UID`と`GID`を調べる。

    ```bash
    id
    # 出力結果例
    # uid=1000(yuruto) gid=1000(yuruto)
    # -> 今回の場合、UID=1000, GID=1000
    ```

1. `docker-compose.yml`をエディタで開き、`PUID`と`PGID`を設定する。

    ```yml
    environment:
      - PUID=1000 # ここを調べたUIDの値に更新
      - PGID=1000 # ここを調べたGIDの値に更新
    ```

### 起動
1. 以下のコマンドを実行し、コンテナを起動する。

    ```bash
    docker-compose up -d
    ```

1. Webブラウザから以下のリンクにアクセスする。ここで、`server-ip-address`は、コンテナを起動したマシンのIPアドレスを入力すること。

    ```bash
    http://server-ip-address:18580/
    ```

## 処理
1. 上記の手順でJupyterLabのページにアクセスする。
1. `scrape.ipynb`を立ち上げ、`_RaceParams`、`ExecParams`の内容を自身の目的に合わせて変更する。
1. 「Run」→「Run All Cells」を押下し、スクレイピングを実施する。（暫くかかる）
1.  3.の処理終了後、`preprocess_learning.ipynb`を起動し、「Run」→「Run All Cells」を押下し、前処理・学習・評価を行う。
