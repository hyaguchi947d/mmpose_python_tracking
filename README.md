# mmposeを用いた人物姿勢推定

- 著者: 矢口 裕明 (クシナダ機巧株式会社)
- Author: Hiroaki Yaguchi (947D-tech)

## 本ソフトウェアについて

mmposeを用いて人物姿勢推定を行います。

https://github.com/open-mmlab/mmpose

Apache2.0ライセンスです。
一部にmmposeのサンプルコードを利用しています。

## 動作確認環境

- OS: Ubuntu 24.04
- Webカメラ: Logicool C920

## 使用方法

### インストール

mmposeの依存関係が複雑すぎるためdocker上に環境を構築します。

#### RTX30XX/40XX用

```bash
$ docker compose --profile stable build --build-arg BASE_IMAGE=pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
```

#### RTX50XX用

```bash
$ docker compose --profile latest build
```

#### CPUのみ用

TODO:未テストです

```bash
$ docker compose --profile cpu build
```

### 実行

セットアップした種類によって一部読み替えが必要となります。
stableを前提として説明します。

#### 共通項目

このディレクトリを/appとしてマウントします。
インファレンスを行うための画像や動画はこのディレクトリの下に置くか、
docker-compose.ymlを書き換えてマウントするようにしてください。

#### スクリプトの実行

```bash
$ docker compose --profile stable run --rm mmpose-stable python3 /app/mmpose_result_sample.py <入力画像>
```

#### シェルの起動

```bash
$ docker compose --profile stable run --rm mmpose-stable bash
```


## スクリプトについて

mmpose_result_sample.pyは単一画像に対してinferenceを行います。
引数として画像ファイル名を与えます。

mmpose_tracking.pyはビデオデバイスに対して連続でinferenceを行います。
以下のオプションが利用できます。

- `-i|--input`: 入力デバイスID、default=0
- `-d|--device`: 入力デバイスファイル、inputとdeviceはdeviceが優先されます。
- `-r|--rate`: フレームレート、default=30
- `--width`: 横解像度、default=1920
- `--height`: 縦解像度、default=1080
- `--codec`: コーデックを4文字で、default=MJPG
