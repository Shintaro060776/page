#!/bin/bash

# ディレクトリを作成
WORK_DIR="/page/cattoon/lambda_package"
TARGET_DIR="${WORK_DIR}/python/lib/python3.8/site-packages/"

mkdir -p "${TARGET_DIR}"
cd "${WORK_DIR}"

# ライブラリをインストール

# レイヤー1: requests, urllib3
pip install requests --target "${TARGET_DIR}"

# レイヤー2: boto3, pyheif
# pip install boto3 pyheif Pillow --target "${TARGET_DIR}"

# レイヤー3: requests, urllib3
# pip install urllib3==1.25.4 cffi --target "${TARGET_DIR}"

# lambda_function.py をディレクトリに移動
cp /page/cattoon/lambda_function.py "${WORK_DIR}"

# 不要なディレクトリやファイルを削除
rm -rf venv bin certifi certifi-2023.7.22.dist-info charset_normalizer charset_normalizer-3.3.2.dist-info idna idna-3.4.dist-info requests-2.31.0.dist-info urllib3 urllib3-2.0.7.dist-info

# ライブラリを含むZIPパッケージを生成
zip -r9 /page/cattoon/function.zip .

# スクリプト終了
echo "Script completed!"