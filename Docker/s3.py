
import boto3
import re

# S3のバケット名を取得
bucket = "vhrthrtyergtcere"

# S3のファイル一覧を取得
s3_client = boto3.client('s3')
objects = s3_client.list_objects(Bucket=bucket)

# 'music_xx(日付).wav'というパターンに一致するファイルをフィルタリング
music_files = [obj['Key'] for obj in objects['Contents']
               if re.match(r'music_\d{8,14}\.wav', obj['Key'])]

if not music_files:
    print("No music files found in the S3 bucket.")
else:
    # 最新のファイルを取得（ここではファイル名の降順で最初のファイルを取得しています）
    latest_file = sorted(music_files, reverse=True)[0]
    # 最新のファイルを読み込む
    obj = s3_client.get_object(Bucket=bucket, Key=latest_file)
