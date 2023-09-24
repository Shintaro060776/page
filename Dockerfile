FROM php:7.4-fpm-alpine

# Nginxをインストール
RUN apk --no-cache add nginx

# 不要なNginx設定を削除
RUN rm /etc/nginx/http.d/default.conf

# ファイルをコピー
COPY . /var/www/html/

# Nginx設定をコピー
COPY default.conf /etc/nginx/http.d/default.conf

# NginxとPHP-FPMを起動するスクリプト
COPY start.sh /start.sh
CMD ["/start.sh"]