#!/bin/sh

# PHP-FPMをバックグラウンドで起動
php-fpm -D

# Nginxをフォアグラウンドで起動
exec nginx -g "daemon off;"