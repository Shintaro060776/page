<?php

require 'vendor/autoload.php';

use Aws\Ses\SesClient;
use Aws\Exception\AwsException;

$SesClient = new SesClient([
    'version' => 'latest',
    'region'  => 'ap-northeast-1'
]);

$errors = [];
$logFile = '/var/log/php-fpm/error.log'; // あなたのカスタムログファイルへのパスを設定してください

function logError($message) {
    global $logFile;
    file_put_contents($logFile, date('Y-m-d H:i:s') . " - " . $message . "\n", FILE_APPEND);
}

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    // (前述のコードを継続...)

    if (empty($errors)) {
        $subject = "お問い合わせ from " . $name;
        $plaintext_body = $content;
        $html_body = "<p>{$content}</p>";
        $char_set = 'UTF-8';

        $retryCount = 0;
        $maxRetries = 3;

        while ($retryCount < $maxRetries) {
            try {
                $result = $SesClient->sendEmail([
                    // (前述のコードを継続...)
                ]);

                header('Location: ./thankyou/thankyou.html');
                exit; // 成功したのでループを終了
            } catch (AwsException $e) {
                $retryCount++;
                if ($retryCount < $maxRetries) {
                    sleep(10);  // 10秒待って再試行
                    continue;
                }

                logError("Error sending email: " . $e->getAwsErrorMessage());
                logError("Request ID: " . $e->getAwsRequestId());
                logError("Error Code: " . $e->getAwsErrorCode());

                echo "申し訳ございません、現在メールの送信ができません。後ほど再試行してください。";
            }
        }
    } else {
        foreach ($errors as $error) {
            echo $error . "<br>";
        }
    }
}
?>