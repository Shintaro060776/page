<?php
require 'vendor/autoload.php';

use Aws\Ses\SesClient;

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    $name = $_POST["name"];
    $email = $_POST["email"];
    $content = $_POST["content"];

    $client = SesClient::factory(array(
        'version'=> 'latest',
        'region' => 'ap-northeast-1',
        'credentials' => [
            'key' => 'AKIA2NG4APPN2B5XFV3R',
            'secret' => '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',
        ]
    ));

    $result = $client->sendEmail([
        'Destination' => [
            'ToAddresses' => ['shintaro060776@gmail.com'],
        ],
        'Message' => [
            'Body' => [
                'Text' => [
                    'Data' => $content,
                    'Charset' => 'UTF-8',
                ],
            ],
            'Subject' => [
                'Data' => "お問い合わせ from " . $name,
                'Charset' => 'UTF-8',
            ],
        ],
        'Source' => $email,
    ]);

    if (!$result) {
        echo "メールの送信に失敗しました。";
        exit; 
    }

    header('Location: ./thankyou/thankyou.html');
}
?>