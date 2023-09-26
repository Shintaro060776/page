<?php
require 'vendor/autoload.php';

use Aws\Ses\SesClient;

$credentials = require 'credentials.php';

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    $name = isset($_POST['name']) ? $_POST['name'] : '';
    $email = isset($_POST['email']) ? $_POST['email'] : '';
    $content = isset($_POST['content']) ? $_POST['content'] : '';

    $client = SesClient::factory(array(
        'version'=> 'latest',
        'region' => 'ap-northeast-1',
        'credentials' => $credentials
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