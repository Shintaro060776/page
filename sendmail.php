<?php

require 'vendor/autoload.php';

use Aws\Ses\SesClient;
use Aws\Exception\AwsException;

$credentials = require 'credentials.php';

$SesClient = new SesClient([
    'version' => 'latest',
    'region'  => 'ap-northeast-1',
    'credentials' => $credentials
]);

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    $name = isset($_POST['name']) && !empty($_POST['name']) ? $_POST['name'] : 'No Name';
    $email = isset($_POST['email']) && filter_var($_POST['email'], FILTER_VALIDATE_EMAIL) ? $_POST['email'] : 'default_sender@example.com';
    $content = isset($_POST['content']) && !empty($_POST['content']) ? $_POST['content'] : 'No Content';

    $subject = "お問い合わせ from " . $name;
    $plaintext_body = $content;
    $html_body = "<p>{$content}</p>";
    $char_set = 'UTF-8';

    try {
        $result = $SesClient->sendEmail([
            'Destination' => [
                'ToAddresses' => ['shintaro060776@gmail.com'],
            ],
            'ReplyToAddresses' => [$email],
            'Source' => $email,
            'Message' => [
                'Body' => [
                    'Html' => [
                        'Charset' => $char_set,
                        'Data' => $html_body,
                    ],
                    'Text' => [
                        'Charset' => $char_set,
                        'Data' => $plaintext_body,
                    ],
                ],
                'Subject' => [
                    'Charset' => $char_set,
                    'Data' => $subject,
                ],
            ],
        ]);

        header('Location: ./thankyou/thankyou.html');
    } catch (AwsException $e) {
        echo $e->getMessage();
        echo("The email was not sent. Error message: ".$e->getAwsErrorMessage()."\n");
    }
}
?>