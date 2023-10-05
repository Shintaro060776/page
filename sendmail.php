<?php

require 'vendor/autoload.php';

use Aws\Ses\SesClient;
use Aws\Exception\AwsException;

$SesClient = new SesClient([
    'version' => 'latest',
    'region'  => 'ap-northeast-1'
]);

$errors = [];

if ($_SERVER['REQUEST_METHOD'] == "POST") {
    // Check if POST data is set and not empty
    $name = !empty($_POST['name']) ? $_POST['name'] : null;
    $email = !empty($_POST['email']) && filter_var($_POST['email'], FILTER_VALIDATE_EMAIL) ? $_POST['email'] : null;
    $content = !empty($_POST['content']) ? $_POST['content'] : null;

    // Validation
    if (!$name) {
        $errors[] = "名前を入力してください。";
    }
    if (!$email) {
        $errors[] = "有効なメールアドレスを入力してください。";
    }
    if (!$content) {
        $errors[] = "お問い合わせ内容を入力してください。";
    }

    if (empty($errors)) {
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
                'Source' => 'shintaro060776@gmail.com',
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
    } else {
        foreach ($errors as $error) {
            echo $error . "<br>";
        }
    }
}
?>