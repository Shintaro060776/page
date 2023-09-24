<?php
    if($_SERVER['REQUEST_METHOD'] == "POST") {
        $name = $_POST["name"];
        $email = $_POST["email"];
        $content = $_POST["content"];

        $to = "shintaro060776@gmail.com";
        $subject = "お問い合わせ from" . $name;
        $headers = "From:" . $email;

        // mail関数の呼び出しとエラーチェック
        // if(!mail($to, $subject, $content, $headers)) {
        //     echo "メールの送信に失敗しました。";
        //     exit; // ここでスクリプトの実行を停止
        // }

        header('Location: ./thankyou/thankyou.html');
    }
?>