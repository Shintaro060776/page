<?php
    if($_SERVER['REQUEST_METHOD'] == "POST") {
        $name = $_POST["name"];
        $email = $_POST["email"];
        $content = $_POST["content"];

        $to = "shintaro060776@gmail.com";
        $subject = "お問い合わせ from" . $name;
        $headers = "From:" . $email;

        mail($to, $subject, $message. $headers);
        header('Location: ./thankyou/thankyou.html');
    }
?>