const express = require('express');
const axios = require('axios');
const multer = require('multer');
const upload = multer({ dest: 'upload/' });
const fs = require('fs');

const app = express();
const API_GATEWAY_URL = 'https://4n5nhyipoc.execute-api.ap-northeast-1.amazonaws.com/prd/predict';

app.post('/api/upload', upload.single('image'), async (req, res) => {
    const file = req.file;
    console.log("Received file:", file);

    const imageBuffer = fs.readFileSync(file.path);
    const base64Image = imageBuffer.toString('base64');
    console.log("Base64 Encoded Image:", base64Image);

    const requestData = {
        image: base64Image
    };
    console.log("Sending Request Data:", requestData);

    try {
        const lambdaResponse = await axios.post(API_GATEWAY_URL, requestData, {
            headers: { 'Content-Type': 'application/json' }
        });
        console.log("Received Response:", lambdaResponse.data);

        res.send({ prediction: lambdaResponse.data });
    } catch (error) {
        console.error('Error calling Lambda function:', error);
        res.status(500).send('Error processing image');
    }

    fs.unlinkSync(file.path);
});

const port = 8000;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});