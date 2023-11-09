const express = require('express');
const axios = require('axios');
const multer = require('multer');
const upload = multer({ dest: 'upload/' });
const fs = require('fs');

const app = express();
const API_GATEWAY_URL = 'https://4n5nhyipoc.execute-api.ap-northeast-1.amazonaws.com/prd/predict';

app.post('/api/upload', upload.single('image'), async (req, res) => {
    const file = req.file;
    const imageBuffer = fs.readFileSync(file.path);

    try {
        const lambdaResponse = await axios({
            method: 'post',
            url: API_GATEWAY_URL,
            data: {
                image: imageBuffer.toString('base64')
            },
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const predictResult = lambdaResponse.data;
        res.send({
            prediction: predictionResult
        });
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
