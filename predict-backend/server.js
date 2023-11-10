const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json({ limit: '50mb' }));

const API_GATEWAY_URL = 'https://4n5nhyipoc.execute-api.ap-northeast-1.amazonaws.com/prd/predict';

app.post('/api/upload', async (req, res) => {
    const base64Image = req.body.image;
    if (!base64Image) {
        return res.status(400).send({ message: 'No image data provided' });
    }

    const requestData = { image: base64Image };
    try {
        const lambdaResponse = await axios.post(API_GATEWAY_URL, requestData, {
            headers: { 'Content-Type': 'application/json' }
        });
        res.send({ prediction: lambdaResponse.data });
    } catch (error) {
        console.error('Error calling Lambda function:', error);
        res.status(500).send({ message: 'Error processing image' });
    }
});

const port = 8000;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});