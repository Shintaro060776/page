const express = require('express');
const AWS = require('aws-sdk');
const axios = require('axios');
const bodyParser = require('body-parser');

const app = express();

app.use(bodyParser.json());

const apiGatewayUrl = 'https://b0w8833via.execute-api.ap-northeast-1.amazonaws.com/prd/predict';

const invokeLambda = async (data) => {
    try {
        const response = await axios.post(apiGatewayUrl, data, {
            headers: {
                'Content-Type': 'application/json',
            },
        });

        return response.data;
    } catch (error) {
        console.error('Error calling Lambda through API Gateway:', error);
        throw error;
    }
};

app.post('generate-lyrics', async (req, res) => {
    try {
        const requestData = req.body;
        const lambdaResponse = await invokeLambda(requestData);
        res.json(lambdaResponse);
    } catch (error) {
        res.status(500).send('Internal Server Error');
    }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

