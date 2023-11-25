const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const API_GATEWAY_ENDPOINT = 'https://gajt9atce1.execute-api.ap-northeast-1.amazonaws.com/prd/fashion'

app.post('/fashion', async (req, res) => {
    try {
        const response = await axios.post(API_GATEWAY_ENDPOINT, {
            body: req.body
        });

        res.json(response.data);
    } catch (error) {
        console.error('Error calling Lambda function:', error);
        res.status(500).send('Internal Server Error');
    }
});

const PORT = 11000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
