const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

const API_GATEWAY_URL = 'https://dbm9ma8e8l.execute-api.ap-northeast-1.amazonaws.com/prd/natural';

app.post('/api/generate-joke', async (req, res) => {
    const prompt = req.body.prompt;

    if (!prompt) {
        return res.status(400).send({ error: 'No prompt provided' });
    }

    const requestData = { prompt: prompt };

    try {
        const lambdaResponse = await axios.post(API_GATEWAY_URL, requestData, {
            headers: { 'Content-Type': 'application/json' }
        });

        if (lambdaResponse.data && lambdaResponse.data.joke) {
            res.send({ joke: lambdaResponse.data.joke });
        } else {
            throw new Error('Incomplete response data from Lambda');
        }
    } catch (error) {
        console.error('Error calling API Gateway:', error.message);
        const status = error.response?.status || 500;
        const message = error.response?.data?.error || 'Error generating joke';
        res.status(status).send({ error: message });
    }
});

const port = process.env.PORT || 8000;
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});