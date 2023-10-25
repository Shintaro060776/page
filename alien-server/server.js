const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
const PORT = 3002;

app.use(express.json());
app.use(cors());

app.post('/ask-alien', async (req, res) => {
    try {
        const response = await axios.post('https://r67d0rhvj9.execute-api.ap-northeast-1.amazonaws.com/prd/ask-alien', req.body);
        // const response = await axios.post('https://r67d0rhvj9.execute-api.ap-northeast-1.amazonaws.com/prd/ask-alien', lambdaRequestBody);

        if (response.data && response.data.answer) {
            res.json({ answer: response.data.answer });
        } else if (response.data && response.data.error) {
            console.error("Error from Lambda:", response.data.error);
            res.status(400).json({ error: 'Error from Lambda function' });
        } else {
            console.error("Unexpected response from Lambda:", response.data);
            res.status(500).json({ error: 'Internal server error' });
        }
    } catch (error) {
        console.error("Error when calling Lambda:", error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});