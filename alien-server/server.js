const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
const PORT = 3001;

app.use(express.json());
app.use(cors());

app.post('/ask-alien', async (req, res) => {
    try {
        const lambdaRequestBody = {
            body: JSON.stringify(req.body)
        };
        const response = await axios.post('https://ub64vu6qrd.execute-api.ap-northeast-1.amazonaws.com/prd/ask-alien', lambdaRequestBody);
        res.json(response.data);
    } catch (error) {
        console.error("Error when calling Lambda:", error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});