const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
const PORT = 3001;

app.use(express.json());
app.use(cors());

app.post('/askalien', async (req, res) => {
    try {
        const lambdaRequestBody = {
            body: JSON.stringify(req.body)
        };
        const response = await axios.post('https://zqunvv49hc.execute-api.ap-northeast-1.amazonaws.com/$default/askalien', lambdaRequestBody);
        res.json(response.data);
    } catch (error) {
        console.error("Error when calling Lambda:", error.message);
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});