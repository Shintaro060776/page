const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();
const PORT = 3001;

app.use(express.json());
app.use(cors());

app.post('/ask-alien', async (req, res) => {
    try {
        console.log("Received request body:", req.body);

        const response = await axios.post('https://fb22ga018h.execute-api.ap-northeast-1.amazonaws.com/prd/ask-alien', req.body);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error' });
    }
});

app.listen(PORT, () => {
    console.log(`Server started on port ${PORT}`);
});