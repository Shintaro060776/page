const express = require("express");
const AWS = require("aws-sdk");
const WavEncoder = require("wav-encoder");
const { Synth, Transport, Pattern, start, Offline } = require("tone");
const { AudioContext } = require("web-audio-api");
const toWav = require("audiobuffer-to-wav");
const cors = require("cors");
const fs = require('fs');
const bodyParser = require('body-parser');

const app = express();
const port = 7000;

const S3_OUTPUT_BUCKET = 'vhrthrtyergtcere';

let soundsData = [];

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

AWS.config.update({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.region
});

const LOG_FILE = '/home/ec2-user/.pm2/logs/sagemaker-error.log';

function logErrorToFile(error) {
    const errorMessage = `${new Date().toISOString()} - ${error}\n`;
    fs.appendFileSync(LOG_FILE, errorMessage);
}

app.post("/aisound", async (req, res) => {
    try {
        const musicData = await generateMusicFromData(req.body.data);
        const uploadedURL = await uploadMusicToS3(musicData);
        res.json({ url: uploadedURL });
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});

async function generateMusicFromData(data) {
    const notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4'];
    const synth = new Synth();

    const patterns = data.map(item => {
        const noteIndex = item.value % notes.length;
        return { note: notes[noteIndex], duration: item.duration };
    });

    const pattern = new Pattern((time, noteData) => {
        synth.triggerAttackRelease(noteData.note, noteData.duration, time);
    }, patterns);

    await start();
    const recorded = await Offline(() => {
        pattern.start(0);
        Transport.start();
    }, patterns.length * 0.5);

    const audioData = {
        sampleRate: recorded.sampleRate,
        channelData: [recorded.getChannelData(0)]
    };

    const audioBuffer = new AudioContext().createBuffer(1, audioData.channelData[0].length, audioData.sampleRate);
    audioBuffer.copyToChannel(audioData.channelData[0], 0);

    return toWav(audioBuffer);
}

async function uploadMusicToS3(musicData) {
    const date = new Date();
    const filename = `music_${date.getFullYear()}${date.getMonth() + 1}${date.getDate()}${date.getHours()}${date.getMinutes()}${date.getSeconds()}.wav`;
    const params = {
        Bucket: S3_OUTPUT_BUCKET,
        Key: filename,
        Body: musicData,
        ContentType: "audio/wav",
    };

    await new AWS.S3().putObject(params).promise();

    const url = `https://d1al6usgg5x7a.cloudfront.net/${params.Key}`;
    soundsData.push(url);
    return url;
}

app.get("/sound/sagemaker", (req, res) => {
    res.json(soundsData);
});