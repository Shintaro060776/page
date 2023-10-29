const express = require("express");
const AWS = require("aws-sdk");
const WavEncoder = require("wav-encoder");
const { Synth, Transport, Pattern, start, Offline } = require("tone");
const { AudioContext } = require("web-audio-api");
const toWav = require("audiobuffer-to-wav");
const cors = require("cors");

const app = express();
const port = 3500;

const DEVICE_ARN = 'arn:aws:braket:::device/quantum-simulator/amazon/sv1';
const S3_OUTPUT_BUCKET = 'vhrthrtyergtcere';
const S3_OUTPUT_DIRECTORY = '';

let soundsData = [];

app.use(cors());

AWS.config.update({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION
});

app.post("/generate", async (req, res) => {
    try {
        const musicData = await runQuantumMusicGenerator();
        const uploadedURL = await uploadMusicToS3(musicData);
        res.json({ url: uploadedURL });
    } catch (error) {
        res.status(500).send(error.message);
    }
});

app.listen(port, () => {
    console.log(`Server started on port ${port}`);
});

async function runQuantumMusicGenerator() {
    const quantumResults = await getQuantumResults();
    return parseQuantumResultToMusic(quantumResults);
}

async function getQuantumResults() {
    try {
        const resultFileName = generateResultFileName();

        const taskId = await createQuantumTask(resultFileName);
        await waitForTaskCompletion(taskId);

        return await fetchResultFromS3(resultFileName);
    } catch (error) {
        console.error("Error generating music with AWS Braket:", error);
        console.error("Error in running Quantum Music Generator:", error);
        throw new Error("音楽の生成に問題が発生しました。");
    }
}

function generateResultFileName() {
    const currentDate = new Date();
    const formattedDate = currentDate.toISOString().replace(/[:.-]/g, '');
    return `result_${formattedDate}.txt`;
}

async function createQuantumTask(resultFileName) {
    const quantumCircuit = {
        qubits: 2,
        gates: [
            { name: 'h', targets: [0] },
            { name: 'cx', controls: [0], target: [1] }
        ]
    };

    const response = await new AWS.Braket().createQuantumTask({
        action: JSON.stringify(quantumCircuit),
        deviceArn: DEVICE_ARN,
        outputS3Bucket: S3_OUTPUT_BUCKET,
        outputS3Directory: S3_OUTPUT_DIRECTORY,
        shots: 1000
    }).promise();

    return response.quantumTaskArn;
}

async function waitForTaskCompletion(taskId) {
    let taskStatus;
    do {
        const response = await new AWS.Braket().getQuantumTask({ quantumTaskArn: taskId }).promise();
        taskStatus = response.status;
        if (['FAILED', 'CANCELLED'].includes(taskStatus)) {
            throw new Error('Quantum task did not complete successfully.');
        }
        await new Promise(resolve => setTimeout(resolve, 5000));
    } while (taskStatus !== 'COMPLETED');
}

async function fetchResultFromS3(resultFileName) {
    const s3Params = {
        Bucket: S3_OUTPUT_BUCKET,
        Key: `${S3_OUTPUT_DIRECTORY}/${resultFileName}`
    };
    const s3Response = await new AWS.S3().getObject(s3Params).promise();
    return s3Response.Body;
}

async function parseQuantumResultToMusic(resultData) {
    const quantumResults = resultData.toString().split("\n");
    const notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4'];
    const synth = new Synth();

    const patterns = quantumResults.map(bitString => {
        const noteIndex = parseInt(bitString, 2) % notes.length;
        if (bitString.endsWith('00')) {
            return { note: notes[noteIndex], duration: '4n' };
        } else if (bitString.endsWith('01')) {
            return { note: notes[noteIndex], duration: '8n' };
        } else if (bitString.endsWith('10')) {
            return { note: notes[noteIndex], duration: '16n' };
        } else {
            return { note: notes[noteIndex], duration: '2n' };
        }
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

app.get("/api/data", (req, res) => {
    res.json(soundsData);
});