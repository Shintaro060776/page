const AWS = require('aws-sdk');
const dynamo = new AWS.DynamoDB.DocumentClient();
const tableName = "BlogPosts";

exports.handler = async (event) => {
    let response;

    switch (event.httpMethod) {
        case 'GET':
            response = await getPosts();
            break;
        case 'POST':
            response = await createPost(JSON.parse(event.body));
            break;
        default:
            response = buildResponse(400, 'Unsupported method');
    }

    return response;
};

const getPosts = async () => {
    const result = await dynamo.scan({ TableName: tableName }).promise();
    return buildResponse(200, result.Items);
};

const createPost = async (post) => {
    await dynamo.put({
        TableName: tableName,
        Item: post
    }).promise();

    return buildResponse(200, post);
};


const buildResponse = (statusCode, body) => {
    return {
        statusCode: statusCode,
        headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify(body)
    };
};