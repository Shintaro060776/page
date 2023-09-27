import React from 'react';
import { useParams } from 'react-router-dom';

function PostDetail() {
    const { id } = useParams();

    const post = {
        id: id,
        title: "First post",
        content: "This is the first post in detail."
    };

    return (
        <div>
            <h2>{post.title}</h2>
            <p>{post.content}</p>
        </div>
    );
}

export default PostDetail;