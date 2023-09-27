import React from 'react';

function PostDetail({ match }) {

    const post = {
        id: match.params.id,
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