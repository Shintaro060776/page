import React from 'react';

function Home() {

    const posts = [
        { id: 1, title: "First post", content: "This is the first post." },

    ];

    return (
        <div>
            {posts.map(post => (
                <div key={post.id}>
                    <h2>{post.title}</h2>
                    <p>{post.content}</p>
                </div>
            ))}
        </div>
    );
}

export default Home;