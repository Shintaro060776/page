import React from 'react';
import AddPost from './AddPost';

function Home() {

    const posts = [
        { id: 1, title: "First post", content: "This is the first post." },
    ];

    return (
        <div className="wrapper">
            <h2 className="sec-title">Blog Posts</h2>

            {posts.map(post => (
                <div key={post.id} className="card">
                    <h2>{post.title}</h2>
                    <p>{post.content}</p>
                </div>
            ))}

            <AddPost />
        </div>
    );
}

export default Home;