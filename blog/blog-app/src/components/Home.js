import React, { useState, useEffect } from 'react';
import AddPost from './AddPost';
import { getAllPosts } from '../api';

function Home() {
    const [posts, setPosts] = useState([]);

    useEffect(() => {
        const fetchPosts = async () => {
            const allPosts = await getAllPosts();
            setPosts(allPosts);
        };

        fetchPosts();
    }, []);

    const handlePostAdded = async () => {
        const allPosts = await getAllPosts();
        setPosts(allPosts);
    };

    return (
        <div className="wrapper">
            <h2 className="sec-title">Blog Posts</h2>

            {posts.map(post => (
                <div key={post.id} className="card">
                    <h2>{post.title}</h2>
                    <p>{post.content}</p>
                </div>
            ))}

            <AddPost onPostAdded={handlePostAdded} />
        </div>
    );
}

export default Home;