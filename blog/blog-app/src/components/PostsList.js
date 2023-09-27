import React, { useState, useEffect } from 'react';
import { getAllPosts, deletePost } from '../api';
import { Link } from 'react-router-dom';

function PostsList() {
    const [posts, setPosts] = useState([]);

    useEffect(() => {
        const fetchPosts = async () => {
            const allPosts = await getAllPosts();
            setPosts(allPosts);
        };

        fetchPosts();
    }, []);

    const handleDelete = async (id) => {
        try {
            await deletePost(id);
            const updatedPosts = posts.filter(post => post.id !== id);
            setPosts(updatedPosts);
        } catch (error) {
            console.error("Failed to delete post:", error);
        }
    };

    return (
        <div>
            <h2>All Posts</h2>
            {posts.map(post => (
                <div key={post.id} className="post">
                    <h3>{post.title}</h3>
                    <p>{post.content}</p>
                    <Link to={`/edit/${post.id}`}>Edit</Link>
                    <button onClick={() => handleDelete(post.id)}>Delete</button>
                </div>
            ))}
        </div>
    );
}

export default PostsList;