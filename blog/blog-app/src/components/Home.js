import React, { useState, useEffect } from 'react';
import AddPost from './AddPost';
import { getAllPosts } from '../api';
import { useNavigate } from 'react-router-dom';
import { getAllPosts, deletePost } from '../api';

function Home() {
    const [posts, setPosts] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchPosts = async () => {
            const allPosts = await getAllPosts();
            setPosts(allPosts);
        };

        fetchPosts();
    }, []);

    const handleEdit = (id) => {
        navigate(`/edit/${id}`);
    }

    const handleDelete = async (id) => {
        try {
            await deletePost(id);
            const updatedPosts = posts.filter(post => post.id !== id);
            setPosts(updatedPosts);
        } catch (error) {
            console.log("Failed to delete post:", error);
        }
    };

    const handlePostAdded = async () => {
        const allPosts = await getAllPosts();
        setPosts(allPosts);
    };

    return (
        <div className="wrapper">
            <h2 className="sec-title">Blog Posts</h2>

            <AddPost onPostAdded={handlePostAdded} />

            {posts.map(post => (
                <div key={post.id} className="card">
                    <h2>{post.title}</h2>
                    <p>{post.content}</p>
                    <button onClick={() => handleEdit(post.id)}>編集</button>
                    <button onClick={() => handleDelete(post.id)}>削除</button>
                </div>
            ))}

        </div>
    );
}

export default Home;