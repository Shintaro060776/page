import React, { useState, useEffect } from 'react';
import AddPost from './AddPost';
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
            <h2 className="sec-title">Bulletin Board</h2>
            <h5><a href='http://18.177.70.187/'>トップページに戻る</a></h5>

            <AddPost onPostAdded={handlePostAdded} />

            {posts.map(post => (
                <div key={post.id} className="card">
                    <h2>{post.title}</h2>
                    <p>{post.content}</p>
                    <button className="edit-button" onClick={() => handleEdit(post.id)}>編集</button>
                    <button className="delete-button" onClick={() => handleDelete(post.id)}>削除</button>
                </div>
            ))}

        </div>
    );
}

export default Home;