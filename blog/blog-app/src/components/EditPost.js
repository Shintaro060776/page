import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { updatePost, getAllPosts } from '../api';

function EditPost() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [post, setPost] = useState(null);
    const [title, setTitle] = useState("");
    const [content, setContent] = useState("");

    useEffect(() => {
        const fetchPost = async () => {
            const allPosts = await getAllPosts();
            const matchedPost = allPosts.find(p => p.id === id);
            if (matchedPost) {
                setPost(matchedPost);
                setTitle(matchedPost.title);
                setContent(matchedPost.content);
            }
        };

        fetchPost();
    }, [id]);

    const handleUpdate = async (e) => {
        e.preventDefault();
        try {
            await updatePost(id, { title, content });
            navigate('/');
        } catch (error) {
            console.error("Failed to update the post:", error);
        }
    };

    if (!post) return <div>Loading...</div>;

    return (
        <div>
            <h2>Edit Post</h2>
            <form onSubmit={handleUpdate}>
                <div>
                    <label htmlFor="title">Title:</label>
                    <input
                        type="text"
                        id="title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="content">Content:</label>
                    <textarea
                        id="content"
                        value={content}
                        onChange={(e) => setContent(e.target.value)}
                        required
                    />
                </div>
                <button type="submit">Update Post</button>
            </form>
        </div>
    );
}

export default EditPost;