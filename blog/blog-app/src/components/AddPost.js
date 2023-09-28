import React, { useState } from 'react';
import { createPost } from '../api';

function AddPost({ onPostAdded }) {
    const [title, setTitle] = useState("");
    const [date, setDate] = useState("");
    const [content, setContent] = useState("");
    const [person, setPerson] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        const newPost = { title, date, content, person };  // personフィールドも追加

        try {
            await createPost(newPost);
            if (onPostAdded) {
                onPostAdded();  // 新しい投稿が成功した場合、コールバックを呼び出す
            }
        } catch (error) {
            console.error("Failed to add the post:", error);
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label htmlFor="person">投稿者:</label>
                <input
                    type="text"
                    id="person"
                    value={person}
                    onChange={(e) => setPerson(e.target.value)}
                    required
                />
            </div>
            <div>
                <label htmlFor="title">タイトル:</label>
                <input
                    type="text"
                    id="title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                />
            </div>
            <div>
                <label htmlFor="date">日付:</label>
                <input
                    type="date"
                    id="date"
                    value={date}
                    onChange={(e) => setDate(e.target.value)}
                    required
                />
            </div>
            <div>
                <label htmlFor="content">内容:</label>
                <textarea
                    id="content"
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    required
                />
            </div>
            <button type="submit">投稿</button>
        </form>
    );
};

export default AddPost;