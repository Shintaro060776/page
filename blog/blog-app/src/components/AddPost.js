import React, { useState } from 'react';
import { createPost } from '../api';
import { toast } from 'react-hot-toast';

function AddPost({ onPostAdded }) {
    const [title, setTitle] = useState("");
    const [date, setDate] = useState("");
    const [content, setContent] = useState("");
    const [person, setPerson] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);  // 新しく追加

    const handleSubmit = async (e) => {
        e.preventDefault();
        const newPost = { title, date, content, person };

        setIsSubmitting(true);  // 送信開始前にtrueに設定

        try {
            await createPost(newPost);
            if (onPostAdded) {
                onPostAdded();
            }
            toast.success('投稿が成功しました!');
        } catch (error) {
            console.error("Failed to add the post:", error);
            toast.error('投稿に失敗しました。');
        } finally {
            setIsSubmitting(false);  // 送信後にfalseに設定
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
            <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? '送信中・・・・' : '投稿'}
            </button>
        </form>
    );
};

export default AddPost;