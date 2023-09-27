import React, { useState } from 'react';

function AddPost() {
    const [title, setTitle] = useState("");
    const [date, setDate] = useState("");
    const [content, setContent] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log({ title, date, content });
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