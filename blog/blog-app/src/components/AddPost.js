import React, { useState } from 'react';

function AddPost() {
    const [title, setTitle] = useState("");
    const [content, setContent] = useState("");

    const handleSubmit = () => {

        console.log("New Post:", { title, content });
    };

    return (
        <div>
            <input value={title} onChange={e => setTitle(e.target.value)} placeholder="Title" />
            <textarea value={content} onChange={e => setContent(e.target.value)} placeholder="Content" />
            <button onClick={handleSubmit}>Submit</button>
        </div>
    );
}

export default AddPost;