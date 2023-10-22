import React, { useState } from 'react';
import './App.css';

function App() {
  const [userInput, setUserInput] = useState('');
  const [alienResponse, setAlienResponse] = useState('');

  const handleInputChange = (event) => {
    setUserInput(event.target.value);

    const target = event.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const handleSubmit = () => {
    setAlienResponse('宇宙人の回答...')
  };

  return (
    <div className='App'>
      <div className='video-container'>
        <video src='alien_video_720p.mp4' autoPlay loop />
      </div>
      <div className='interaction-container'>
        <textarea
          value={userInput}
          onChange={handleInputChange}
          placeholder='質問を入力'
          className='user-input'
        />
        <button onClick={handleSubmit}>Ask Alien</button>
      </div>
      <div className='response-container'>
        <p>{alienResponse}</p>
      </div>
    </div>
  );
}

export default App;