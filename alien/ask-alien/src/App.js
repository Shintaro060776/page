import React, { useState } from 'react';
import './App.css';
import alienIcon from './alien-icon.png';

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
      <div className='interaction-container'>
        <textarea
          value={userInput}
          onChange={handleInputChange}
          placeholder='質問を入力'
          className='user-input'
        />
        <button className="alien-button" onClick={handleSubmit}>
          Ask Alien
          <img src={alienIcon} alt="Alien Icon" className="alien-icon" />
        </button>
      </div>
      <div className='response-container'>
        <p>{alienResponse}</p>
      </div>
      <div className='service-configuration'>
        <h2 style={{ color: 'hotpink' }}>このサービスの技術スタック</h2>
        <img src='/system4.png' alt='システムの構成画像' />
      </div>
    </div>
  );
}

export default App;