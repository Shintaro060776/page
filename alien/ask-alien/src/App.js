import React, { useState } from 'react';
import './App.css';
import alienIcon from './alien-icon.png';
import axios from 'axios';

function App() {
  const [userInput, setUserInput] = useState('');
  const [alienResponse, setAlienResponse] = useState('');

  const handleInputChange = (event) => {
    setUserInput(event.target.value);

    const target = event.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const handleSubmit = async () => {
    try {
      const response = await axios.post('http://localhost:5000/ask-alien', { question: userInput });
      if (response.data && response.data.answer) {
        setAlienResponse(response.data.answer);
      } else {
        setAlienResponse('宇宙人からの回答がありません・・・');
      }
    } catch (error) {
      setAlienResponse('エラーが発生しました');
    }
  };

  return (
    <div className='App'>
      <div className="video-bg">
        <video src="par.mp4" muted loop autoPlay></video>
      </div>
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
        <h5><a href='http://18.177.70.187/'>トップページに戻る</a></h5>
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