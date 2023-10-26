import React, { useState, useEffect } from 'react';
import './App.css';
import alienIcon from './alien-icon.png';
import axios from 'axios';

function App() {
  const [userInput, setUserInput] = useState('');
  const [alienResponse, setAlienResponse] = useState('');
  const [displayedText, setDisplayedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (displayedText.length < alienResponse.length) {
      const timeoutId = setTimeout(() => {
        setDisplayedText(alienResponse.slice(0, displayedText.length + 1));
      }, 100);
      return () => clearTimeout(timeoutId);
    } else if (alienResponse.length > 0) {
      setIsLoading(false);
    }
  }, [displayedText, alienResponse]);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);

    const target = event.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setDisplayedText('');
    try {
      const response = await axios.post('http://18.177.70.187:3003/ask-alien', { question: userInput });
      console.log(response.data);
      if (response.data && response.data.answer) {
        setAlienResponse(response.data.answer);
        if (response.data.answer.length > 200) {
          setAlienResponse(response.data.answer.slice(0, 200));
          setTimeout(() => {
            setAlienResponse(response.data.answer.slice(200));
          }, 5000);
        } else {
          setAlienResponse(response.data.answer);
        }
      } else if (response.data && response.data.error) {
        setAlienResponse(`エラーが発生しました: ${response.data.error}`);
      } else {
        setAlienResponse('宇宙人からの回答がありません・・・');
      }
    } catch (error) {
      console.error(error);
      setAlienResponse(`エラーが発生しました: ${error.message}`);
      setIsLoading(false);
    }
  };

  return (
    <div className='App' style={{ filter: isLoading ? 'brightness(50%)' : 'none' }}>
      {isLoading && <img src="loading.gif" alt="Loading..." className="loading-gif" />}
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
        <h5><a href='http://neilaeden.com'>トップページに戻る</a></h5>
      </div>
      <div className='response-container'>
        <p>{displayedText}</p>
      </div>
      <div className='service-configuration'>
        <h2 style={{ color: 'hotpink' }}>このサービスの技術スタック</h2>
        <img src='system4.png' alt='システムの構成画像' />
      </div>
    </div>
  );
}

export default App;