import React, { useState } from 'react';
import './App.css';
import alienIcon from './alien-icon.png';
import axios from 'axios';

function App() {
  const [userInput, setUserInput] = useState('');
  const [alienResponse, setAlienResponse] = useState('');
  const [displayedTextLength, setDisplayedTextLength] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);

    const target = event.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const typeText = (text) => {
    setDisplayedTextLength(0);
    setAlienResponse('');
    const intervalId = setInterval(() => {
      setDisplayedTextLength((prevLength) => {
        if (prevLength < text.length) {
          return prevLength + 1;
        } else {
          clearInterval(intervalId);
          setIsLoading(false);
          return prevLength;
        }
      });
    }, 50);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('http://18.177.70.187:3001/ask-alien', { question: userInput });
      console.log(response.data);
      if (response.data && response.data.answer) {
        typeText(response.data.answer);
      } else if (response.data && response.data.error) {
        setAlienResponse(`エラーが発生しました: ${response.data.error}`);
        setDisplayedTextLength(response.data.error.length);
        setIsLoading(false);
      } else {
        const defaultMessage = '宇宙人からの回答がありません・・・';
        setAlienResponse(defaultMessage);
        setDisplayedTextLength(defaultMessage.length);
        setIsLoading(false);
      }
    } catch (error) {
      console.error(error);
      const errorMessage = 'エラーが発生しました: ' + error.message;
      setAlienResponse(errorMessage);
      setDisplayedTextLength(errorMessage.length);
      setIsLoading(false);
    }
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
      <h5><a href='http://18.177.70.187/'>トップページに戻る</a></h5>
    </div>
    <div className='response-container'>
      <p>{alienResponse.slice(0, displayedTextLength)}</p>
    </div>
    <div className='service-configuration'>
      <h2 style={{ color: 'hotpink' }}>このサービスの技術スタック</h2>
      <img src='system4.png' alt='システムの構成画像' />
    </div>
  </div>
);

export default App;