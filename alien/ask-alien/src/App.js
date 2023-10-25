import React, { useState, useEffect } from 'react';
import './App.css';
import alienIcon from './alien-icon.png';
import axios from 'axios';

function App() {
  const [userInput, setUserInput] = useState('');
  const [alienResponseChunks, setAlienResponseChunks] = useState([]);
  const [displayedText, setDisplayedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentChunkIndex, setCurrentChunkIndex] = useState(0);

  useEffect(() => {
    let timer;

    if (alienResponseChunks.length > 0 && currentChunkIndex < alienResponseChunks.length) {
      const currentChunk = alienResponseChunks[currentChunkIndex];
      setDisplayedText(prevText => prevText + currentChunk);

      if (currentChunkIndex + 1 < alienResponseChunks.length) {
        timer = setTimeout(() => {
          setCurrentChunkIndex(prevIndex => prevIndex + 1);
        }, 5000);
      }
    }

    // クリーンアップ関数でtimerをクリア
    return () => clearTimeout(timer);
  }, [currentChunkIndex, alienResponseChunks]);

  const handleInputChange = (event) => {
    setUserInput(event.target.value);
    const target = event.target;
    target.style.height = 'auto';
    target.style.height = target.scrollHeight + 'px';
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setDisplayedText('');
    setCurrentChunkIndex(0);

    try {
      const response = await axios.post('http://18.177.70.187:3001/ask-alien', { question: userInput });
      if (response.data && response.data.answer) {
        const MAX_CHARACTERS = 200;
        const responseChunks = [];

        for (let i = 0; i < response.data.answer.length; i += MAX_CHARACTERS) {
          responseChunks.push(response.data.answer.substring(i, i + MAX_CHARACTERS));
        }

        setAlienResponseChunks(responseChunks);
      } else if (response.data && response.data.error) {
        setAlienResponseChunks([`エラーが発生しました: ${response.data.error}`]);
      } else {
        setAlienResponseChunks(['宇宙人からの回答がありません・・・']);
      }
      setIsLoading(false);
    } catch (error) {
      console.error(error);
      setAlienResponseChunks([`エラーが発生しました: ${error.message}`]);
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