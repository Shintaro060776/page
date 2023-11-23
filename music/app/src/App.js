import React, { useState, useEffect } from 'react';
import './App.css';
import RoundButton from './components/RoundButton';
import TransparentTextArea from './components/TransparentTextArea';
import BackLink from './components/BackLink';
import axios from 'axios';
import { motion } from 'framer-motion';

function App() {
  const [userInput, setUserInput] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [typedLyrics, setTypedLyrics] = useState('');
  const [showImage, setShowImage] = useState(false);
  const typingDelay = 100;

  useEffect(() => {
    if (lyrics.length > 0) {
      let index = 0;
      const interval = setInterval(() => {
        setTypedLyrics((typed) => typed + lyrics[index]);
        index++;
        if (index === lyrics.length) clearInterval(interval);
      }, typingDelay);

      return () => clearInterval(interval);
    }
  }, [lyrics]);

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleGenerateLyrics = async () => {
    try {
      const response = await axios.post('http://neilaeden.com/generate-lyrics', { text: userInput });
      setLyrics(response.data.result.lyrics);
      setTypedLyrics('');
    } catch (error) {
      console.error('Error generating lyrics:', error);
      if (error.response) {
        alert(`Error: ${error.response.status} ${error.response.data}`);
      } else if (error.request) {
        alert("Error: No response from the server. Please try again later");
      } else {
        alert("An unknown error occurred. please try again");
      }
    }
  };

  const handleShowSystemImage = () => {
    setShowImage(!showImage);
  };

  const hideImage = () => {
    if (showImage) {
      setShowImage(false);
    }
  };

  const getBackgroundStyle = () => {
    if (showImage) {
      return { backgroundColor: 'rgba(0, 0, 0, 0.7)' };
    }
    return {};
  };

  return (
    <div className="App" onClick={hideImage} style={getBackgroundStyle()}>
      <div className="SystemButton">
        <RoundButton text="System" onClick={(e) => { e.stopPropagation(); handleShowSystemImage(); }} />
      </div>
      {showImage && <img src="system11.png" alt="SystemImage" className="SystemImage" />}
      <motion.div className="LyricsCard" whileHover={{ scale: 1.05 }} transition={{ type: "spring", stiffness: 300 }}>
        <TransparentTextArea value={typedLyrics} className="TransparentTextArea" readOnly />
        <textarea value={userInput} onChange={handleInputChange} placeholder='歌詞のイメージを入力してください' className="UserInputTextArea" />
        <RoundButton text="Generate Lyrics" onClick={handleGenerateLyrics} className="GenerateButton" />
      </motion.div>
      <div className="BackLinkContainer">
        <BackLink href="http://neilaeden.com" text="Back to Top Page" />
      </div>
    </div>
  );
}

export default App;