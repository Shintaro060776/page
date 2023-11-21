import React, { useState } from 'react';
import './App.css';
import RoundButton from './components/RoundButton';
import TransparentTextArea from './components/TransparentTextArea';
import BackLink from './components/BackLink';
import axios from 'axios';
import { motion } from 'framer-motion';

function App() {
  const [userInput, setUserInput] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [showImage, setShowImage] = useState(false);

  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };

  const handleGenerateLyrics = async () => {
    try {
      const response = await axios.post('http://neilaeden.com/generate-lyrics', {
        text: userInput
      });
      setLyrics(response.data.lyrics);
    } catch (error) {
      console.error('Error generating lyrics:', error);
      alert('歌詞の生成に失敗しました');
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

  return (
    <div className="App" onClick={hideImage} style={{ backgroundColor: '#222', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', overflowX: 'hidden' }}>
      <div style={{ position: 'absolute', top: 0, right: 0, padding: '20px', marginBottom: '30px' }}>
        <RoundButton text="System" onClick={(e) => { e.stopPropagation(); handleShowSystemImage(); }} />
      </div>
      {showImage && (
        <img
          src="system11.png"
          alt="SystemImage"
          style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', zIndex: 10 }}
        />
      )}
      <motion.div
        whileHover={{ scale: 1.05 }}
        transition={{ type: "spring", stiffness: 300 }}
        style={{
          perspective: 1000,
          backgroundColor: 'gray',
          borderRadius: '20px',
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.1)',
          padding: '40px',
          width: '60%',
          maxWidth: '700px',
          minHeight: '350px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'relative',
          overflow: 'hidden',
          margin: '0 10px',
        }}
      >
        <TransparentTextArea value={lyrics} onChange={handleInputChange} style={{ backgroundColor: 'transparent' }} />
        <textarea
          value={userInput}
          onChange={handleInputChange}
          placeholder='歌詞のイメージを入力してください'
          style={{ width: '100%', padding: '10px', margin: '20px 0', marginTop: '300px', borderRadius: '10px', border: '1px solid #ddd', outline: 'none' }}
        />
        <RoundButton
          text="Generate Lyrics"
          onClick={handleGenerateLyrics}
          style={{
            margin: '20px 0',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            cursor: 'pointer'
          }}
        />
      </motion.div>
      <div style={{ marginTop: '20px' }}>
        <BackLink href="http://neilaeden.com" text="Back to Top Page" />
      </div>
    </div>
  );
}

export default App;