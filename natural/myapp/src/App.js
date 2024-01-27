import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [isTyping, setIsTyping] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [showImage, setShowImage] = useState(false);
  const [displayedJoke, setDisplayedJoke] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [welcomeMessage, setWelcomeMessage] = useState('');
  const [welcomeMessageVisible, setWelcomeMessageVisible] = useState(false);
  const [messageVisible, setMessageVisible] = useState(true);

  useEffect(() => {
    setWelcomeMessageVisible(true);
    let jokeText = '';
    let index = 0;
    const welcomeText = 'Enjoy Joke GPT. I want you to explore this service and laugh at generated text from AI. Server for machine-learning is expensive, hence i usually stop this. Please let me know if you wanna use it';

    function typeWriter(text) {
      if (index < text.length) {
        jokeText += text.charAt(index);
        setWelcomeMessage(jokeText);
        index++;
        setTimeout(() => typeWriter(text), 150);
      } else {
        setTimeout(() => {
          setMessageVisible(false);
        }, 5000);
      }
    }

    typeWriter(welcomeText);

    return () => {
      clearTimeout();
    };
  }, []);


  const handleGenerateJoke = async () => {
    try {
      setIsTyping(true);
      const response = await axios.post('http://neilaeden.com/api/generate-joke', { prompt: inputValue });

      if (response.data && response.data.joke) {
        let jokeText = '';
        let index = 0;
        const typeWriter = () => {
          if (index < response.data.joke.length) {
            jokeText += response.data.joke[index];
            setDisplayedJoke(jokeText);
            index++;
            setTimeout(typeWriter, 60);
          } else {
            setIsTyping(false);
          }
        };
        typeWriter();
      } else {
        console.error('Joke data is not available in the response');
        setIsTyping(false);
      }
    } catch (error) {
      console.error('Error fetching joke:', error.response?.data?.error, error.message);
      setErrorMessage(error.response?.data?.error || 'An unexpected error occurred');
      setIsTyping(false);
    }
  };

  const handleDesignLinkClick = () => {
    setShowImage(true);
  };

  const handleCloseImage = () => {
    setShowImage(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <span className='App-title'>Joke GPT</span>
        <a href='#design' onClick={handleDesignLinkClick} className="Design-link">Design</a>
        <a href='http://neilaeden.com' className='Design-link'>BackToTopPage</a>
      </header>
      <div className="Joke-output">
        {displayedJoke}
        {isTyping && <span className='caret'></span>}
        {errorMessage && <div className='Error-message'>{errorMessage}</div>}
      </div>
      {welcomeMessageVisible && (
        <div className={`Welcome-message ${messageVisible ? 'visible' : 'hidden'}`}>
          {welcomeMessage}
        </div>
      )}
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Enter a topic for the joke"
        className="Joke-input"
      />
      <button onClick={handleGenerateJoke} className="Generate-joke-button">
        Generate Joke
      </button>
      {showImage && (
        <div className="Image-modal">
          <div className="Image-content">
            <img src="system10.png" alt="Design" />
            <button onClick={handleCloseImage} className="Close-button">&times;</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;