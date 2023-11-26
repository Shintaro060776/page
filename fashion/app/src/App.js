import React, { useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [imageUrl, setImageUrl] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);

  const toggleModal = () => {
    setIsModalOpen(!isModalOpen);
  };

  const handleGenerateClick = async () => {
    try {
      const endpoint = "http://neilaeden.com/fashion";

      const response = await axios.post(endpoint);
      const base64Image = response.data.image;
      setImageUrl(`data:image/png;base64,${base64Image}`);
    } catch (error) {
      console.error('Error fetching image:', error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Fashion Generator</h1>
        <a href="#system" onClick={toggleModal}>Design</a>
      </header>
      {isModalOpen && (
        <React.Fragment>
          <div className="overlay" onClick={toggleModal}></div>
          <div className="modal">
            <img src="system12.png" alt="System" className='modal-image' />
            {/* <button onClick={toggleModal}>Close</button> */}
          </div>
        </React.Fragment>
      )}
      <main className="App-main">
        <div className='image-container'>
          {imageUrl && <img src={imageUrl} alt="Generated Fashion" />}
        </div>
        <button onClick={handleGenerateClick}>Generate</button>
        <a href="http://neilaeden.com" className="BackToTopLink">Back To Top Page</a>
      </main>
      <footer className="App-footer">
        <p>© 2023 Fashion Generator</p>
      </footer>
    </div>
  );
}

export default App;