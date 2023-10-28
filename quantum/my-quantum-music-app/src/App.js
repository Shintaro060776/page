import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Welcome to My Quantum Music App</h1>
          <nav>
            <ul className="header-menu">
              <li><Link to="/explore">Explore</Link></li>
              <li><Link to="/playlist">Playlist</Link></li>
              <li><Link to="/share">Share</Link></li>
            </ul>
          </nav>
        </header>
        <main className="App-main">
          <div className="feature">
            <h2>Explore Quantum Sounds</h2>
            <p>Discover a new dimension of music with our quantum sound generator.</p>
          </div>
          <div className="feature">
            <h2>Create Your Playlist</h2>
            <p>Combine your favorite quantum sounds into a unique playlist.</p>
          </div>
          <div className="feature">
            <h2>Share with Friends</h2>
            <p>Share your quantum playlists with friends and explore theirs.</p>
          </div>
        </main>
        <footer className="App-footer">
          <div className="social-links">
            <a href="http://neilaeden.com/"><img src="facebook.png" alt="Facebook" /></a>
            <a href="http://neilaeden.com/"><img src="twitter.png" alt="Twitter" /></a>
            <a href="http://neilaeden.com/"><img src="instagram.png" alt="Instagram" /></a>
          </div>
          <p>&copy; 2023 Quantum Music</p>
          <a href="http://neilaeden.com">Back to Top Page</a>
        </footer>

        <Routes>
          <Route path="/explore" element={<ExplorePage />} />
          <Route path="/playlist" element={<PlaylistPage />} />
          <Route path="/share" element={<SharePage />} />
        </Routes>
      </div>
    </Router>
  );
}

function ExplorePage() {
  const [showModal, setShowModal] = React.useState(true);

  return (
    <>
      {showModal && <div className='overlay'></div>}

      <div className={showModal ? 'explore-container' : 'hide'}>
        <div className='audio-spectrum'>
          <audio id='audio' controls>
            <source src='/quantum/effect.mp4' type="video/mp4" />
          </audio>
        </div>

        {/* クローズボタンをクリックしたときの処理を追加 */}
        <button className='close-button' onClick={() => setShowModal(false)}>✖</button>
        <button className='generate-button' onClick={() => generateSound()}>Generate Sound</button>
        <button className='play-button' onClick={() => playSound()}>Play Sound</button>
        <button className='stop-button' onClick={() => stopSound()}>Stop Sound</button>
      </div>
    </>
  );
}

async function generateSound() {
  try {
    const response = await fetch("http://18.177.70.187:4000/generate", {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    const musicUrl = data.url;

    const audio = document.getElementById("audio");
    audio.src = musicUrl;
  } catch (error) {
    console.error("There was a problem with the fetch operation", error.message);
  }
}

function playSound() {
  const audio = document.getElementById("audio");
  audio.play();
}

function stopSound() {
  const audio = document.getElementById("audio");
  audio.pause();
  audio.currentTime = 0;
}

function PlaylistPage() {
  return <div>Create Your Playlist Page</div>;
}

function SharePage() {
  return <div>Share with Friends Page</div>;
}

export default App;