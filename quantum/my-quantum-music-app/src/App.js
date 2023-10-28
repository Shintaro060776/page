import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';
import { useLocation } from 'react-router-dom';

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
              <li><Link to="/design">Design</Link></li>
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
          <Route path="/design" element={<Design />} />
        </Routes>
      </div>
    </Router>
  );
}

function ExplorePage() {
  const [showModal, setShowModal] = React.useState(true);
  const location = useLocation();

  React.useEffect(() => {
    setShowModal(true);
  }, [location]);

  return (
    <>
      {showModal && <div className='overlay'></div>}

      <div className={showModal ? 'explore-container' : 'hide'}>
        <div className='audio-spectrum'>
          <video id='video' controls width="100%" height="200px">
            <source src='/quantum/effect.mp4' type="video/mp4" />
          </video>
        </div>

        <button className='close-button' onClick={() => setShowModal(false)}>✖</button>
        <button className='generate-button' onClick={() => generateSound()}>Generate Sound</button>
        <button className='play-button' onClick={() => playSound()}>Play Sound</button>
        <button className='stop-button' onClick={() => stopSound()}>Clear Sound</button>
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

    const video = document.getElementById("video");
    video.src = musicUrl;
  } catch (error) {
    console.error("There was a problem with the fetch operation", error.message);
  }
}

function playSound() {
  const video = document.getElementById("video");
  video.play();
}

function stopSound() {
  const video = document.getElementById("video");
  video.pause();
  video.currentTime = 0;
}

function PlaylistPage() {
  return <div>Create Your Playlist Page</div>;
}

function SharePage() {
  return <div>Share with Friends Page</div>;
}

function Design() {
  return <div>Show whole Design</div>;
}

export default App;