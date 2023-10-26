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
          <Link to="/explore" className="feature">
            <h2>Explore Quantum Sounds</h2>
            <p>Discover a new dimension of music with our quantum sound generator.</p>
          </Link>
          <Link to="/playlist" className="feature">
            <h2>Create Your Playlist</h2>
            <p>Combine your favorite quantum sounds into a unique playlist.</p>
          </Link>
          <Link to="/share" className="feature">
            <h2>Share with Friends</h2>
            <p>Share your quantum playlists with friends and explore theirs.</p>
          </Link>
        </main>
        <footer className="App-footer">
          <div className="social-links">
            <a href="#"><img src="facebook.png" alt="Facebook" /></a>
            <a href="#"><img src="twitter.png" alt="Twitter" /></a>
            <a href="#"><img src="instagram.png" alt="Instagram" /></a>
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
  return <div>Explore Quantum Sounds Page</div>;
}

function PlaylistPage() {
  return <div>Create Your Playlist Page</div>;
}

function SharePage() {
  return <div>Share with Friends Page</div>;
}

export default App;