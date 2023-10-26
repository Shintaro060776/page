import React from 'react';
import { BrowserRouter as Router, Route, Link } from 'react-router-dom';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Welcome to My Quantum Music App</h1>
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
          <p>&copy; 2023 Quantum Music</p>
        </footer>

        {/* ここにルートを追加してページのコンテンツを定義します */}
        <Route path="/explore" component={ExplorePage} />
        <Route path="/playlist" component={PlaylistPage} />
        <Route path="/share" component={SharePage} />
      </div>
    </Router>
  );
}

function ExplorePage() {
  // 量子音楽生成のページコンテンツをここに追加
  return <div>Explore Quantum Sounds Page</div>;
}

function PlaylistPage() {
  // プレイリストのページコンテンツをここに追加
  return <div>Create Your Playlist Page</div>;
}

function SharePage() {
  // シェアのページコンテンツをここに追加
  return <div>Share with Friends Page</div>;
}

export default App;