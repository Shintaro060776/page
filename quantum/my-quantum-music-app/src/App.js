import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to My Quantum Music App</h1>
      </header>
      <main>
        <section className="feature">
          <h2>Explore Quantum Sounds</h2>
          <p>Discover a new dimension of music with our quantum sound generator.</p>
        </section>
        <section className="feature">
          <h2>Create Your Playlist</h2>
          <p>Combine your favorite quantum sounds into a unique playlist.</p>
        </section>
        <section className="feature">
          <h2>Share with Friends</h2>
          <p>Share your quantum playlists with friends and explore theirs.</p>
        </section>
      </main>
      <footer className="App-footer">
        <p>&copy; 2023 Quantum Music</p>
      </footer>
    </div>
  );
}

export default App;