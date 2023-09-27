import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import PostDetail from './components/PostDetail';
import './App.css';

function App() {
  return (
    <div>
      <div className="video-background">
        <video playsInline="playsinline" autoPlay="autoplay" muted="muted" loop="loop">
          <source src="/background-video.mp4" type="video/mp4" />
        </video>
      </div>

      <Router basename="/blog">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/post/:id" element={<PostDetail />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;