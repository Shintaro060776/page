import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import './App.css';
import axios from 'axios';

const endpointUrl = 'https://your-server-url.com/api/data';

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
  const [status, setStatus] = React.useState();
  const location = useLocation();

  React.useEffect(() => {
    setShowModal(true);
  }, [location]);

  async function generateSound() {
    setStatus("processing");
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
      setStatus("done");
    } catch (error) {
      console.error("There was a problem with the fetch operation", error.message);
      setStatus("error");
    }
  }

  function playExploreSound() {
    const video = document.getElementById("video");
    video.play();
  }

  function stopExploreSound() {
    const video = document.getElementById("video");
    video.pause();
    video.currentTime = 0;
  }

  return (
    <>
      {showModal && <div className='overlay'></div>}

      <div className={showModal ? 'explore-container' : 'hide'}>
        <div className='audio-spectrum'>
          <video id='video' controls width="100%" height="200px" playsInline>
            <source src='/quantum/effect.mp4' type="video/mp4" />
          </video>
        </div>

        {status === "processing" && <p className='status-text processing'>Processing...</p>}
        {status === "done" && <p className='status-text done'>Done</p>}
        {status === "error" && <p className='status-text error'>Error</p>}

        <button className='close-button' onClick={() => setShowModal(false)}>✖</button>
        <button className='generate-button' onClick={() => generateSound()}>Generate Sound</button>
        <button className='play-button' onClick={() => playExploreSound()}>Play Sound</button>
        <button className='stop-button' onClick={() => stopExploreSound()}>Clear Sound</button>
      </div>
    </>
  );
}

// async function generateSound() {
//   setStatus("processing");
//   try {
//     const response = await fetch("http://18.177.70.187:4000/generate", {
//       method: "POST",
//     });

//     if (!response.ok) {
//       throw new Error("Network response was not ok");
//     }

//     const data = await response.json();
//     const musicUrl = data.url;

//     const video = document.getElementById("video");
//     video.src = musicUrl;
//     setStatus("done");
//   } catch (error) {
//     console.error("There was a problem with the fetch operation", error.message);
//     setStatus("error");
//   }
// }

// function playExploreSound() {
//   const video = document.getElementById("video");
//   video.play();
// }

// function stopExploreSound() {
//   const video = document.getElementById("video");
//   video.pause();
//   video.currentTime = 0;
// }

export function PlaylistPage() {
  const [sounds, setSounds] = React.useState([]);
  const [showModal, setShowModal] = React.useState(true);

  React.useEffect(() => {
    async function fetchData() {
      try {
        const response = await axios.get(endpointUrl);
        setSounds(response.data);
      } catch (error) {
        console.error("エラーが発生しました:", error);
      }
    }
    fetchData();
  }, []);

  return (
    <>
      {showModal && <div className='overlay'></div>}

      <div className={showModal ? 'explore-container' : 'hide'}>
        {sounds.map(sound => (
          <div key={sound.id}>
            <video id={`video-${sound.id}`} controls width="100%" height="200px" playsInline>
              <source src={sound.url} type="video/mp4" />
            </video>

            <button className='download-button' onClick={() => downloadSound(sound.id)}>Download</button>
            <button className='play-button' onClick={() => playSound(sound.id)}>Play Sound</button>
            <button className='stop-button' onClick={() => stopSound(sound.id)}>Stop Sound</button>
          </div>
        ))}

        <button className='close-button' onClick={() => setShowModal(false)}>✖</button>
      </div>
    </>
  );
}

function playSound(id) {
  const video = document.getElementById(`video-${id}`);
  video.play();
}

function stopSound(id) {
  const video = document.getElementById(`video-${id}`);
  video.pause();
  video.currentTime = 0;
}

function downloadSound(id) {
  const video = document.getElementById(`video-${id}`);
  const a = document.createElement('a');
  a.href = video.src;
  a.download = `sound-${id}.mp4`;
  a.click();
}

export function SharePage() {
  const [isOverlayVisible, setOverlayVisible] = useState(false);
  const location = useLocation();
  const currentUrl = window.location.origin + location.pathname;
  const lineIcon = "/line.png";
  const instagramIcon = "/instagram.png";
  const twitterIcon = "/twitter.png";

  const shareToPlatform = (platform) => {
    let shareUrl = "";

    switch (platform) {
      case "LINE":
        shareUrl = `https://line.me/R/msg/text/?${currentUrl}`;
        break;
      case "Twitter":
        shareUrl = `https://twitter.com/share?url=${currentUrl}&Check%20out%20this%20page!`;
        break;
      case "Instagram":
        alert("URL copied to clipboard");
        navigator.clipboard.writeText(currentUrl);
        return;
      default:
        break;
    }
    window.open(shareUrl, '_blank');
  }

  return (
    <div>
      <h2>Share with Friends</h2>
      <button onClick={() => setOverlayVisible(true)}>Shareリンク</button>

      {isOverlayVisible && (
        <>
          <div className="overlay-icon" onClick={() => setOverlayVisible(false)}></div>
          <button className="close-button-icon" onClick={() => setOverlayVisible(false)}>×</button>
          <div className="icon-container">
            <img
              src={lineIcon} alt="LINE"
              className="icon"
              onClick={() => shareToPlatform("LINE")}
            />
            <img
              src={instagramIcon} alt="Instagram"
              className="icon"
              onClick={() => shareToPlatform("Instagram")}
            />
            <img
              src={twitterIcon} alt="Twitter"
              className="icon"
              onClick={() => shareToPlatform("Twitter")}
            />
          </div>
        </>
      )}
    </div>
  );
}

function Design() {
  const [isOverlayVisible, setOverlayVisible] = useState(false);

  return (
    <div>
      <h2>System Design</h2>
      <button onClick={() => setOverlayVisible(true)}>Design Link</button>

      {isOverlayVisible && (
        <>
          <div className='overlay-system' onClick={() => setOverlayVisible(false)}></div>
          <img src='/system5.png' alt='Design Image' className='design-image' />
          <button className='close-button-system' onClick={() => setOverlayVisible(false)}>×</button>
        </>
      )}
    </div>
  );
}

export default App;