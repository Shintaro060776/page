import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>My Blog Test</h1>
      </header>
      <main>
        <h2>Blog Post Title</h2>
        <p>This is the content of the blog post.</p>
        {/* 他のブログポストもここに追加できます */}
      </main>
    </div>
  );
}

export default App;