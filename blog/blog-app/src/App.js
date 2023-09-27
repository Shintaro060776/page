import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './components/Home';
import PostDetail from './components/PostDetail';
import AddPost from './components/AddPost';

function App() {
  return (
    <Router>
      <Routes>
        <Route exact path="/blog" component={Home} />
        <Route path="/blog/post/:id" component={PostDetail} />
        <Route path="/blog/add" component={AddPost} />
      </Routes>
    </Router>
  );
}

export default App;