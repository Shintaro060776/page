import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import Home from './components/Home';
import PostDetail from './components/PostDetail';
import AddPost from './components/AddPost';

function App() {
  return (
    <Router>
      <Switch>
        <Route exact path="/blog" component={Home} />
        <Route path="/blog/post/:id" component={PostDetail} />
        <Route path="/blog/add" component={AddPost} />
      </Switch>
    </Router>
  );
}

export default App;