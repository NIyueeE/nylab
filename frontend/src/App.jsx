import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import TrainPage from './pages/TrainPage';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TrainPage />} />
      </Routes>
    </Router>
  );
}

export default App;