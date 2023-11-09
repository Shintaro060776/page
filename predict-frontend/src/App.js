import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showForm, setShowForm] = useState(false);
  const [showDesign, setShowDesign] = useState(false);

  const resizeCanvas = () => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const container = canvas.parentNode;
      if (container) {
        // Set the canvas size to be the container size minus the height of the buttons
        const buttonsHeight = 60; // Adjust as needed based on your button size
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight - buttonsHeight;
      }
    }
  };

  useEffect(() => {
    // This code runs after the component is mounted
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // This will clean up the event listener when the component unmounts
    return () => window.removeEventListener('resize', resizeCanvas);
  }, []);

  const startDrawing = ({ nativeEvent }) => {
    const { offsetX, offsetY } = nativeEvent;
    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = ({ nativeEvent }) => {
    if (!isDrawing) {
      return;
    }
    const { offsetX, offsetY } = nativeEvent;
    const ctx = canvasRef.current.getContext('2d');
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.closePath();
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const executeAnimation = (predictedLabel) => {
    const validLabels = ['smiling', 'openmouth', 'standing', 'nosmile', 'angry', 'stare', 'smile', 'sitting', 'walking', 'running', 'jumping', 'dancing', 'seesomething', 'understand', 'protect'];

    if (validLabels.includes(predictedLabel)) {
      animateMouth();
    } else {
      console.log('No valid label found for animation');
    }
  };

  const animateMouth = (canvas) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let mouthOpen = true;

    const drawMouth = () => {
      ctx.beginPath();
      ctx.arc(400, 300, 50, 0, Math.PI * (mouthOpen ? 1 : 0.5));
      ctx.stroke();
      mouthOpen = !mouthOpen;
    };

    setInterval(() => {
      ctx.clearRect(350, 250, 100, 100);
      drawMouth();
    }, 500);
  };

  const sendDrawingForAnimation = async () => {
    const canvas = canvasRef.current;
    const imageData = canvas.toDataURL('image/png');

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData })
      });

      const data = await response.json();

      if (data.prediction) {
        executeAnimation(data.prediction.label);
      }
    } catch (error) {
      console.error('Error during image animation:', error);
    }
  };

  const handleDrawingClick = () => {
    setShowForm(true);
  }

  const handleDesignClick = () => {
    setShowDesign(true);
  }

  return (
    <div className='App'>
      <header className='App-header'>
        <div className='logo'>
          動くヌコ
        </div>
        <nav>
          <ul>
            <li><a href='#drawing' onClick={handleDrawingClick}>Drawing</a></li>
            <li><a href='#design' onClick={handleDesignClick}>Design</a></li>
          </ul>
        </nav>
      </header>

      {showForm && <div className='overlay' onClick={() => setShowForm(false)}></div>}
      {showDesign && <div className='overlay' onClick={() => setShowDesign(false)}></div>}

      <main>
        {showForm && (
          <div className='drawing-form'>
            <div className='canvas-container'>
              <form>
                <div className="form-content">
                  <canvas
                    ref={canvasRef}
                    onMouseDown={startDrawing}
                    onMouseUp={stopDrawing}
                    onMouseMove={draw}
                    width="800"
                    height="600"
                  />
                  <div className="buttons">
                    <button type="button" onClick={sendDrawingForAnimation}>Animation</button>
                    <button type="button" onClick={clearCanvas}>Clear</button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        )
        }
        {
          showDesign && (
            <div className='design-container'>
              <img src="system9.png" alt="Design" />
            </div>
          )
        }
      </main >
      <section className='intro'>
        <h1>Discover connection between Human and AI</h1>
        <p>This service is gonna give new oppotunity to draw cat image and make it move/animate. Therefore I want you to explore this service a lot</p>
        <p><a href='http://neilaeden.com'>Back to Top Page</a></p>
      </section>
    </div >
  );
}

export default App;
