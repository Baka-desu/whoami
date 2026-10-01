import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import FeatureScroller from './components/FeatureScroller.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    <FeatureScroller />
  </StrictMode>,
)
