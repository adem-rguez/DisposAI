import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './setup-wizard.css'
import SetupWizard from './SetupWizard.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SetupWizard />
  </StrictMode>,
)
