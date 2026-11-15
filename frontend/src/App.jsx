import { BrowserRouter, Route, Routes } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Indexing from './pages/Indexing'
import Query from './pages/Query'
import Compare from './pages/Compare'
import { AppProvider } from './hooks/AppContext'
import './styles/global.css'

export default function App() {
  return <BrowserRouter><AppProvider><Routes><Route element={<AppLayout />}>
    <Route path="/" element={<Dashboard />} />
    <Route path="/documents" element={<Documents />} />
    <Route path="/indexing" element={<Indexing />} />
    <Route path="/query" element={<Query />} />
    <Route path="/compare" element={<Compare />} />
    <Route path="*" element={<Dashboard />} />
  </Route></Routes></AppProvider></BrowserRouter>
}
