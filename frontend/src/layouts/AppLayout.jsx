import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { Activity, ArrowUpRight, Boxes, ChevronRight, Files, GitCompareArrows, Menu, MessageSquareText, PanelLeftClose, X } from 'lucide-react'
import { useApp } from '../hooks/AppContext'
import { baseURL } from '../api/client'
import StatusPill from '../components/StatusPill'

const navigation = [
  { to: '/', label: 'Overview', icon: Activity, end: true },
  { to: '/documents', label: 'Documents', icon: Files },
  { to: '/indexing', label: 'Indexing', icon: Boxes },
  { to: '/query', label: 'Query lab', icon: MessageSquareText },
  { to: '/compare', label: 'Compare methods', icon: GitCompareArrows },
]

const pageTitles = { '/': 'Overview', '/documents': 'Documents', '/indexing': 'Indexing', '/query': 'Query lab', '/compare': 'Compare methods' }

function SidebarContent({ close }) {
  const { health, healthError } = useApp()
  const online = Boolean(health)
  return <>
    <div className="brand-lockup">
      <div className="brand-symbol"><span /><span /><span /></div>
      <div className="brand-name">QueryAdaptive<span>RAG</span><small>RESEARCH WORKSPACE</small></div>
      <button type="button" aria-label="Close navigation" className="icon-button mobile-close" onClick={close}><X size={18} /></button>
    </div>
    <div className="workspace-label">WORKSPACE</div>
    <nav className="primary-nav" aria-label="Main navigation">
      {navigation.map(({ to, label, icon: Icon, end }) => <NavLink key={to} to={to} end={end} onClick={close} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
        <Icon size={18} strokeWidth={1.9} /><span>{label}</span>{to === '/compare' && <span className="nav-count">4</span>}
      </NavLink>)}
    </nav>
    <div className="sidebar-spacer" />
    <div className="sidebar-connection">
      <div className="connection-heading"><span>API connection</span><StatusPill status={online} label={online ? 'Connected' : 'Offline'} /></div>
      <p title={healthError || baseURL}>{online ? `Backend · ${health?.version || 'API'}` : healthError || baseURL}</p>
      <div className="api-base"><span className={`connection-dot ${online ? 'connected' : ''}`} />{new URL(baseURL).host}</div>
    </div>
    <div className="sidebar-footer"><span className="avatar">QR</span><div><strong>Research workspace</strong><small>Local instance</small></div><ChevronRight size={15} /></div>
  </>
}

export default function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const title = pageTitles[location.pathname] || 'Workspace'
  return <div className="app-shell">
    <aside className="sidebar"><SidebarContent /></aside>
    {mobileOpen && <div className="mobile-nav-layer"><button className="mobile-nav-backdrop" aria-label="Close navigation" onClick={() => setMobileOpen(false)} /><aside className="sidebar mobile-sidebar"><SidebarContent close={() => setMobileOpen(false)} /></aside></div>}
    <main className="main-panel">
      <header className="topbar"><button type="button" className="icon-button menu-toggle" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>
        <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>{title}</strong></div>
        <a className="docs-link" href={`${baseURL}/docs`} target="_blank" rel="noreferrer">API docs <ArrowUpRight size={14} /></a>
      </header>
      <div className="page-content"><Outlet /></div>
      <footer className="main-footer"><span>QueryAdaptiveRAG</span><span>Retrieval research workspace</span></footer>
    </main>
  </div>
}
