import { Link } from 'react-router-dom'
import { Activity, ArrowRight, ArrowUpRight, Boxes, FilePlus2, Files, GitCompareArrows, MessageSquareText, RefreshCw, Sparkles } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import StatusPill from '../components/StatusPill'
import EmptyState from '../components/EmptyState'
import { useApp } from '../hooks/AppContext'
import { METHODS, methodName } from '../utils/format'

const actions = [
  { to: '/documents', title: 'Upload documents', text: 'Add PDF and TXT source files', icon: FilePlus2, tint: 'blue' },
  { to: '/documents', title: 'Manage documents', text: 'Review files and indexing state', icon: Files, tint: 'slate' },
  { to: '/indexing', title: 'Build indexes', text: 'Prepare documents for retrieval', icon: Boxes, tint: 'violet' },
  { to: '/query', title: 'Ask a question', text: 'Retrieve context from an index', icon: MessageSquareText, tint: 'teal' },
  { to: '/compare', title: 'Compare methods', text: 'Inspect all four methods side by side', icon: GitCompareArrows, tint: 'amber' },
]

export default function Dashboard() {
  const { health, healthError, documents, documentsError, documentsLoading, refreshAll } = useApp()
  const availableCount = health?.available_methods?.length
  const readyCount = Object.values(health?.indexes || {}).filter(Boolean).length
  return <>
    <PageHeader eyebrow="RESEARCH WORKSPACE" title="QueryAdaptiveRAG" description="Query-Adaptive Retrieval-Augmented Generation" action={<button className="button button-secondary" onClick={refreshAll}><RefreshCw size={15} />Refresh status</button>} />
    {!health && <div className="banner banner-error"><Activity size={18} /><div><strong>Backend is not reachable</strong><p>{healthError || 'Start the FastAPI backend to load live workspace status.'}</p></div><a href="http://127.0.0.1:8001/docs" target="_blank" rel="noreferrer" className="text-link">Open API docs <ArrowUpRight size={13} /></a></div>}
    <section className="stat-grid" aria-label="Workspace status">
      <article className="stat-card"><div className="stat-top"><span>Backend connection</span><span className="stat-icon blue"><Activity size={17} /></span></div><div className="stat-main"><StatusPill status={health ? 'online' : 'error'} label={health ? 'Connected' : 'Offline'} /></div><div className="stat-foot">{health ? `API version ${health.version}` : 'Live status unavailable'}</div></article>
      <article className="stat-card"><div className="stat-top"><span>Uploaded documents</span><span className="stat-icon violet"><Files size={17} /></span></div><div className="stat-value">{documentsLoading ? <span className="skeleton skeleton-number" /> : documentsError && !documents.length ? '—' : documents.length}</div><div className="stat-foot">{documentsLoading ? 'Loading document list' : documentsError && !documents.length ? 'Document list unavailable' : 'Available in this local workspace'}</div></article>
      <article className="stat-card"><div className="stat-top"><span>Retrieval methods</span><span className="stat-icon teal"><Boxes size={17} /></span></div><div className="stat-value">{health ? availableCount : '—'}<small> / 4</small></div><div className="stat-foot">M1 · M2 · M3 · M4-v3</div></article>
      <article className="stat-card"><div className="stat-top"><span>Indexes ready</span><span className="stat-icon amber"><Sparkles size={17} /></span></div><div className="stat-value">{health ? readyCount : '—'}<small> / 4</small></div><div className="stat-foot">{health ? 'Persisted indexes available now' : 'Waiting for backend status'}</div></article>
    </section>

    <section className="section-block"><div className="section-heading"><div><div className="eyebrow">GET STARTED</div><h2>Workspace actions</h2></div><span className="section-caption">Choose a step to continue</span></div>
      <div className="action-grid">{actions.map(({ to, title, text, icon: Icon, tint }) => <Link className="action-card" to={to} key={title}><span className={`action-icon ${tint}`}><Icon size={19} /></span><span className="action-copy"><strong>{title}</strong><small>{text}</small></span><ArrowRight size={16} className="action-arrow" /></Link>)}</div>
    </section>

    <section className="section-block method-overview"><div className="section-heading"><div><div className="eyebrow">INDEX INVENTORY</div><h2>Retrieval methods</h2></div><Link className="text-link" to="/indexing">Manage indexing <ArrowRight size={14} /></Link></div>
      <div className="method-grid">{METHODS.map((method) => <article className="method-card" key={method.id}><div className="method-card-top"><span className={`method-mark ${method.color}`}>{method.id === 'M4-v3' ? <Sparkles size={16} /> : method.id}</span><StatusPill status={health?.indexes?.[method.id] ?? null} label={health ? (health.indexes?.[method.id] ? 'Index ready' : 'Not indexed') : 'Unknown'} /></div><h3>{methodName(method.id)}</h3><p>{method.description}</p><div className="method-docs">{documents.filter((doc) => doc.indexed_methods?.includes(method.id)).length} documents indexed</div></article>)}</div>
    </section>
    {!documentsLoading && documents.length === 0 && <div className="inline-empty"><EmptyState title="Your workspace is ready" description="Upload a source document to create your first retrieval index." action={<Link to="/documents" className="button button-primary">Upload a document <ArrowRight size={15} /></Link>} /></div>}
  </>
}
