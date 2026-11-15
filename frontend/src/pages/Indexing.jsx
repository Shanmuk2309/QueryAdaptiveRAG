import { useState } from 'react'
import { AlertCircle, ArrowRight, Boxes, Check, LoaderCircle, Play, RefreshCw, Sparkles } from 'lucide-react'
import { indexDocuments } from '../api/indexing'
import { messageFromError } from '../api/client'
import { useApp } from '../hooks/AppContext'
import PageHeader from '../components/PageHeader'
import EmptyState from '../components/EmptyState'
import DocumentPicker from '../components/DocumentPicker'
import StatusPill from '../components/StatusPill'
import { METHODS, formatDuration } from '../utils/format'

export default function Indexing() {
  const { documents, documentsLoading, health, refreshDocuments, refreshHealth, notify } = useApp()
  const [selectedDocuments, setSelectedDocuments] = useState([])
  const [selectedMethods, setSelectedMethods] = useState(['M1'])
  const [running, setRunning] = useState(false)
  const [activeMethod, setActiveMethod] = useState('')
  const [results, setResults] = useState([])

  const toggleMethod = (id) => setSelectedMethods((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  const runIndexing = async () => {
    if (!selectedDocuments.length || !selectedMethods.length || running) return
    setRunning(true); setResults([])
    const completed = []
    for (const method of METHODS.filter((item) => selectedMethods.includes(item.id))) {
      setActiveMethod(method.id)
      try { completed.push({ method: method.id, result: await indexDocuments(method.id, selectedDocuments) }) }
      catch (error) { completed.push({ method: method.id, error: messageFromError(error) }) }
      setResults([...completed])
    }
    await Promise.all([refreshDocuments(), refreshHealth()])
    setRunning(false); setActiveMethod('')
    const failures = completed.filter((item) => item.error)
    notify(failures.length ? `Indexing finished with ${failures.length} method error${failures.length === 1 ? '' : 's'}.` : `Indexed ${completed.length} method${completed.length === 1 ? '' : 's'} successfully.`,
      failures.length ? 'error' : 'success')
  }

  return <>
    <PageHeader eyebrow="INDEX MANAGEMENT" title="Build retrieval indexes" description="Select source documents and create method-specific indexes using the existing backend pipeline." action={<button className="button button-secondary" onClick={() => { refreshDocuments(); refreshHealth() }}><RefreshCw size={15} />Refresh status</button>} />
    {!health?.embedding_model || health.embedding_model.status !== 'loaded' ? <div className="banner banner-warning"><AlertCircle size={17} /><div><strong>Embedding runtime unavailable</strong><p>Indexing requires the backend embedding model to be loaded. The API health endpoint reports it as {health?.embedding_model?.status || 'unknown'}.</p></div></div> : null}
    <div className="indexing-layout">
      <section className="panel index-step"><div className="step-heading"><span className="step-number">01</span><div><h2>Select documents</h2><p>Only selected documents are added or replaced in each method index.</p></div></div>
        {documentsLoading ? <div className="loading-row"><LoaderCircle className="spin" size={17} />Loading documents…</div> : <><div className="selection-tools"><span>{selectedDocuments.length} of {documents.length} selected</span><div><button className="text-button" onClick={() => setSelectedDocuments(documents.map((doc) => doc.document_id))} disabled={running || !documents.length}>Select all</button><span>·</span><button className="text-button" onClick={() => setSelectedDocuments([])} disabled={running || !selectedDocuments.length}>Clear</button></div></div>
          <DocumentPicker documents={documents} selected={selectedDocuments} onChange={setSelectedDocuments} disabled={running} /></>}
      </section>
      <section className="panel index-step"><div className="step-heading"><span className="step-number">02</span><div><h2>Choose methods</h2><p>Each method has its own isolated index.</p></div></div>
        <div className="method-select-list">{METHODS.map((method) => {
          const checked = selectedMethods.includes(method.id)
          return <label className={`method-select ${checked ? 'selected' : ''}`} key={method.id}><input type="checkbox" checked={checked} onChange={() => toggleMethod(method.id)} disabled={running} />
            <span className={`method-mark ${method.color}`}>{method.id === 'M4-v3' ? <Sparkles size={16} /> : method.id}</span><span className="method-select-main"><strong>{method.id} <small>{method.name}</small></strong><span>{method.description}</span></span><StatusPill status={health?.indexes?.[method.id] ?? null} label={health ? (health.indexes?.[method.id] ? 'Ready' : 'Not indexed') : 'Unknown'} />
          </label>
        })}</div>
        <div className="index-submit"><button className="button button-primary button-large" onClick={runIndexing} disabled={!documents.length || !selectedDocuments.length || !selectedMethods.length || running || health?.embedding_model?.status === 'unavailable'}>{running ? <LoaderCircle size={16} className="spin" /> : <Play size={16} />}{running ? `Indexing ${activeMethod}…` : `Index ${selectedMethods.length} method${selectedMethods.length === 1 ? '' : 's'}`}<ArrowRight size={15} /></button>
          {running && <span className="muted-small">Please keep this page open until the operation completes.</span>}</div>
      </section>
    </div>
    <section className="panel indexing-results"><div className="panel-heading"><div><span className="eyebrow">OPERATION LOG</span><h2>Indexing results</h2><p>Results appear after the backend completes each method.</p></div></div>
      {!results.length ? <div className="results-placeholder"><Boxes size={18} /><span>No indexing operations in this session yet.</span></div> : <div className="index-result-list">{results.map((item) => <article className={`index-result ${item.error ? 'failed' : ''}`} key={item.method}><span className={`method-mark ${METHODS.find((method) => method.id === item.method)?.color}`}>{item.method === 'M4-v3' ? <Sparkles size={16} /> : item.method}</span><div className="index-result-main"><strong>{item.method} · {item.error ? 'Failed' : item.result.index_status}</strong>{item.error ? <p className="error-text">{item.error}</p> : <small>{item.result.document_count} documents · {item.result.chunk_count} chunks · {formatDuration(item.result.indexing_duration_ms)} · {item.result.embedding_model}</small>}</div><StatusPill status={item.error ? 'error' : 'ready'} label={item.error ? 'Error' : 'Complete'} />{!item.error && <Check className="result-check" size={17} />}</article>)}</div>}
    </section>
  </>
}
