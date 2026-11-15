import { useState } from 'react'
import { AlertCircle, ArrowDown, ArrowRight, LoaderCircle, MessageSquareText, Search, Sparkles } from 'lucide-react'
import { runQuery } from '../api/query'
import { messageFromError } from '../api/client'
import { useApp } from '../hooks/AppContext'
import PageHeader from '../components/PageHeader'
import StatusPill from '../components/StatusPill'
import EmptyState from '../components/EmptyState'
import { ChunkCard, MetricsStrip } from '../components/ChunkCard'
import { METHODS, formatDuration } from '../utils/format'

export default function Query() {
  const { health } = useApp()
  const [question, setQuestion] = useState('')
  const [method, setMethod] = useState('M4-v3')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [showDecisions, setShowDecisions] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    if (!question.trim() || loading) return
    setLoading(true); setError(''); setResult(null)
    try { setResult(await runQuery(question.trim(), method, Number(topK))) }
    catch (requestError) { setError(messageFromError(requestError)) }
    finally { setLoading(false) }
  }
  const available = Boolean(health?.indexes?.[method])

  return <>
    <PageHeader eyebrow="RETRIEVAL WORKSPACE" title="Query lab" description="Run a question against one method index and inspect the retrieved evidence context." />
    {!health && <div className="banner banner-warning"><AlertCircle size={17} /><p>Backend status is unavailable. You can still submit a query, but it may fail if the API is offline.</p></div>}
    <div className="query-layout">
      <section className="panel query-form-panel"><div className="panel-heading"><div><span className="eyebrow">NEW RETRIEVAL</span><h2>Ask a question</h2><p>Results show retrieved context; this API does not generate an answer.</p></div><span className="panel-icon"><MessageSquareText size={18} /></span></div>
        <form onSubmit={submit} className="query-form"><label className="field-label" htmlFor="question">Question</label><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={10000} rows={4} placeholder="For example: Which film was directed by Harold Ramis and stars Bill Murray?" required />
          <div className="input-meta"><span>Ask a focused question against your indexed source documents.</span><span>{question.length.toLocaleString()} / 10,000</span></div>
          <div className="query-controls"><div className="query-method-select"><label className="field-label" htmlFor="query-method">Retrieval method</label><select id="query-method" value={method} onChange={(event) => setMethod(event.target.value)}>{METHODS.map((item) => <option key={item.id} value={item.id}>{item.id} — {item.name}</option>)}</select><span className="method-availability"><StatusPill status={health ? (health.indexes?.[method] ? 'ready' : 'error') : null} label={health ? (available ? 'Index available' : 'No index') : 'Status unknown'} /></span></div>
            <div className="topk-select"><label className="field-label" htmlFor="top-k">Top K</label><input id="top-k" type="number" min="1" max="100" required value={topK} onChange={(event) => setTopK(event.target.value)} /></div></div>
          {error && <div className="form-error" role="alert"><AlertCircle size={16} />{error}</div>}
          <div className="query-submit-row"><span className="muted-small">Queries search the selected method’s index. The API does not support per-document filtering.</span><button className="button button-primary" type="submit" disabled={!question.trim() || loading}>{loading ? <LoaderCircle size={16} className="spin" /> : <Search size={16} />}{loading ? 'Retrieving…' : 'Run query'}</button></div>
        </form>
      </section>
      <aside className="panel query-method-card"><span className="eyebrow">SELECTED METHOD</span><div className="query-method-hero"><span className={`method-mark ${METHODS.find((item) => item.id === method)?.color}`}>{method === 'M4-v3' ? <Sparkles size={18} /> : method}</span><div><h3>{METHODS.find((item) => item.id === method)?.name}</h3><p>{METHODS.find((item) => item.id === method)?.description}</p></div></div><div className="method-info-line"><span>Index status</span><StatusPill status={health ? (available ? 'ready' : 'error') : null} label={health ? (available ? 'Ready' : 'Not built') : 'Unknown'} /></div>
        <div className="query-method-note">{method === 'M4-v3' ? 'M4-v3 classifies the query and evaluates linked neighboring chunks. It may reject every expansion candidate.' : 'This method returns its top retrieved chunks without query-time context expansion.'}</div></aside>
    </div>
    {loading && <div className="panel query-loading"><LoaderCircle className="spin" size={19} /><div><strong>Searching the {method} index</strong><p>Retrieving chunks and assembling the context window.</p></div></div>}
    {result && <section className="query-results" aria-live="polite"><div className="results-heading"><div><div className="eyebrow">RETRIEVAL OUTPUT</div><h2>Retrieved context</h2><p>“{result.query}”</p></div><StatusPill status="ready" label={`${result.retrieved_chunks.length} chunks retrieved`} /></div>
      <div className="result-overview panel"><div className="result-overview-top"><div className="result-method-title"><span className={`method-mark ${METHODS.find((item) => item.id === result.method)?.color}`}>{result.method === 'M4-v3' ? <Sparkles size={16} /> : result.method}</span><div><strong>{result.method} · {METHODS.find((item) => item.id === result.method)?.name}</strong><small>Context assembled from returned retrieval evidence</small></div></div>{result.predicted_complexity && <div className="complexity-pill"><Sparkles size={14} /><span>Predicted complexity</span><strong>{result.predicted_complexity.replace('_', ' ')}</strong></div>}</div>
        <MetricsStrip metrics={result.metrics} />
        <div className="context-box"><div className="context-label"><span>FINAL CONTEXT</span><span>{result.metrics.context_tokens} tokens</span></div><pre>{result.final_context || 'The backend returned an empty context.'}</pre></div>
        {result.sources?.length > 0 && <div className="sources-line"><span>Sources</span>{result.sources.map((source) => <span className="source-chip" key={source.document_id}>{source.filename || source.document_id}</span>)}</div>}
      </div>
      <div className="chunk-section"><div className="chunk-section-heading"><div><h3>Retrieved chunks</h3><p>Ranked source units returned by the selected index</p></div><span className="count-badge">{result.retrieved_chunks.length}</span></div>
        {result.retrieved_chunks.length ? <div className="chunk-list">{result.retrieved_chunks.map((chunk) => <ChunkCard key={chunk.chunk_id} chunk={chunk} />)}</div> : <EmptyState title="No chunks returned" description="Try another query or index additional documents." icon="search" />}</div>
      {result.expanded_chunks?.length > 0 && <div className="chunk-section"><div className="chunk-section-heading"><div><h3>Expanded chunks</h3><p>Additional linked context accepted by M4-v3</p></div><span className="count-badge">{result.expanded_chunks.length}</span></div><div className="chunk-list">{result.expanded_chunks.map((chunk) => <ChunkCard key={chunk.chunk_id} chunk={chunk} expanded />)}</div></div>}
      {result.method === 'M4-v3' && <section className="panel decisions-panel"><button className="decisions-toggle" type="button" onClick={() => setShowDecisions((value) => !value)} aria-expanded={showDecisions}><div><Sparkles size={16} /><span><strong>Expansion candidate decisions</strong><small>{result.candidate_decisions.length} candidates evaluated · {result.expanded_chunks.length} accepted</small></span></div><ArrowDown size={16} className={showDecisions ? 'rotate-180' : ''} /></button>
        {showDecisions && <div className="decision-list">{result.candidate_decisions.length ? result.candidate_decisions.map((decision, index) => <article className={`decision-row ${decision.accepted ? 'accepted' : 'rejected'}`} key={`${decision.candidate_chunk_id}-${index}`}><span className="decision-state">{decision.accepted ? 'Accepted' : 'Rejected'}</span><div><strong>{decision.candidate_chunk_id}</strong><small>{decision.reason?.replaceAll('_', ' ') || 'No reason returned'}</small>{decision.candidate_facts?.length > 0 && <details><summary>{decision.candidate_facts.length} candidate facts</summary><pre>{JSON.stringify(decision.candidate_facts, null, 2)}</pre></details>}</div></article>) : <div className="results-placeholder">No expansion candidates were evaluated for this query.</div>}</div>}
      </section>}
    </section>}
    {!result && !loading && <div className="query-empty-panel"><EmptyState title="Ready when you are" description="Submit a question to see retrieved chunks, source metadata, and the assembled context." icon="search" /></div>}
  </>
}
