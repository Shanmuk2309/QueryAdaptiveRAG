import { useState } from 'react'
import { AlertCircle, ArrowRight, GitCompareArrows, LoaderCircle, Sparkles } from 'lucide-react'
import { compareMethods } from '../api/comparison'
import { messageFromError } from '../api/client'
import { useApp } from '../hooks/AppContext'
import PageHeader from '../components/PageHeader'
import StatusPill from '../components/StatusPill'
import EmptyState from '../components/EmptyState'
import { ChunkCard } from '../components/ChunkCard'
import { METHODS, formatDuration } from '../utils/format'

export default function Compare() {
  const { health } = useApp()
  const [question, setQuestion] = useState('')
  const [methods, setMethods] = useState(METHODS.map((method) => method.id))
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)

  const toggleMethod = (id) => setMethods((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id])
  const submit = async (event) => {
    event.preventDefault()
    if (!question.trim() || !methods.length || loading) return
    setLoading(true); setError(''); setResults(null)
    try { setResults((await compareMethods(question.trim(), methods, Number(topK))).results) }
    catch (requestError) { setError(messageFromError(requestError)) }
    finally { setLoading(false) }
  }

  const rows = methods.map((method) => ({ method, result: results?.[method] }))
  const successCount = rows.filter(({ result }) => result && !result.error).length
  return <>
    <PageHeader eyebrow="SIDE-BY-SIDE ANALYSIS" title="Compare methods" description="Run one query across selected retrieval indexes and inspect their real returned contexts and timings." />
    <section className="panel compare-form-panel"><div className="panel-heading"><div><span className="eyebrow">COMPARISON INPUT</span><h2>Configure the run</h2><p>Methods are queried independently through the backend’s comparison endpoint.</p></div><span className="panel-icon"><GitCompareArrows size={18} /></span></div>
      <form onSubmit={submit} className="compare-form"><label className="field-label" htmlFor="compare-query">Question</label><textarea id="compare-query" value={question} onChange={(event) => setQuestion(event.target.value)} rows={3} maxLength={10000} placeholder="Ask the same question of each selected retrieval method…" required />
        <div className="compare-controls"><div className="compare-methods"><div className="field-label">Methods <span className="muted-small">({methods.length} selected)</span></div><div className="method-check-grid">{METHODS.map((method) => <label className={`method-check ${methods.includes(method.id) ? 'checked' : ''}`} key={method.id}><input type="checkbox" checked={methods.includes(method.id)} onChange={() => toggleMethod(method.id)} disabled={loading} /><span className={`method-mark ${method.color}`}>{method.id === 'M4-v3' ? <Sparkles size={14} /> : method.id}</span><span>{method.id}</span><StatusPill status={health ? (health.indexes?.[method.id] ? 'ready' : 'bad') : null} label={health ? (health.indexes?.[method.id] ? 'Ready' : 'Missing') : 'Unknown'} /></label>)}</div><small className="muted-small">Unavailable method indexes are reported individually by the API; results from other methods can still be displayed.</small></div>
          <div className="compare-submit"><label className="field-label" htmlFor="compare-top-k">Top K</label><input id="compare-top-k" type="number" min="1" max="100" required value={topK} onChange={(event) => setTopK(event.target.value)} /><button type="submit" className="button button-primary" disabled={!question.trim() || !methods.length || loading}>{loading ? <LoaderCircle size={16} className="spin" /> : <GitCompareArrows size={16} />}{loading ? 'Comparing…' : 'Compare methods'}<ArrowRight size={15} /></button></div>
        </div>
        {error && <div className="form-error" role="alert"><AlertCircle size={16} />{error}</div>}
      </form>
    </section>
    {loading && <div className="panel compare-loading"><LoaderCircle size={18} className="spin" /><div><strong>Running comparison</strong><p>The backend is retrieving context for {methods.length} selected method{methods.length === 1 ? '' : 's'}.</p></div></div>}
    {results && <section className="compare-results" aria-live="polite"><div className="results-heading"><div><div className="eyebrow">COMPARISON OUTPUT</div><h2>Method results</h2><p>“{question.trim()}”</p></div><StatusPill status={successCount > 0 ? 'ready' : 'error'} label={`${successCount} of ${methods.length} available`} /></div>
      <div className="summary-table-wrap panel"><table className="summary-table"><thead><tr><th>METHOD</th><th>CONTEXT TOKENS</th><th>RETRIEVED</th><th>EXPANDED</th><th>RETRIEVAL</th><th>EXPANSION</th><th>COMPLEXITY</th></tr></thead><tbody>{rows.map(({ method, result }) => {
        const failed = !result || result.error
        const metrics = failed ? null : result.metrics
        return <tr key={method}><th><span className={`method-mark mini ${METHODS.find((item) => item.id === method)?.color}`}>{method === 'M4-v3' ? <Sparkles size={13} /> : method}</span>{method}</th>{failed ? <td colSpan={6} className="summary-error">{typeof result?.error === 'object' ? result.error.message || JSON.stringify(result.error) : result?.error || 'No result returned.'}</td> : <><td>{metrics.context_tokens}</td><td>{metrics.retrieved_chunk_count}</td><td>{metrics.expanded_chunk_count}</td><td>{formatDuration(metrics.retrieval_latency_ms)}</td><td>{formatDuration(metrics.expansion_latency_ms)}</td><td>{result.predicted_complexity?.replaceAll('_', ' ') || '—'}</td></>}</tr>
      })}</tbody></table><p className="table-note">Values above come from this live API request. They are not research evaluation metrics.</p></div>
      <div className="compare-card-grid">{rows.map(({ method, result }) => {
        const details = METHODS.find((item) => item.id === method)
        const failed = !result || result.error
        const candidateCount = result?.candidate_decisions?.length || 0
        const acceptedCount = result?.candidate_decisions?.filter((item) => item.accepted).length || 0
        return <article className={`method-result-card ${failed ? 'method-result-error' : ''}`} key={method}>
          <header className="method-result-header"><span className={`method-mark ${details.color}`}>{method === 'M4-v3' ? <Sparkles size={16} /> : method}</span><div><strong>{method}</strong><small>{details.name}</small></div><StatusPill status={failed ? 'error' : 'ready'} label={failed ? 'Unavailable' : 'Complete'} /></header>
          {failed ? <div className="method-result-failure"><AlertCircle size={17} /><p>{typeof result?.error === 'object' ? result.error.message || JSON.stringify(result.error) : result?.error || 'No result returned.'}</p></div> : <>
            <div className="method-result-meta">{result.predicted_complexity && <span className="complexity-pill small-complexity"><Sparkles size={12} />{result.predicted_complexity.replaceAll('_', ' ')}</span>}{method === 'M4-v3' && <span className="mini-badge">{acceptedCount} / {candidateCount} expansions accepted</span>}</div>
            <div className="method-context-preview"><div className="context-label"><span>FINAL CONTEXT</span><span>{result.metrics.context_tokens} tokens</span></div><pre>{result.final_context || 'No context returned.'}</pre></div>
            <div className="method-result-chunks"><div className="chunk-section-heading"><h4>Retrieved chunks</h4><span className="count-badge">{result.retrieved_chunks.length}</span></div>{result.retrieved_chunks.length ? result.retrieved_chunks.map((chunk) => <ChunkCard key={chunk.chunk_id} chunk={chunk} />) : <p className="muted-small">No retrieved chunks.</p>}
              {result.expanded_chunks?.length > 0 && <><div className="chunk-section-heading expanded-chunk-heading"><h4>Expanded chunks</h4><span className="count-badge">{result.expanded_chunks.length}</span></div>{result.expanded_chunks.map((chunk) => <ChunkCard key={chunk.chunk_id} chunk={chunk} expanded />)}</>}
            </div>
            {method === 'M4-v3' && candidateCount > 0 && <details className="compact-decisions"><summary>Candidate decisions ({acceptedCount} accepted, {candidateCount - acceptedCount} rejected)</summary><div>{result.candidate_decisions.map((decision, index) => <p key={`${decision.candidate_chunk_id}-${index}`}><strong>{decision.accepted ? 'Accepted' : 'Rejected'} · {decision.candidate_chunk_id}</strong><span>{decision.reason?.replaceAll('_', ' ')}</span></p>)}</div></details>}
          </>}
        </article>
      })}</div>
      {successCount === 0 && <div className="inline-empty"><EmptyState title="No method results available" description="Build one or more indexes, then run the comparison again." icon="search" /></div>}
    </section>}
    {!results && !loading && <div className="query-empty-panel"><EmptyState title="Compare retrieval, not research scores" description="This workspace displays live retrieved chunks, context tokens, and latency returned for each method. It does not calculate Recall@K or use evaluation labels." icon="search" /></div>}
  </>
}
