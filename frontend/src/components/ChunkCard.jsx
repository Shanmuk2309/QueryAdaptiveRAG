import { Check, Copy, FileText, Sparkles } from 'lucide-react'
import { useState } from 'react'
import { formatDuration } from '../utils/format'

export function ChunkCard({ chunk, expanded = false }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(chunk.text); setCopied(true); window.setTimeout(() => setCopied(false), 1600) }
    catch { setCopied(false) }
  }
  return <article className={`chunk-card ${expanded ? 'chunk-expanded' : ''}`}>
    <div className="chunk-card-header"><div className="chunk-source"><span className={`chunk-type-icon ${expanded ? 'expanded' : ''}`}>{expanded ? <Sparkles size={14} /> : <FileText size={14} />}</span><div><strong>{chunk.filename || 'Source document'}</strong><small title={chunk.document_id}>{chunk.chunk_id}</small></div></div>
      <div className="chunk-badges">{chunk.rank != null && <span className="mini-badge">Rank {chunk.rank}</span>}{Number.isFinite(chunk.score) && <span className="mini-badge score-badge">Score {chunk.score.toFixed(3)}</span>}<button className="icon-button small" onClick={copy} aria-label="Copy chunk text" title="Copy chunk text">{copied ? <Check size={14} /> : <Copy size={14} />}</button></div></div>
    <p className="chunk-text">{chunk.text}</p><div className="chunk-foot"><span>Characters {chunk.start}–{chunk.end}</span>{expanded && <span className="expanded-label">Expanded context</span>}</div>
  </article>
}

export function MetricsStrip({ metrics }) {
  if (!metrics) return null
  return <div className="metrics-strip"><div><span>Context tokens</span><strong>{metrics.context_tokens}</strong></div><div><span>Retrieved chunks</span><strong>{metrics.retrieved_chunk_count}</strong></div><div><span>Expanded chunks</span><strong>{metrics.expanded_chunk_count}</strong></div><div><span>Retrieval latency</span><strong>{formatDuration(metrics.retrieval_latency_ms)}</strong></div><div><span>Expansion latency</span><strong>{formatDuration(metrics.expansion_latency_ms)}</strong></div></div>
}
