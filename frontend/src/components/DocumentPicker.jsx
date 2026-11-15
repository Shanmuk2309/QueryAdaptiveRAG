import { Check, FileText } from 'lucide-react'
import EmptyState from './EmptyState'
import StatusPill from './StatusPill'

export default function DocumentPicker({ documents, selected, onChange, disabled = false }) {
  const toggle = (id) => onChange(selected.includes(id) ? selected.filter((value) => value !== id) : [...selected, id])
  if (!documents.length) return <EmptyState title="No documents yet" description="Upload a PDF or TXT file to get started." />
  return <div className="document-picker">
    {documents.map((doc) => {
      const checked = selected.includes(doc.document_id)
      return <button type="button" key={doc.document_id} className={`document-option ${checked ? 'selected' : ''}`}
        aria-pressed={checked} disabled={disabled} onClick={() => toggle(doc.document_id)}>
        <span className={`check-box ${checked ? 'checked' : ''}`}>{checked && <Check size={13} />}</span>
        <span className="doc-type-icon"><FileText size={18} /></span>
        <span className="document-option-main"><strong>{doc.filename}</strong><small>{doc.document_id.slice(0, 12)}… · {doc.text_length.toLocaleString()} extracted characters</small></span>
        <span className="document-option-status"><StatusPill status={doc.processing_status} label={doc.processing_status || 'unknown'} /></span>
      </button>
    })}
  </div>
}
