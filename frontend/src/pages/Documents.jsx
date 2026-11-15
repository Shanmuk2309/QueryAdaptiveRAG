import { useRef, useState } from 'react'
import { AlertCircle, Check, CloudUpload, FileText, LoaderCircle, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import { deleteDocument, getDocument, uploadDocuments } from '../api/documents'
import { messageFromError } from '../api/client'
import { useApp } from '../hooks/AppContext'
import PageHeader from '../components/PageHeader'
import StatusPill from '../components/StatusPill'
import EmptyState from '../components/EmptyState'
import Dialog from '../components/Dialog'
import { formatBytes, METHODS } from '../utils/format'

function allowedFile(file) { return /\.(pdf|txt)$/i.test(file.name) }

export default function Documents() {
  const { documents, documentsError, documentsLoading, refreshDocuments, refreshHealth, notify } = useApp()
  const inputRef = useRef(null)
  const [selectedFiles, setSelectedFiles] = useState([])
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [pageError, setPageError] = useState('')
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [deleting, setDeleting] = useState(false)

  const addFiles = (incoming) => {
    const files = [...incoming]
    const invalid = files.filter((file) => !allowedFile(file))
    if (invalid.length) {
      setPageError(`Unsupported file${invalid.length > 1 ? 's' : ''}: ${invalid.map((file) => file.name).join(', ')}. Choose PDF or TXT files.`)
    } else setPageError('')
    setSelectedFiles((current) => {
      const merged = [...current, ...files.filter(allowedFile)]
      return merged.filter((file, index) => merged.findIndex((item) => item.name === file.name && item.size === file.size && item.lastModified === file.lastModified) === index)
    })
  }

  const submitUpload = async (event) => {
    event.preventDefault()
    if (!selectedFiles.length || uploading) return
    setUploading(true); setProgress(0); setPageError('')
    try {
      const response = await uploadDocuments(selectedFiles, setProgress)
      setSelectedFiles([])
      if (inputRef.current) inputRef.current.value = ''
      await refreshDocuments()
      notify(`${response.documents.length} document${response.documents.length === 1 ? '' : 's'} uploaded successfully.`)
    } catch (error) {
      setPageError(messageFromError(error))
    } finally { setUploading(false); setProgress(0) }
  }

  const showDetails = async (document) => {
    setDetail(document); setDetailLoading(true)
    try { setDetail(await getDocument(document.document_id)) }
    catch (error) { notify(messageFromError(error), 'error') }
    finally { setDetailLoading(false) }
  }

  const confirmDelete = async () => {
    if (!deleteTarget || deleting) return
    setDeleting(true)
    try {
      await deleteDocument(deleteTarget.document_id)
      await Promise.all([refreshDocuments(), refreshHealth()])
      notify(`${deleteTarget.filename} deleted. Existing method indexes were invalidated.`)
      setDeleteTarget(null)
      if (detail?.document_id === deleteTarget.document_id) setDetail(null)
    } catch (error) { notify(messageFromError(error), 'error') }
    finally { setDeleting(false) }
  }

  return <>
    <PageHeader eyebrow="SOURCE LIBRARY" title="Documents" description="Upload and manage the source files used by your retrieval indexes. PDF and TXT are supported." action={<button className="button button-secondary" onClick={refreshDocuments} disabled={documentsLoading}><RefreshCw size={15} className={documentsLoading ? 'spin' : ''} />Refresh list</button>} />
    <div className="document-layout">
      <section className="panel upload-panel">
        <div className="panel-heading"><div><span className="eyebrow">ADD SOURCES</span><h2>Upload documents</h2><p>Choose files or drop them into the upload area.</p></div><span className="panel-icon"><Upload size={18} /></span></div>
        <form onSubmit={submitUpload}>
          <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
            onDragOver={(event) => event.preventDefault()} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false) }}
            onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files) }}>
            <input ref={inputRef} id="document-files" type="file" multiple accept=".pdf,.txt,application/pdf,text/plain" className="visually-hidden"
              onChange={(event) => { addFiles(event.target.files); event.target.value = '' }} disabled={uploading} />
            <span className="drop-icon"><CloudUpload size={23} /></span>
            <strong>Drop files here to upload</strong><span>or <label htmlFor="document-files" className="inline-link">browse files</label> from your computer</span>
            <small>PDF or TXT · multiple files supported</small>
          </div>
          {selectedFiles.length > 0 && <div className="selected-files"><div className="selected-files-heading"><strong>{selectedFiles.length} file{selectedFiles.length === 1 ? '' : 's'} selected</strong><button type="button" className="text-button" onClick={() => setSelectedFiles([])} disabled={uploading}>Clear all</button></div>
            {selectedFiles.map((file, index) => <div className="selected-file" key={`${file.name}-${file.lastModified}-${index}`}><span className="file-badge"><FileText size={15} /></span><span className="selected-file-copy"><strong>{file.name}</strong><small>{formatBytes(file.size)}</small></span><button type="button" className="icon-button small" aria-label={`Remove ${file.name}`} onClick={() => setSelectedFiles((current) => current.filter((_, fileIndex) => fileIndex !== index))} disabled={uploading}><X size={15} /></button></div>)}
          </div>}
          {uploading && <div className="upload-progress"><div><span>Uploading files</span><strong>{progress}%</strong></div><div className="progress-track"><span style={{ width: `${progress}%` }} /></div></div>}
          {pageError && <div className="form-error" role="alert"><AlertCircle size={16} />{pageError}</div>}
          <div className="upload-actions"><span className="muted-small">Maximum size is enforced by the backend.</span><button className="button button-primary" type="submit" disabled={!selectedFiles.length || uploading}><Upload size={15} />{uploading ? 'Uploading…' : 'Upload files'}</button></div>
        </form>
      </section>

      <section className="panel document-list-panel">
        <div className="panel-heading"><div><span className="eyebrow">LIBRARY</span><h2>Uploaded documents <span className="count-badge">{documents.length}</span></h2><p>Document IDs and extracted text metadata from the backend.</p></div></div>
        {documentsError && <div className="form-error" role="status"><AlertCircle size={16} />{documentsError}</div>}
        {documentsLoading ? <div className="document-skeletons">{[0, 1, 2].map((item) => <div className="document-skeleton" key={item}><span className="skeleton skeleton-icon" /><span className="skeleton skeleton-line" /></div>)}</div>
          : documentsError && documents.length === 0 ? <EmptyState title="Document list unavailable" description="The backend did not return the document list. Check the API connection and refresh." />
            : documents.length === 0 ? <EmptyState title="No documents uploaded" description="Uploaded documents will appear here with their processing and index status." />
            : <div className="document-table-wrap"><table className="document-table"><thead><tr><th>DOCUMENT</th><th>EXTRACTED TEXT</th><th>STATUS</th><th>INDEXES</th><th aria-label="Actions" /></tr></thead><tbody>
              {documents.map((document) => <tr key={document.document_id}><td><div className="table-document"><span className="doc-type-icon"><FileText size={17} /></span><span><button className="document-name" onClick={() => showDetails(document)}>{document.filename}</button><small title={document.document_id}>{document.document_id.slice(0, 12)}…</small></span></div></td>
                <td className="table-number">{document.text_length.toLocaleString()} <small>chars</small></td><td><StatusPill status={document.processing_status} label={document.processing_status || 'unknown'} /></td>
                <td><div className="method-dots" aria-label={`${document.indexed_methods?.length || 0} methods indexed`}>{METHODS.map((method) => <span key={method.id} className={`method-dot ${document.indexed_methods?.includes(method.id) ? `dot-${method.color}` : ''}`} title={`${method.id}: ${document.indexed_methods?.includes(method.id) ? 'indexed' : 'not indexed'}`}>{method.id}</span>)}</div></td>
                <td><div className="row-actions"><button className="icon-button small" aria-label={`View ${document.filename} details`} onClick={() => showDetails(document)}><FileText size={15} /></button><button className="icon-button small danger-hover" aria-label={`Delete ${document.filename}`} onClick={() => setDeleteTarget(document)}><Trash2 size={15} /></button></div></td>
              </tr>)}
            </tbody></table></div>}
      </section>
    </div>

    {detail && <Dialog title="Document details" onClose={() => setDetail(null)} footer={<button className="button button-secondary" onClick={() => setDetail(null)}>Close</button>}>
      {detailLoading ? <div className="loading-row"><LoaderCircle className="spin" size={18} />Loading metadata…</div> : <div className="detail-list"><div className="detail-file"><span className="doc-type-icon large"><FileText size={20} /></span><div><strong>{detail.filename}</strong><small>{detail.document_id}</small></div></div>
        <div className="detail-row"><span>Extracted text</span><strong>{detail.text_length.toLocaleString()} characters</strong></div><div className="detail-row"><span>Processing</span><StatusPill status={detail.processing_status} label={detail.processing_status || 'unknown'} /></div><div className="detail-row detail-method-row"><span>Indexed with</span><div>{METHODS.map((method) => <StatusPill key={method.id} status={detail.indexed_methods?.includes(method.id)} label={method.id} />)}</div></div>
        <p className="detail-note">The document endpoint returns metadata only; extracted text is used by the backend indexer and is not exposed through this API.</p></div>}
    </Dialog>}
    {deleteTarget && <Dialog title="Delete document?" onClose={() => !deleting && setDeleteTarget(null)} footer={<><button className="button button-secondary" onClick={() => setDeleteTarget(null)} disabled={deleting}>Cancel</button><button className="button button-danger" onClick={confirmDelete} disabled={deleting}>{deleting ? <LoaderCircle size={15} className="spin" /> : <Trash2 size={15} />}{deleting ? 'Deleting…' : 'Delete document'}</button></>}>
      <div className="confirm-copy"><span className="confirm-icon"><Trash2 size={19} /></span><p><strong>{deleteTarget.filename}</strong> and its stored metadata will be permanently removed.</p></div><div className="banner banner-warning"><AlertCircle size={17} /><p>Deleting a document invalidates all method indexes. Re-index the remaining documents before querying again.</p></div>
    </Dialog>}
  </>
}
