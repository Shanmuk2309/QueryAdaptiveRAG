import { api } from './client'

let pendingDocumentList = null
export function listDocuments() {
  if (!pendingDocumentList) pendingDocumentList = api.get('/api/v1/documents').then(({ data }) => data).finally(() => { pendingDocumentList = null })
  return pendingDocumentList
}
export const getDocument = (id) => api.get(`/api/v1/documents/${encodeURIComponent(id)}`).then(({ data }) => data)
export const deleteDocument = (id) => api.delete(`/api/v1/documents/${encodeURIComponent(id)}`)

export function uploadDocuments(files, onProgress) {
  const body = new FormData()
  files.forEach((file) => body.append('files', file, file.name))
  return api.post('/api/v1/documents/upload', body, {
    onUploadProgress: (event) => {
      if (event.total) onProgress?.(Math.round((event.loaded / event.total) * 100))
    },
  }).then(({ data }) => data)
}
