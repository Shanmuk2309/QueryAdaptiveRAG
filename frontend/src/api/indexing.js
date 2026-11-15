import { api } from './client'

export const indexDocuments = (method, documentIds) => api.post('/api/v1/documents/index', {
  method,
  ...(documentIds?.length ? { document_ids: documentIds } : {}),
}).then(({ data }) => data)
