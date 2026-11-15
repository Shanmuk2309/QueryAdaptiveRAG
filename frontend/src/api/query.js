import { api } from './client'

export const runQuery = (query, method, top_k = 5) => api.post('/api/v1/query', { query, method, top_k }).then(({ data }) => data)
