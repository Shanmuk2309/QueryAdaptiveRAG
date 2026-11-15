import { api } from './client'

export const compareMethods = (query, methods, top_k = 5) => api.post('/api/v1/compare', { query, methods, top_k }).then(({ data }) => data)
