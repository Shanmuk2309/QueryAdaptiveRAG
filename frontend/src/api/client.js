import axios from 'axios'

const baseURL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001').replace(/\/$/, '')

export const api = axios.create({ baseURL, timeout: 120000 })

export function messageFromError(error) {
  if (error?.response) {
    const detail = error.response.data?.detail
    if (typeof detail === 'string') return detail
    if (detail?.message) return detail.message
    if (Array.isArray(detail)) return detail.map((item) => item.msg).filter(Boolean).join('; ')
    return `Request failed (${error.response.status}).`
  }
  if (error?.code === 'ECONNABORTED') return 'The request took too long. Please try again.'
  return 'Cannot reach the backend. Check that the API is running and the URL is correct.'
}

export { baseURL }
