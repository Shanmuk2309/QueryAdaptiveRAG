import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { getHealth } from '../api/health'
import { listDocuments } from '../api/documents'
import { messageFromError } from '../api/client'

const AppContext = createContext(null)

export function AppProvider({ children }) {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState('')
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState('')
  const [documentsLoading, setDocumentsLoading] = useState(true)
  const [toasts, setToasts] = useState([])

  const notify = useCallback((message, tone = 'success') => {
    const id = `${Date.now()}-${Math.random()}`
    setToasts((current) => [...current, { id, message, tone }])
    window.setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), 4200)
  }, [])

  const refreshHealth = useCallback(async () => {
    try {
      const result = await getHealth()
      setHealth(result)
      setHealthError('')
      return result
    } catch (error) {
      setHealth(null)
      setHealthError(messageFromError(error))
      return null
    }
  }, [])

  const refreshDocuments = useCallback(async () => {
    setDocumentsLoading(true)
    try {
      const result = await listDocuments()
      setDocuments(Array.isArray(result) ? result : [])
      setDocumentsError('')
      return result
    } catch (error) {
      setDocumentsError(messageFromError(error))
      return null
    } finally {
      setDocumentsLoading(false)
    }
  }, [])

  const refreshAll = useCallback(async () => Promise.all([refreshHealth(), refreshDocuments()]), [refreshHealth, refreshDocuments])

  useEffect(() => { refreshAll() }, [refreshAll])

  const value = useMemo(() => ({ health, healthError, documents, documentsError, documentsLoading,
    refreshHealth, refreshDocuments, refreshAll, notify }),
  [health, healthError, documents, documentsError, documentsLoading, refreshHealth, refreshDocuments, refreshAll, notify])

  return <AppContext.Provider value={value}>
    {children}
    <div className="toast-stack" aria-live="polite" aria-atomic="false">
      {toasts.map((toast) => <div className={`toast toast-${toast.tone}`} key={toast.id} role="status">
        <span className="toast-mark" />{toast.message}
      </div>)}
    </div>
  </AppContext.Provider>
}

export function useApp() {
  const context = useContext(AppContext)
  if (!context) throw new Error('useApp must be used inside AppProvider')
  return context
}
