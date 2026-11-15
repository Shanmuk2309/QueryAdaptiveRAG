export const METHODS = [
  { id: 'M1', name: 'Fixed 256', description: '256-token chunks · 32-token overlap', color: 'blue' },
  { id: 'M2', name: 'Fixed 512', description: '510 content tokens · 32-token overlap', color: 'violet' },
  { id: 'M3', name: 'Semantic', description: 'Meaning-based chunk boundaries', color: 'teal' },
  { id: 'M4-v3', name: 'Adaptive', description: 'Fact-level context expansion', color: 'amber' },
]

export const methodName = (id) => METHODS.find((method) => method.id === id)?.name || id

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function formatDuration(value) {
  if (!Number.isFinite(value)) return '—'
  return value < 1000 ? `${value.toFixed(0)} ms` : `${(value / 1000).toFixed(2)} s`
}

export function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? '—' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(date)
}
