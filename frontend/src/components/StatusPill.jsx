import { Check, Circle, LoaderCircle, X } from 'lucide-react'

export default function StatusPill({ status, label }) {
  const state = status === true || status === 'ready' || status === 'loaded' || status === 'online' ? 'good'
    : status === 'loading' ? 'loading'
      : status === false || status === 'error' || status === 'unavailable' ? 'bad' : 'neutral'
  const Icon = state === 'good' ? Check : state === 'loading' ? LoaderCircle : state === 'bad' ? X : Circle
  return <span className={`status-pill status-${state}`}><Icon size={13} strokeWidth={2.4} className={state === 'loading' ? 'spin' : ''} />{label}</span>
}
