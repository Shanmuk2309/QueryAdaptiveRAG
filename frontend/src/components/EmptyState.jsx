import { FileSearch, Inbox } from 'lucide-react'

export default function EmptyState({ title, description, action, icon = 'inbox' }) {
  const Icon = icon === 'search' ? FileSearch : Inbox
  return <div className="empty-state"><span className="empty-icon"><Icon size={22} /></span>
    <h3>{title}</h3><p>{description}</p>{action}</div>
}
