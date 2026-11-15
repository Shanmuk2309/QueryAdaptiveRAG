import { X } from 'lucide-react'

export default function Dialog({ title, children, onClose, footer, wide = false }) {
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose?.() }}>
    <section className={`dialog ${wide ? 'dialog-wide' : ''}`} role="dialog" aria-modal="true" aria-labelledby="dialog-title">
      <header className="dialog-header"><h2 id="dialog-title">{title}</h2><button type="button" className="icon-button" aria-label="Close dialog" onClick={onClose}><X size={18} /></button></header>
      <div className="dialog-body">{children}</div>{footer && <footer className="dialog-footer">{footer}</footer>}
    </section>
  </div>
}
