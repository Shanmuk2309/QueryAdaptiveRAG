import { api } from './client'

let pendingHealth = null
export function getHealth() {
  if (!pendingHealth) pendingHealth = api.get('/api/v1/health').then(({ data }) => data).finally(() => { pendingHealth = null })
  return pendingHealth
}
