/** Observe a process-local hold around the scenario's original recorded turn. */
import { writeFileSync } from 'node:fs'

/**
 * @param ctx - Scenario context providing the public Agent event stream.
 * @param config - Observation path under the scenario's private DSH home.
 */
export function apply(ctx, config) {
  let entered = false
  ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
    if (entered) return next()
    entered = true
    if (agent.requestPause === undefined) throw new Error('scenario requires turn-admission pause')
    const hold = agent.requestPause()
    const requested = hold.state
    void hold.ready.then(ready => {
      const beforeRelease = hold.state
      hold.release()
      writeFileSync(config.path, JSON.stringify({
        requested, ready, beforeRelease, afterRelease: hold.state, aborted: signal.aborted,
      }))
    })
    return next()
  })
}
