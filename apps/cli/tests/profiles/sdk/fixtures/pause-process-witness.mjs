/** Test-only process identity witness; no model, tool, or lifecycle replacement. */
import { writeFileSync } from 'node:fs'
export const name = 'sdk-pause-process-witness'
export function apply(_ctx, config) {
  writeFileSync(config.path, String(process.pid))
}
