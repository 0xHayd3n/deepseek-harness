/** Process-local SDK admission receipts; the Agent retains execution and callbacks. */
import type { Agent, AgentPauseHandle } from '@deepseek-ai/dsh-agent'
import type { PauseRequestId, PauseRuntimeId, SessionPauseResult } from '@deepseek-ai/dsh-sdk-protocol'

/** One SDK-owned Agent retains only its latest, monotonically identified receipt. */
export class SdkSessionPause {
  private current: { readonly generation: number; readonly requestId: PauseRequestId; readonly handle: AgentPauseHandle } | undefined

  /**
   * @param agent - exact live Agent whose SDK record owns this controller.
   * @param runtimeId - the server instance identity; not an authorization credential.
   */
  constructor(private readonly agent: Agent, private readonly runtimeId: PauseRuntimeId) {}

  /**
   * Request the next hold, or reconcile this generation without reacquiring it.
   * @param generation - positive safe integer, exactly one after the latest receipt.
   * @param requestId - the caller's retained identity for this intent.
   * @returns a receipt; `pending` is not a completed pause.
   * @throws on stale/skipped generations, a different active hold, or an unsupported driver.
   */
  request(generation: number, requestId: PauseRequestId): SessionPauseResult {
    this.validateGeneration(generation)
    if (this.current?.generation === generation) {
      if (this.current.requestId !== requestId) throw new Error('generation belongs to a different pause request')
      return this.state()
    }
    const latest = this.current?.generation ?? 0
    if (generation !== latest + 1) throw new Error('pause generation is stale or out of sequence')
    if (this.current !== undefined && (this.current.handle.state === 'pending' || this.current.handle.state === 'paused')) {
      throw new Error('a different pause generation still owns turn admission')
    }
    if (this.agent.requestPause === undefined) throw new Error('Agent driver does not support cooperative turn-admission pause')
    const handle = this.agent.requestPause()
    this.current = { generation, requestId, handle }
    return this.state()
  }

  /**
   * Read the latest process-local receipt without creating, resuming, or waking an Agent.
   * @returns generation zero and `unrequested` before this SDK owner's first request.
   */
  state(): SessionPauseResult {
    return {
      sessionId: this.agent.session.id,
      runtimeId: this.runtimeId,
      scope: 'turn-admission',
      generation: this.current?.generation ?? 0,
      ...this.current === undefined ? {} : { requestId: this.current.requestId },
      state: this.current?.handle.state ?? 'unrequested',
    }
  }

  /**
   * Release only the identified latest hold; terminal receipts remain terminal.
   * @param generation - the exact generation of the hold being released.
   * @param requestId - the exact retained intent identity of that hold.
   * @returns the current receipt after release, without cancelling the entered turn.
   * @throws on missing or stale generations; no other holder is affected.
   */
  release(generation: number, requestId: PauseRequestId): SessionPauseResult {
    this.validateGeneration(generation)
    if (this.current === undefined || generation !== this.current.generation || requestId !== this.current.requestId) {
      throw new Error('no matching current pause generation')
    }
    this.current.handle.release()
    return this.state()
  }

  /** Reject lossy or non-positive wire counters before any ownership mutation. */
  private validateGeneration(generation: number): void {
    if (!Number.isSafeInteger(generation) || generation <= 0) {
      throw new TypeError('pause generation must be a positive safe integer')
    }
  }
}
