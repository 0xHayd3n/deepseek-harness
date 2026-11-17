/** SDK pause receipts are scoped to one runtime and monotonically identified hold. */
import { Context } from '@deepseek-ai/cordis'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import { mountAgentLoopTestDependencies } from '@deepseek-ai/dsh-agent-loop-testkit'
import { SessionId } from '@deepseek-ai/dsh-session'
import type { PauseRequestId } from '@deepseek-ai/dsh-sdk-protocol'
import { describe, expect, it } from 'vitest'
import { mountWorkingDirectoryFixture } from '../../../subagent/subagent/tests/working-directory-fixture.ts'
import { MockAdapter, textResponse } from '../../../core/agent-loop/tests/mock-adapter.ts'
import { HarnessSdkJsonRpcServer } from '../src/server.ts'

const requestId = 'fixture-intent' as PauseRequestId

async function setup(adapter = new MockAdapter([textResponse('opened'), textResponse('queued')])) {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  const loop = await ctx.plugin(AgentLoop, { agents: [] })
  await mountWorkingDirectoryFixture(ctx)

  ctx.llm.registerAdapter(['mock'], adapter)
  const server = new HarnessSdkJsonRpcServer(ctx, {
    request: () => Promise.reject(new Error('no external callback is expected')),
    notify: () => {},
  })
  const info = await server.initialize({ cwd: process.cwd(), provider: 'mock', model: 'mock', capabilities: { cooperativePause: true } })
  return { ctx, server, adapter, info, loop }
}

describe('SDK cooperative pause', () => {
  it.each([undefined, null, '', 7])('rejects an invalid wire intent %s before acquiring or releasing a hold', async (invalid) => {
    const { ctx, server, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      const target = { sessionId: 'invalid-intent', runtimeId }
      await server.prompt({ sessionId: target.sessionId, contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: target.sessionId })
      const malformed = { ...target, generation: 1, requestId: invalid }
      await expect(server.handleRequest('session/pause', malformed)).rejects.toThrow('requestId')
      expect(server.getPauseState(target)).toMatchObject({ generation: 0, state: 'unrequested' })
      const owner = { ...target, generation: 1, requestId }
      expect(server.requestPause(owner)).toMatchObject({ state: 'paused' })
      await expect(server.handleRequest('session/release-pause', malformed)).rejects.toThrow('requestId')
      expect(server.getPauseState(target)).toMatchObject({ state: 'paused', requestId })
      server.releasePause(owner)
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('does not equate different caller intents that propose the same generation', async () => {
    const { ctx, server, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'intent-fence', contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: 'intent-fence' })
      const first = { sessionId: 'intent-fence', runtimeId, generation: 1, requestId: 'caller-a-intent' }
      const other = { ...first, requestId: 'caller-b-intent' }
      expect(await server.handleRequest('session/pause', first)).toMatchObject({ state: 'paused' })
      await expect(server.handleRequest('session/pause', other)).rejects.toThrow('different pause request')
      await expect(server.handleRequest('session/release-pause', other)).rejects.toThrow('matching current pause')
      expect(server.getPauseState(first)).toMatchObject({ state: 'paused', requestId: first.requestId })
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('reports no SDK request without claiming that an independently held Agent is unpaused', async () => {
    const { ctx, server, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'external-hold', contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: 'external-hold' })
      const agent = ctx.agents.get(SessionId('external-hold'))!
      const external = agent.requestPause?.()
      expect(external?.state).toBe('paused')
      expect(server.getPauseState({ sessionId: 'external-hold', runtimeId })).toMatchObject({ generation: 0, state: 'unrequested' })
      external?.release()
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('reconciles duplicates and rejects stale, skipped or competing generations without reacquisition', async () => {
    const { ctx, server, adapter, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'generations', contentBlocks: [{ type: 'text', text: 'open' }] })
      const agent = ctx.agents.get(SessionId('generations'))!
      await agent.whenIdle()
      const first = { sessionId: 'generations', runtimeId, generation: 1, requestId }
      const second = { ...first, generation: 2 }
      expect(server.requestPause(first).state).toBe('paused')
      expect(server.requestPause(first).state).toBe('paused')
      expect(() => server.requestPause(second)).toThrow('still owns turn admission')
      expect(() => server.requestPause({ ...first, generation: 3 })).toThrow('out of sequence')
      expect(() => server.releasePause(second)).toThrow('no matching current pause generation')
      expect(server.getPauseState(first).state).toBe('paused')
      expect(server.releasePause(first).state).toBe('released')
      expect(server.releasePause(first).state).toBe('released')
      expect(server.requestPause(first).state).toBe('released')
      expect(server.requestPause(second).state).toBe('paused')
      expect(() => server.requestPause(first)).toThrow('stale')
      expect(() => server.releasePause(first)).toThrow('no matching current pause generation')
      agent.cancel({ kind: 'user' })
      expect(server.getPauseState(first)).toMatchObject({ generation: 2, state: 'invalidated' })
      expect(server.requestPause(second).state).toBe('invalidated')
      expect(server.requestPause({ ...first, generation: 3 }).state).toBe('paused')
      expect(adapter.requests).toHaveLength(1)
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('reports an entered model attempt as pending without abort, retry, or false readiness', async () => {
    const { ctx, server, adapter, info } = await setup(new MockAdapter(['hang']))
    const entered = Promise.withResolvers<undefined>()
    let signal: AbortSignal | undefined
    ctx.on('agent/pre-step', (request, next) => { signal = request.signal; return next() })
    ctx.on('agent/assistant-stream', ({ frame }) => {
      if (frame.type === 'chunk' && frame.chunk.type === 'text-delta') entered.resolve(undefined)
    })
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'entered', contentBlocks: [{ type: 'text', text: 'pending work' }] })
      await entered.promise
      const params = { sessionId: 'entered', runtimeId, generation: 1, requestId }
      expect(await server.handleRequest('session/pause', params)).toMatchObject({ generation: 1, state: 'pending' })
      expect(await server.handleRequest('session/pause-state', params)).toMatchObject({ generation: 1, state: 'pending' })
      expect(await server.handleRequest('session/pause', params)).toMatchObject({ generation: 1, state: 'pending' })
      expect(signal?.aborted).toBe(false)
      expect(adapter.requests).toHaveLength(1)
      ctx.agents.get(SessionId('entered'))!.cancel({ kind: 'user' }, { keepInbox: true })
      expect(await server.handleRequest('session/pause-state', params)).toMatchObject({ generation: 1, state: 'invalidated' })
      expect(await server.handleRequest('session/pause', params)).toMatchObject({ generation: 1, state: 'invalidated' })
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it.each([0, -1, 0.5, Number.NaN, Number.POSITIVE_INFINITY, Number.MAX_SAFE_INTEGER + 1])('denies invalid generation %s before mutation through both executor paths', async (generation) => {
    const { ctx, server, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'invalid', contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: 'invalid' })
      const params = { sessionId: 'invalid', runtimeId, generation, requestId }
      expect(() => server.requestPause(params)).toThrow('positive safe integer')
      expect(() => server.releasePause(params)).toThrow('positive safe integer')
      await expect(server.handleRequest('session/pause', params)).rejects.toThrow('positive safe integer')
      await expect(server.handleRequest('session/release-pause', params)).rejects.toThrow('positive safe integer')
      expect(server.getPauseState(params)).toMatchObject({ generation: 0, state: 'unrequested' })
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('does not discover, create or adopt another Agent or another runtime through a control request', async () => {
    const { ctx, server, info } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      const outside = await ctx.agents.create({ sessionId: SessionId('outside'), agentOptions: { provider: 'mock', model: 'mock' } })
      const before = outside.agent.session.snapshotEvents()
      for (const sessionId of ['missing', 'outside']) {
        for (const method of ['session/pause', 'session/pause-state', 'session/release-pause']) {
          await expect(server.handleRequest(method, { sessionId, runtimeId, generation: 1, requestId })).rejects.toThrow('unknown SDK session')
        }
      }
      expect(ctx.agents.get(SessionId('missing'))).toBeUndefined()
      expect(ctx.agents.get(outside.agent.id)).toBe(outside.agent)
      expect(outside.agent.session.snapshotEvents()).toEqual(before)
      await server.prompt({ sessionId: 'owned-fence', contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: 'owned-fence' })
      const wrong = { sessionId: 'owned-fence', runtimeId: 'not-this-runtime' as typeof runtimeId, generation: 1, requestId }
      expect(() => server.requestPause(wrong)).toThrow('runtime instance does not match')
      expect(() => server.getPauseState(wrong)).toThrow('runtime instance does not match')
      expect(() => server.releasePause(wrong)).toThrow('runtime instance does not match')
      await expect(server.handleRequest('session/pause', wrong)).rejects.toThrow('runtime instance does not match')
      expect(server.getPauseState({ ...wrong, runtimeId })).toMatchObject({ generation: 0, state: 'unrequested' })
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('rejects retained records after actual loop disposal instead of returning a stale paused receipt', async () => {
    const { ctx, server, info, loop } = await setup()
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'detached', contentBlocks: [{ type: 'text', text: 'open' }] })
      await server.wait({ sessionId: 'detached' })
      const params = { sessionId: 'detached', runtimeId, generation: 1, requestId }
      expect(server.requestPause(params).state).toBe('paused')
      await loop.dispose()
      expect(() => server.requestPause(params)).toThrow('disposed outside the server')
      expect(() => server.getPauseState(params)).toThrow('disposed outside the server')
      expect(() => server.releasePause(params)).toThrow('disposed outside the server')
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('retains the unsupported-driver boundary without substituting cancellation', async () => {
    const { ctx, server, adapter, info } = await setup(new MockAdapter(['hang']))
    const entered = Promise.withResolvers<undefined>()
    let signal: AbortSignal | undefined
    ctx.on('agent/pre-step', (request, next) => { signal = request.signal; return next() })
    ctx.on('agent/assistant-stream', ({ frame }) => {
      if (frame.type === 'chunk' && frame.chunk.type === 'text-delta') entered.resolve(undefined)
    })
    try {
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'legacy-driver', contentBlocks: [{ type: 'text', text: 'entered' }] })
      await entered.promise
      const agent = ctx.agents.get(SessionId('legacy-driver'))!
      // This unit fixture hides the optional capability to model a legacy/custom driver.
      Object.defineProperty(agent, 'requestPause', { value: undefined, configurable: true })
      const params = { sessionId: 'legacy-driver', runtimeId, generation: 1, requestId }
      expect(() => server.requestPause(params)).toThrow('does not support cooperative')
      expect(server.getPauseState(params)).toMatchObject({ generation: 0, state: 'unrequested' })
      expect(signal?.aborted).toBe(false)
      expect(adapter.requests).toHaveLength(1)
      delete agent.requestPause
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })

  it('advertises its process-local owner and retains exact queued input until matching generation release', async () => {
    const { ctx, server, adapter, info } = await setup()
    try {
      expect(info).toHaveProperty('capabilities.cooperativePause.runtimeId', expect.any(String))
      const runtimeId = info.capabilities?.cooperativePause?.runtimeId
      if (runtimeId === undefined) throw new Error('cooperative pause was not advertised')
      await server.prompt({ sessionId: 'owned', contentBlocks: [{ type: 'text', text: 'open' }] })
      const agent = ctx.agents.get(SessionId('owned'))!
      await agent.whenIdle()
      expect(await server.handleRequest('session/pause-state', { sessionId: 'owned', runtimeId })).toEqual({
        sessionId: 'owned', runtimeId, scope: 'turn-admission', generation: 0, state: 'unrequested',
      })
      const params = { sessionId: 'owned', runtimeId, generation: 1, requestId }
      expect(await server.handleRequest('session/pause', params)).toEqual({ ...params, scope: 'turn-admission', state: 'paused' })
      const receipt = await server.prompt({ sessionId: 'owned', contentBlocks: [{ type: 'text', text: 'queued' }] })
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([receipt.messageId])
      expect(agent.status).toBe('idle')
      expect(adapter.requests).toHaveLength(1)
      expect(await server.handleRequest('session/release-pause', params)).toEqual({ ...params, scope: 'turn-admission', state: 'released' })
      await server.wait({ sessionId: 'owned' })
      expect(ctx.agents.get(agent.id)).toBe(agent)
      expect(adapter.requests).toHaveLength(2)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'user/message').map(event => event.data.id)).toContain(receipt.messageId)
    } finally {
      await server.shutdown()
      await ctx.fiber.dispose()
    }
  })
})
