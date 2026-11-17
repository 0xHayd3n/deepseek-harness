/**
 * Cooperative pauses retain the Agent, its callbacks, and identified inbox.
 * @module dsh-agent-loop/tests/pause
 */
import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage } from '@deepseek-ai/dsh-llm'
import SessionStore, { SessionId } from '@deepseek-ai/dsh-session'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import * as NativeToolExecutor from '@deepseek-ai/dsh-tool-executor-native'
import AgentRegistry, { type AgentPauseHandle } from '@deepseek-ai/dsh-agent'
import AgentLoop from '@deepseek-ai/dsh-agent-loop'
import SessionProjectionRegistry from '@deepseek-ai/dsh-session-projection'
import { defineContentToolFixture } from '@deepseek-ai/dsh-tool-testkit'
import { MockAdapter, textResponse, toolCallResponse } from './mock-adapter.ts'

async function harness(adapter: MockAdapter) {
  const ctx = new Context()
  await ctx.plugin(LlmRuntime)
  await ctx.plugin(SessionStore)
  await ctx.plugin(SessionProjectionRegistry)
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  await ctx.plugin(NativeToolExecutor)
  await ctx.plugin(AgentRegistry)
  await ctx.plugin(AgentLoop, { agents: [] })
  ctx.llm.registerAdapter(['mock'], adapter)
  return ctx
}

describe('Agent turn-admission pause', () => {
  it('retains a wake deferred from committed turn-end until its holder releases admission', async () => {
    const adapter = new MockAdapter([textResponse('first'), textResponse('deferred input')])
    const ctx = await harness(adapter)
    const delivered = Promise.withResolvers<AgentPauseHandle>()
    let pause: AgentPauseHandle | undefined
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-turn-end-wake'), { provider: 'mock', model: 'mock' })
      const queued = createUserMessage({ content: [{ type: 'text', text: 'deferred input' }], source: { kind: 'user' } })
      ctx.on('session/event', (session, event) => {
        if (session !== agent.session || event.type !== 'turn/end' || event.data.turn !== 1) return
        const holder = agent.requestPause!()
        pause = holder
        queueMicrotask(() => {
          try {
            agent.followup(queued)
            delivered.resolve(holder)
          } catch (error) {
            delivered.reject(error)
          }
        })
      })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'first' }], source: { kind: 'user' } }))
      const holder = await delivered.promise
      expect(await holder.ready).toBe('paused')
      expect(agent.status).toBe('idle')
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([queued.id])
      expect(adapter.requests).toHaveLength(1)
      holder.release()
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(2)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'user/message').map(event => event.data.id)).toContain(queued.id)
    } finally {
      pause?.release()
      await ctx.fiber.dispose()
    }
  })

  it('releases an unready request without aborting or waiting for the entered model call', async () => {
    const adapter = new MockAdapter([textResponse('entered response'), textResponse('queued response')])
    const ctx = await harness(adapter)
    const entered = Promise.withResolvers<undefined>()
    const finish = Promise.withResolvers<undefined>()
    let signal: AbortSignal | undefined
    ctx.on('agent/pre-step', async (request, next) => {
      signal = request.signal
      entered.resolve(undefined)
      await finish.promise
      return next()
    })
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-release-pending'), { provider: 'mock', model: 'mock' })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'entered' }], source: { kind: 'user' } }))
      await entered.promise
      const pause = agent.requestPause!()
      expect(pause.state).toBe('pending')
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'queued' }], source: { kind: 'user' } }))
      pause.release()
      expect(await pause.ready).toBe('released')
      expect(pause.state).toBe('released')
      expect(signal?.aborted).toBe(false)
      finish.resolve(undefined)
      await agent.whenIdle()
      pause.release()
      expect(adapter.requests).toHaveLength(2)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'turn/end').map(event => event.data.reason)).toEqual([{ kind: 'completed' }, { kind: 'completed' }])
    } finally {
      finish.resolve(undefined)
      await ctx.fiber.dispose()
    }
  })

  it('does not let a cancelled holder release or wake its successor', async () => {
    const adapter = new MockAdapter([textResponse('fresh input only')])
    const ctx = await harness(adapter)
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-stale-holder'), { provider: 'mock', model: 'mock' })
      const stale = agent.requestPause!()
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'discard by explicit stop' }], source: { kind: 'user' } }))
      agent.cancel({ kind: 'user' })
      expect(stale.state).toBe('invalidated')
      const current = agent.requestPause!()
      stale.release()
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'fresh input only' }], source: { kind: 'user' } }))
      expect(current.state).toBe('paused')
      expect(agent.status).toBe('idle')
      expect(adapter.requests).toHaveLength(0)
      current.release()
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(1)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'user/message').flatMap(event => event.data.content)).toEqual([{ type: 'text', text: 'fresh input only' }])
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('invalidates a pending pause on explicit keepInbox cancellation without replaying parked input', async () => {
    const adapter = new MockAdapter(['hang', textResponse('parked turn'), textResponse('new wake')])
    const ctx = await harness(adapter)
    const entered = Promise.withResolvers<undefined>()
    ctx.on('agent/assistant-stream', ({ frame }) => {
      if (frame.type === 'chunk' && frame.chunk.type === 'text-delta') entered.resolve(undefined)
    })
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-cancel-pending'), { provider: 'mock', model: 'mock' })
      const first = createUserMessage({ content: [{ type: 'text', text: 'entered' }], source: { kind: 'user' } })
      const parked = createUserMessage({ content: [{ type: 'text', text: 'parked' }], source: { kind: 'user' } })
      agent.followup(first)
      await entered.promise
      const pause = agent.requestPause!()
      agent.followup(parked)
      agent.cancel({ kind: 'user' }, { keepInbox: true })
      expect(pause.state).toBe('invalidated')
      expect(await pause.ready).toBe('invalidated')
      await agent.whenIdle()
      pause.release()
      expect(adapter.requests).toHaveLength(1)
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([parked.id])
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'new wake' }], source: { kind: 'user' } }))
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(3)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'turn/end').map(event => event.data.reason)).toEqual([
        { kind: 'aborted', reason: { kind: 'user' } }, { kind: 'completed' }, { kind: 'completed' },
      ])
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it.each(['user', 'disposed'] as const)('unblocks a ready driver on %s stop without admitting the queued turn', async (kind) => {
    const adapter = new MockAdapter([textResponse('only entered turn')])
    const ctx = await harness(adapter)
    const entered = Promise.withResolvers<undefined>()
    const finish = Promise.withResolvers<undefined>()
    let pause: AgentPauseHandle | undefined
    ctx.on('agent/pre-step', async (_request, next) => {
      entered.resolve(undefined)
      await finish.promise
      return next()
    })
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-stop-' + kind), { provider: 'mock', model: 'mock' })
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'entered turn' }], source: { kind: 'user' } }))
      await entered.promise
      pause = agent.requestPause!()
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'must not enter' }], source: { kind: 'user' } }))
      finish.resolve(undefined)
      expect(await pause.ready).toBe('paused')
      expect(agent.status).toBe('running')
      agent.cancel({ kind })
      expect(pause.state).toBe('invalidated')
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(1)
      expect(agent.inbox.nextTurn).toEqual([])
      expect(agent.inbox.nextStep).toEqual([])
      expect(agent.session.snapshotEvents().filter(event => event.type === 'turn/start')).toHaveLength(1)
    } finally {
      finish.resolve(undefined)
      pause?.release()
      await ctx.fiber.dispose()
    }
  })

  it('waits for already-entered maintenance without aborting it or admitting a queued turn', async () => {
    const adapter = new MockAdapter([textResponse('after maintenance hold')])
    const ctx = await harness(adapter)
    const finish = Promise.withResolvers<undefined>()
    let pause: AgentPauseHandle | undefined
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-maintenance'), { provider: 'mock', model: 'mock' })
      let signal: AbortSignal | undefined
      const task = agent.runMaintenance(async (current) => { signal = current; await finish.promise; return 'preserved maintenance result' })
      pause = agent.requestPause!()
      expect(pause.state).toBe('pending')
      const queued = createUserMessage({ content: [{ type: 'text', text: 'after hold' }], source: { kind: 'user' } })
      agent.followup(queued)
      expect(signal?.aborted).toBe(false)
      finish.resolve(undefined)
      expect(await task).toBe('preserved maintenance result')
      expect(pause.state).toBe('paused')
      expect(await pause.ready).toBe('paused')
      expect(adapter.requests).toHaveLength(0)
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([queued.id])
      pause.release()
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(1)
    } finally {
      finish.resolve(undefined)
      pause?.release()
      await ctx.fiber.dispose()
    }
  })

  it('rejects replacement holders and new maintenance while the ready holder owns admission', async () => {
    const adapter = new MockAdapter([])
    const ctx = await harness(adapter)
    let pause: AgentPauseHandle | undefined
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-exclusive'), { provider: 'mock', model: 'mock' })
      pause = agent.requestPause!()
      expect(() => agent.requestPause!()).toThrow('already has a pause holder')
      let maintenanceEntered = false
      expect(() => agent.runMaintenance(async () => { maintenanceEntered = true })).toThrow('pause holder')
      expect(maintenanceEntered).toBe(false)
      expect(pause.state).toBe('paused')
    } finally {
      pause?.release()
      await ctx.fiber.dispose()
    }
  })

  it('does not manufacture a wake for injected or removed input when admission is released', async () => {
    const adapter = new MockAdapter([])
    const ctx = await harness(adapter)
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-no-ghost-wake'), { provider: 'mock', model: 'mock' })
      const injected = agent.requestPause!()
      agent.inject(createUserMessage({ content: [{ type: 'text', text: 'not a wake' }], source: { kind: 'user' } }))
      injected.release()
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(0)
      expect(agent.inbox.nextStep).toHaveLength(1)
      const removed = agent.requestPause!()
      agent.followup(createUserMessage({ content: [{ type: 'text', text: 'removed before release' }], source: { kind: 'user' } }))
      agent.inbox.clear()
      removed.release()
      removed.release()
      await agent.whenIdle()
      expect(adapter.requests).toHaveLength(0)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'turn/start')).toHaveLength(0)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('retains an idle Agent and its original queued identity until the holder releases admission', async () => {
    const adapter = new MockAdapter([textResponse('after release')])
    const ctx = await harness(adapter)
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-idle'), { provider: 'mock', model: 'mock' })
      expect(agent).toHaveProperty('requestPause', expect.any(Function))
      const pause = agent.requestPause!()
      expect(await pause.ready).toBe('paused')
      const message = createUserMessage({ content: [{ type: 'text', text: 'queued unchanged' }], source: { kind: 'user' } })
      agent.followup(message)
      expect(agent.status).toBe('idle')
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([message.id])
      expect(adapter.requests).toHaveLength(0)
      pause.release()
      await agent.whenIdle()
      expect(pause.state).toBe('released')
      expect(adapter.requests).toHaveLength(1)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'user/message').map(event => event.data.id)).toEqual([message.id])
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('lets the entered tool and turn settle once before holding the original next-turn message', async () => {
    const adapter = new MockAdapter([
      toolCallResponse('original-tool-call', 'receipt', {}),
      textResponse('entered turn completed'),
      textResponse('queued turn after release'),
    ])
    const ctx = await harness(adapter)
    const entered = Promise.withResolvers<undefined>()
    const settled = Promise.withResolvers<undefined>()
    let executions = 0
    let signal: AbortSignal | undefined
    ctx.on('agent/pre-step', (request, next) => { signal = request.signal; return next() })
    ctx.tools.register(defineContentToolFixture({
      name: 'receipt', description: 'an entered operation with a retained result', parameters: {},
      async execute() {
        executions += 1
        entered.resolve(undefined)
        await settled.promise
        return [{ type: 'text', text: 'settled original receipt' }]
      },
    }))
    try {
      const agent = await ctx.agentLoop.create(SessionId('pause-entered-tool'), { provider: 'mock', model: 'mock' })
      const first = createUserMessage({ content: [{ type: 'text', text: 'entered work' }], source: { kind: 'user' } })
      agent.followup(first)
      await entered.promise
      const pause = agent.requestPause!()
      expect(pause.state).toBe('pending')
      const queued = createUserMessage({ content: [{ type: 'text', text: 'next unchanged' }], source: { kind: 'user' } })
      agent.followup(queued)
      expect(executions).toBe(1)
      expect(signal?.aborted).toBe(false)
      expect(adapter.requests).toHaveLength(1)
      settled.resolve(undefined)
      expect(await pause.ready).toBe('paused')
      expect(pause.state).toBe('paused')
      expect(signal?.aborted).toBe(false)
      expect(agent.status).toBe('running')
      expect(agent.inbox.nextTurn.map(item => item.id)).toEqual([queued.id])
      expect(adapter.requests).toHaveLength(2)
      const results = agent.session.snapshotEvents().filter(event => event.type === 'tool/result')
      expect(results).toHaveLength(1)
      expect(results[0]?.data.message.source).toMatchObject({ kind: 'tool', callId: 'original-tool-call' })
      expect(results[0]?.data.message.content).toEqual([{ type: 'text', text: 'settled original receipt' }])
      pause.release()
      await agent.whenIdle()
      expect(executions).toBe(1)
      expect(adapter.requests).toHaveLength(3)
      expect(agent.session.snapshotEvents().filter(event => event.type === 'user/message').map(event => event.data.id)).toEqual([first.id, queued.id])
      expect(agent.session.snapshotEvents().filter(event => event.type === 'turn/end').map(event => event.data.reason)).toEqual([{ kind: 'completed' }, { kind: 'completed' }])
    } finally {
      settled.resolve(undefined)
      await ctx.fiber.dispose()
    }
  })
})
