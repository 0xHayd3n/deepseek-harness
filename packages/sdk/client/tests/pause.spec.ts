/** Client pause receipts; only the remote protocol peer is scripted in these unit tests. */
import { fileURLToPath } from 'node:url'
import { describe, expect, it, vi } from 'vitest'
import type { SessionPauseParams } from '@deepseek-ai/dsh-sdk-protocol'
import { createProcessHarnessClient, SdkProtocolError, TransportClosedError } from '../src/client.ts'

function clientFixture() {
  return createProcessHarnessClient({
    command: process.execPath,
    args: [fileURLToPath(new URL('./fake-runtime.ts', import.meta.url))],
    environment: () => ({ ...process.env }),
    description: 'protocol peer for cooperative pause client tests',
    initializeTimeoutMs: 5_000,
  })
}

const identity = { serverInfo: { name: 'deepseek-harness-sdk-runtime', version: 'test' } }
const capability = { runtimeId: 'fixture-runtime', scope: 'turn-admission' }
const target: SessionPauseParams = { sessionId: 'owned', runtimeId: 'fixture-runtime' as SessionPauseParams['runtimeId'], generation: 1, requestId: 'fixture-intent' as SessionPauseParams['requestId'] }
const controls = [
  ['requestPause', 'session/pause'],
  ['getPauseState', 'session/pause-state'],
  ['releasePause', 'session/release-pause'],
] as const

describe('SDK client cooperative pause', () => {
  it.each([
    null,
    {},
    { ...target, scope: 'turn-admission', state: 'paused', sessionId: 'different' },
    { ...target, scope: 'turn-admission', state: 'paused', runtimeId: 'different' },
    { ...target, scope: 'turn-admission', state: 'paused', requestId: 'different' },
    { ...target, scope: 'turn-admission', state: 'paused', generation: 2 },
    { ...target, scope: 'durable', state: 'paused' },
    { ...target, scope: 'turn-admission', state: 'complete' },
    { ...target, scope: 'turn-admission', state: 'unrequested' },
    { ...target, scope: 'turn-admission', state: 'paused', generation: -1 },
    { ...target, scope: 'turn-admission', state: 'paused', generation: 0.5 },
    { ...target, scope: 'turn-admission', state: 'paused', generation: Number.MAX_SAFE_INTEGER + 1 },
  ])('rejects malformed or mismatched request receipts %j without retry or shutdown', async (receipt) => {
    const client = clientFixture()
    client.start()
    const request = vi.spyOn(client, 'request').mockResolvedValueOnce(receipt)
    const close = vi.spyOn(client, 'close')
    try {
      await expect(client.requestPause(target)).rejects.toThrow(SdkProtocolError)
      expect(request).toHaveBeenCalledTimes(1)
      expect(close).not.toHaveBeenCalled()
    } finally {
      close.mockRestore()
      request.mockRestore()
      await client.close()
    }
  })

  it('accepts generation zero only for an unrequested state query', async () => {
    const client = clientFixture()
    client.start()
    const request = vi.spyOn(client, 'request')
    try {
      request.mockResolvedValueOnce({ sessionId: target.sessionId, runtimeId: target.runtimeId, generation: 0, scope: 'turn-admission', state: 'unrequested' })
      expect(await client.getPauseState(target)).toMatchObject({ generation: 0, state: 'unrequested' })
      request.mockResolvedValueOnce({ ...target, generation: 0, scope: 'turn-admission', state: 'paused' })
      await expect(client.getPauseState(target)).rejects.toThrow(SdkProtocolError)
      request.mockResolvedValueOnce({ ...target, generation: 1, scope: 'turn-admission', state: 'unrequested' })
      await expect(client.getPauseState(target)).rejects.toThrow(SdkProtocolError)
    } finally {
      request.mockRestore()
      await client.close()
    }
  })

  it.each(controls)('%s sends the exact target on the existing transport', async (method, wireMethod) => {
    const client = clientFixture()
    client.start()
    const receipt = { ...target, scope: 'turn-admission', state: method === 'releasePause' ? 'released' : 'paused' }
    const request = vi.spyOn(client, 'request').mockResolvedValueOnce(receipt)
    try {
      expect(await client[method](target)).toEqual(receipt)
      expect(request).toHaveBeenCalledExactlyOnceWith(wireMethod, method === 'getPauseState' ? { sessionId: target.sessionId, runtimeId: target.runtimeId } : target)
    } finally {
      request.mockRestore()
      await client.close()
    }
  })

  it.each([false, true])('never starts a replacement runtime for a control without the original transport (closed=%s)', async (closed) => {
    const client = clientFixture()
    if (closed) { client.start(); await client.close() }
    const request = vi.spyOn(client, 'request')
    try {
      for (const [method] of controls) await expect(client[method](target)).rejects.toThrow(TransportClosedError)
      expect(request).not.toHaveBeenCalled()
    } finally {
      request.mockRestore()
      await client.close()
    }
  })

  it.each([
    null,
    { cooperativePause: null },
    { cooperativePause: 'invalid' },
    { cooperativePause: { ...capability, runtimeId: '' } },
    { cooperativePause: { ...capability, runtimeId: 4 } },
    { cooperativePause: { ...capability, scope: 'durable' } },
  ])('rejects malformed pause capability metadata %j', async (capabilities) => {
    const client = clientFixture()
    const request = vi.spyOn(client, 'request').mockResolvedValueOnce({ ...identity, capabilities })
    try {
      await expect(client.initialize({ cwd: process.cwd(), provider: 'fixture', model: 'fixture' })).rejects.toThrow(SdkProtocolError)
    } finally {
      request.mockRestore()
      await client.close()
    }
  })

  it('retains the negotiated runtime identity rather than dropping pause capabilities', async () => {
    const client = clientFixture()
    const request = vi.spyOn(client, 'request').mockResolvedValueOnce({ ...identity, capabilities: { cooperativePause: capability } })
    try {
      const info = await client.initialize({ cwd: process.cwd(), provider: 'fixture', model: 'fixture', capabilities: { cooperativePause: true } })
      expect(info.capabilities).toEqual({ cooperativePause: capability })
      expect(request).toHaveBeenCalledWith('initialize', expect.objectContaining({ capabilities: { cooperativePause: true } }), 5_000)
    } finally {
      request.mockRestore()
      await client.close()
    }
  })
})
