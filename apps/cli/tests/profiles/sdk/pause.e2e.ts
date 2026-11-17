/** Real Loader/profile/stdio admission pause with an original native question callback. */
import { createServer } from 'node:http'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { HarnessClient, type PauseRequestId, type SdkUserQuestionRequest, type SdkUserQuestionResult } from '@deepseek-ai/dsh-sdk-client'
import { expect, it, onTestFinished } from 'vitest'
import { messagesResponse } from './fixtures/messages-response.ts'
import { profileEnvironment } from './fixtures/profile-environment.ts'

it('retains the original runtime, stdio callback and queued message across turn admission pause', async () => {
  const root = await mkdtemp(join(tmpdir(), 'sdk-admission-pause-'))
  const requests: Record<string, unknown>[] = []
  const questions = [{ id: 'format', question: 'Which format?', options: [{ label: 'CSV' }, { label: 'JSON' }] }]
  const model = createServer((incoming, response) => {
    let body = ''
    incoming.setEncoding('utf8')
    incoming.on('data', (chunk: string) => { body += chunk })
    incoming.on('end', () => {
      requests.push(JSON.parse(body) as Record<string, unknown>)
      response.writeHead(200, { 'content-type': 'text/event-stream' })
      response.end(messagesResponse(requests.length === 1
        ? { type: 'tool_use', id: 'original-question-call', name: 'ask_user_question', input: { questions } }
        : { type: 'text', text: requests.length === 2 ? 'original answer consumed' : 'queued input consumed' },
      requests.length === 1 ? 'tool_use' : 'end_turn'))
    })
  })
  const runtime: { client?: HarnessClient } = {}
  onTestFinished(async () => {
    try { await runtime.client?.close() } finally {
      try { await new Promise<void>((resolve, reject) => model.close((error?: Error) => {
        if (error === undefined) resolve()
        else reject(error)
      })) }
      finally { await rm(root, { recursive: true, force: true }) }
    }
  })
  await new Promise<void>(resolve => model.listen(0, '127.0.0.1', resolve))
  const address = model.address()
  if (address === null || typeof address === 'string') throw new Error('model fixture has no port')
  const pidPath = join(root, 'runtime.pid')
  const patchPath = join(root, 'pause-witness.patch.yml')
  await writeFile(patchPath, JSON.stringify([{ insert: [{
    id: 'sdk-pause-process-witness',
    name: fileURLToPath(new URL('./fixtures/pause-process-witness.mjs', import.meta.url)),
    config: { path: pidPath },
  }] }]))
  const seen: SdkUserQuestionRequest[] = []
  const entered = Promise.withResolvers<AbortSignal>()
  const answer = Promise.withResolvers<SdkUserQuestionResult>()
  const client = new HarnessClient({
    dshBin: fileURLToPath(new URL('../../../lib/bin.js', import.meta.url)),
    profile: 'sdk', patches: [patchPath], dshHome: join(root, 'home'), processCwd: root,
    env: profileEnvironment(root, `http://127.0.0.1:${address.port}`), initializeTimeoutMs: 30_000,
    onUserQuestion: async (request, { signal }) => {
      seen.push(request)
      entered.resolve(signal)
      return answer.promise
    },
  })
  runtime.client = client
  const initialized = await client.initialize({
    cwd: root, provider: 'deepseek-official', model: 'deepseek-v4-pro', capabilities: { cooperativePause: true },
  })
  const runtimeId = initialized.capabilities?.cooperativePause?.runtimeId
  if (runtimeId === undefined) throw new Error('real runtime did not advertise cooperative pause')
  const pid = Number(await readFile(pidPath, 'utf8'))
  expect(Number.isSafeInteger(pid) && pid > 0).toBe(true)
  const subscription = client.subscribe(notification => notification.params.sessionId === 'pause-original')
  const sessionId = 'pause-original'
  const originalId = await client.prompt(sessionId, [{ type: 'text', text: 'Ask which output format to use' }])
  const signal = await entered.promise
  const params = { sessionId, runtimeId, generation: 1, requestId: 'native-original-intent' as PauseRequestId }
  expect(await client.requestPause(params)).toMatchObject({ generation: 1, state: 'pending', scope: 'turn-admission' })
  expect(await client.requestPause(params)).toMatchObject({ generation: 1, state: 'pending' })
  const queuedId = await client.prompt(sessionId, [{ type: 'text', text: 'QUEUED_NEXT_TURN_ONLY' }])
  expect(queuedId).not.toBe(originalId)
  expect(await client.getPauseState(params)).toMatchObject({ generation: 1, state: 'pending' })
  expect(signal.aborted).toBe(false)
  expect(requests).toHaveLength(1)
  expect(seen).toHaveLength(1)
  expect(() => process.kill(pid, 0)).not.toThrow()
  answer.resolve({ answers: [{ id: 'format', selected: ['CSV'] }] })
  await expect.poll(() => client.getPauseState(params), { timeout: 10_000 }).toMatchObject({ generation: 1, state: 'paused' })
  expect(signal.aborted).toBe(false)
  expect(requests).toHaveLength(2)
  expect(JSON.stringify(requests[1]?.messages)).toContain('CSV')
  expect(JSON.stringify(requests[1]?.messages)).not.toContain('QUEUED_NEXT_TURN_ONLY')
  expect(await readFile(pidPath, 'utf8')).toBe(String(pid))
  expect(() => process.kill(pid, 0)).not.toThrow()
  expect(await client.releasePause(params)).toMatchObject({ generation: 1, state: 'released' })
  await client.request('session/wait', { sessionId })
  expect(requests).toHaveLength(3)
  expect(JSON.stringify(requests[2]?.messages)).toContain('QUEUED_NEXT_TURN_ONLY')
  const userIds: unknown[] = []
  const assistantText: string[] = []
  for (let notification; (notification = subscription.tryNext()) !== undefined;) {
    if (notification.method !== 'session.event') continue
    const event = notification.params.event as {
      type: string
      data: { id?: unknown; message?: { content?: { type: string; text?: string }[] } }
    }
    if (event.type === 'user/message') userIds.push(event.data.id)
    if (event.type === 'assistant/message') {
      for (const block of event.data.message?.content ?? []) if (block.type === 'text' && block.text !== undefined) assistantText.push(block.text)
    }
  }
  expect(userIds.filter(id => id === originalId)).toHaveLength(1)
  expect(userIds.filter(id => id === queuedId)).toHaveLength(1)
  expect(assistantText).toContain('original answer consumed')
  expect(assistantText).toContain('queued input consumed')
  expect(seen).toHaveLength(1)
  const successor = { ...params, generation: 2 }
  expect(await client.requestPause(successor)).toMatchObject({ generation: 2, state: 'paused' })
  await expect(client.releasePause(params)).rejects.toThrow('no matching current pause generation')
  expect(await client.getPauseState(params)).toMatchObject({ generation: 2, state: 'paused' })
  expect(await client.releasePause(successor)).toMatchObject({ generation: 2, state: 'released' })
  expect(await readFile(pidPath, 'utf8')).toBe(String(pid))
  expect(() => process.kill(pid, 0)).not.toThrow()
  subscription.close()
})
