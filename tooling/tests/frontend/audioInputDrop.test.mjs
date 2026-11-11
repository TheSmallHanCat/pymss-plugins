import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'
import { ref } from 'vue'
import { sameAudioDirectory } from '../../frontend/src/features/audio-tools/paths.ts'

const source = ts.createSourceFile('drop.ts', readFileSync(new URL('../../frontend/src/features/audio-tools/useAudioInputDrop.ts', import.meta.url), 'utf8'), ts.ScriptTarget.Latest, true)
const code = ts.transpileModule(source.statements.filter(statement => !ts.isImportDeclaration(statement)).map(statement => statement.getText(source)).join('\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function host({ mode, scan = async args => ({ files: args.paths, warnings: [] }) } = {}) {
  const exports = {}, accepted = [], calls = [], warnings = [], errors = []
  let handler, cleanup, removed = false, disabled = false, inside = true
  const target = {}
  vm.runInNewContext(code, { exports, ref, sameAudioDirectory, Error, watch() {},
    onBeforeUnmount(fn) { cleanup = fn }, useI18n: () => ({ t: key => key }),
    useMessage: () => ({ warning: message => warnings.push(message), error: message => errors.push(message) }),
    onFileDrop(fn) { handler = fn; return () => { removed = true } },
    invoke(_command, args) { calls.push(args); return scan(args) },
    document: { elementFromPoint: () => inside ? target : {} },
  })
  const result = exports.useAudioInputDrop({ mode, disabled: () => disabled,
    accept: (files, paths) => accepted.push({ files: Array.from(files), paths: Array.from(paths) }),
  })
  result.dropZone.value = { contains: el => el === target }
  return { result, accepted, calls, warnings, errors, removed: () => removed,
    send(event) { return handler({ position: { x: 20, y: 20 }, ...event }) },
    disable() { disabled = true }, outside() { inside = false }, cleanup: () => cleanup(),
  }
}

test('hover, drop and leave track only the active input zone', async () => {
  const h = host(); await h.send({ type: 'over' }); assert.equal(h.result.dragging.value, true)
  await h.send({ type: 'leave' }); assert.equal(h.result.dragging.value, false)
  h.outside(); await h.send({ type: 'drop', paths: ['audio.wav'] }); assert.equal(h.calls.length, 0)
})

test('file and folder drops are scanned, deduplicated and batched within SDK limits', async () => {
  const h = host({ scan: async args => ({ files: [...args.paths, args.paths[0]], warnings: ['skipped'] }) })
  const paths = Array.from({ length: 257 }, (_, i) => `/audio/${i}.wav`)
  await h.send({ type: 'drop', paths: [...paths, paths[0]] })
  assert.deepEqual(h.calls.map(call => call.paths.length), [256, 1])
  assert.ok(h.calls.every(call => call.recursive === true))
  assert.deepEqual(h.accepted[0].files, paths)
  assert.equal(h.warnings[0], 'tools.dropWarnings')
  assert.equal(h.result.scanning.value, false)
})

test('single-file inputs reject multiple files and unsupported drops', async () => {
  const multi = host({ mode: 'file' }); await multi.send({ type: 'drop', paths: ['a.wav', 'b.wav'] })
  assert.equal(multi.accepted.length, 0); assert.equal(multi.warnings[0], 'tools.dropSingle')
  const empty = host({ scan: async () => ({ files: [], warnings: [] }) }); await empty.send({ type: 'drop', paths: ['a.txt'] })
  assert.equal(empty.accepted.length, 0); assert.equal(empty.warnings[0], 'tools.dropNoAudio')
})

test('directory inputs accept an audio folder and reject a file or multiple folders', async () => {
  const folder = host({ mode: 'directory', scan: async () => ({ files: ['C:/Audio/a.wav'], warnings: [] }) })
  await folder.send({ type: 'drop', paths: ['C:\\Audio'] })
  assert.deepEqual(folder.accepted[0].paths, ['C:\\Audio'])
  assert.equal(folder.calls[0].recursive, false)
  const file = host({ mode: 'directory' }); await file.send({ type: 'drop', paths: ['C:/Audio/a.wav'] })
  assert.equal(file.warnings[0], 'tools.dropFolder')
  const multiple = host({ mode: 'directory' }); await multiple.send({ type: 'drop', paths: ['/a', '/b'] })
  assert.equal(multiple.calls.length, 0)
})

test('directory drops preserve Windows drive roots and accept native path variants', async () => {
  for (const [directory, file] of [
    ['C:\\', 'C:\\song.wav'],
    ['c:/', 'C:/song.wav'],
    ['C:\\', 'c:/song.wav'],
    ['C:\\Audio', 'c:\\audio\\song.wav'],
    ['\\\\server\\share\\', '\\\\SERVER\\SHARE\\song.wav'],
    ['/', '/song.wav'],
  ]) {
    const h = host({ mode: 'directory', scan: async () => ({ files: [file], warnings: [] }) })
    await h.send({ type: 'drop', paths: [directory] })
    assert.deepEqual(h.accepted, [{ files: [file], paths: [directory] }], directory)
    assert.deepEqual(h.warnings, [], directory)
    assert.equal(h.calls[0].recursive, false)
  }
})

test('directory drops require audio directly inside the selected directory', async () => {
  for (const [directory, files] of [
    ['C:\\', []],
    ['C:\\', ['C:\\Audio\\song.wav']],
    ['C:\\', ['D:\\song.wav']],
    ['C:\\song.wav', ['C:\\song.wav']],
  ]) {
    const h = host({ mode: 'directory', scan: async () => ({ files, warnings: [] }) })
    await h.send({ type: 'drop', paths: [directory] })
    assert.deepEqual(h.accepted, [], directory)
    assert.deepEqual(h.warnings, ['tools.dropFolder'], directory)
  }
})

test('busy or disposed inputs ignore drops and late scans, without leaking listeners', async () => {
  const busy = host(); busy.disable(); await busy.send({ type: 'drop', paths: ['a.wav'] }); assert.equal(busy.calls.length, 0)
  let resolve
  const h = host({ scan: () => new Promise(done => { resolve = done }) })
  const pending = h.send({ type: 'drop', paths: ['a.wav'] }); assert.equal(h.result.scanning.value, true)
  h.cleanup(); resolve({ files: ['a.wav'], warnings: [] }); await pending
  assert.equal(h.accepted.length, 0); assert.equal(h.removed(), true)
})

test('failed and oversized scans leave inputs unchanged and report a useful message', async () => {
  const failed = host({ scan: async () => { throw new Error('Read failed') } })
  await failed.send({ type: 'drop', paths: ['a.wav'] }); assert.deepEqual(failed.errors, ['Read failed'])
  assert.equal(failed.accepted.length, 0); assert.equal(failed.result.scanning.value, false)
  const large = host(); await large.send({ type: 'drop', paths: Array.from({ length: 4097 }, (_, i) => `${i}.wav`) })
  assert.equal(large.warnings[0], 'tools.dropLimit'); assert.equal(large.calls.length, 0)
})
