import { onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMessage } from 'naive-ui'
import { invoke, onFileDrop } from '@/sdk'
import { sameAudioDirectory } from './paths'

type ScanResult = { files: string[]; warnings: string[] }

export function useAudioInputDrop(options: {
  disabled: () => boolean
  mode?: 'files' | 'file' | 'directory'
  accept: (files: string[], paths: string[]) => void | Promise<void>
}) {
  const { t } = useI18n()
  const message = useMessage()
  const dropZone = ref<HTMLElement | null>(null)
  const dragging = ref(false)
  const scanning = ref(false)
  let disposed = false

  const remove = onFileDrop(async event => {
    dragging.value = false
    if (disposed || event.type === 'leave' || options.disabled() || scanning.value) return
    const target = document.elementFromPoint(event.position.x, event.position.y)
    if (!target || !dropZone.value?.contains(target)) return
    if (event.type !== 'drop') { dragging.value = true; return }
    const paths = [...new Set(event.paths || [])]
    if (!paths.length) return
    if (paths.length > 4096) { message.warning(t('tools.dropLimit', { count: 4096 })); return }
    if (options.mode === 'directory' && paths.length !== 1) { message.warning(t('tools.dropFolder')); return }
    scanning.value = true
    try {
      const files = new Set<string>()
      let warnings = 0
      for (let index = 0; index < paths.length; index += 256) {
        const result = await invoke<ScanResult>('scan_audio_paths_with_options', {
          paths: paths.slice(index, index + 256), recursive: options.mode !== 'directory', sortFiles: true,
        })
        if (disposed || options.disabled()) return
        for (const file of result.files || []) files.add(file)
        warnings += result.warnings?.length || 0
      }
      const selected = [...files]
      if (options.mode === 'directory') {
        const containsAudio = selected.some(file => {
          const normalized = /^(?:[a-z]:[\\/]|\\\\|\/\/)/i.test(file) ? file.replace(/\\/g, '/') : file
          return sameAudioDirectory(normalized.slice(0, normalized.lastIndexOf('/') + 1) || '/', paths[0]!)
        })
        if (!containsAudio) { message.warning(t('tools.dropFolder')); return }
      } else {
        if (!selected.length) { message.warning(t('tools.dropNoAudio')); return }
        if (options.mode === 'file' && selected.length !== 1) { message.warning(t('tools.dropSingle')); return }
      }
      await options.accept(selected, paths)
      if (!disposed && warnings) message.warning(t('tools.dropWarnings', { count: warnings }))
    } catch (error) {
      if (!disposed) message.error(error instanceof Error ? error.message : String(error))
    } finally {
      if (!disposed) scanning.value = false
    }
  })
  watch(options.disabled, () => { dragging.value = false })
  onBeforeUnmount(() => { disposed = true; remove(); dragging.value = false })
  return { dropZone, dragging, scanning }
}
