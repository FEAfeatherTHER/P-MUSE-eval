# Task 2 报告：文件信息缓存

## 实现

- 缓存记录保留 WAV、MIDI、checkpoint 的路径，并为每个文件保存
  `size_bytes` 与 `mtime_ns`。
- 转录配置保存为 JSON 兼容的完整快照；每次按实际值比较，不依赖对象身份。
- 缓存命中同时要求路径、三个文件信息和配置快照一致；任一 WAV、MIDI、checkpoint 或配置变化都会重新转录。
- 旧式、缺少文件信息的缓存记录不会命中；成功转录后会写入新记录。
- onset/offset 的转录证据检查继续要求 `ok` 或 `cached` 状态和缓存记录，并重新核对当前 WAV/MIDI 文件信息。
- 保持 `cache_index.jsonl`、`transcription_status.jsonl` 以及既有 onset/offset 接口和回调兼容行为。

## 文件

- `pmuse_eval/cache.py`
- `pmuse_eval/transcription.py`
- `pmuse_eval/onset.py`
- `tests/test_transcription.py`

## RED 证据

命令：

```bash
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/P-MUSE-eval/bin/python -m unittest tests.test_transcription -v
```

初始结果：7 个测试中 1 个迁移覆盖失败。旧实现写出的记录没有
`midi_file_info`，符合新增行为尚未实现的预期。

## GREEN 证据

定向测试命令与结果：

```bash
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/P-MUSE-eval/bin/python -m unittest tests.test_transcription -v
```

结果：7/7 通过，覆盖缓存命中、WAV/MIDI/checkpoint/配置变化、旧记录替换，及评估证据的 WAV/MIDI 复核。

完整测试命令与结果：

```bash
CUDA_VISIBLE_DEVICES= /root/miniconda3/envs/P-MUSE-eval/bin/python -m unittest discover -s tests -v
```

结果：23/23 通过。

静态检查：已扫描所有源码与测试文件，未发现已移除的摘要实现、相关导入或旧字段名称；`git diff --check` 无格式错误。
