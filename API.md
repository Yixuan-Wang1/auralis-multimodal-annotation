# Auralis 数据集回放 API

```powershell
python server.py --dataset-root G:\avcs --port 8765
```

打开 `http://127.0.0.1:8765`。接口为真实 HTTP JSON 契约，但当前不加载模型权重，输出来自 AVCS 的帧、音频与逐帧 caption 标注。

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/v1/health` | 服务与数据集状态 |
| GET | `/api/v1/dataset/stats` | 数据集统计 |
| GET | `/api/v1/samples?split=train&limit=12` | 样本列表 |
| POST | `/api/v1/models/detection/infer` | 检测 |
| POST | `/api/v1/models/segmentation/infer` | 分割 |
| POST | `/api/v1/models/tracking/infer` | 跟踪 |
| POST | `/api/v1/models/alignment/infer` | 音视频对齐 |
| POST | `/api/v1/models/caption/infer` | 字幕 |
| POST | `/api/v1/pipeline/run` | 完整流水线 |
| POST | `/api/v1/annotations/save` | 保存二次调整 |
| GET | `/api/v1/openapi.json` | OpenAPI 描述 |

所有推理响应均包含 `mode: dataset_replay` 与 `provenance`。未来接入真实模型时，只需替换 `server.py` 中对应 adapter 函数。

