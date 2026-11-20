# Auralis Multimodal Annotation

专业视听多模态数据清洗与标注平台，覆盖分割、检测、跟踪、音视频对齐、字幕与二次调整。

## 运行完整接口版

数据集默认位于 `G:\\avcs`。在 PowerShell 中运行：

\`\`\`powershell
.\\start_auralis.ps1
\`\`\`

然后打开 <http://127.0.0.1:8765>。该地址同时提供前端和真实 HTTP API，可直接读取本机数据集。

也可手动启动：

\`\`\`powershell
python server.py --dataset-root G:\\avcs --port 8765
\`\`\`

## 接口

- `POST /api/v1/models/detection/infer`
- `POST /api/v1/models/segmentation/infer`
- `POST /api/v1/models/tracking/infer`
- `POST /api/v1/models/alignment/infer`
- `POST /api/v1/models/caption/infer`
- `POST /api/v1/pipeline/run`
- `POST /api/v1/annotations/save`
- `GET /api/v1/openapi.json`

当前适配器使用 `dataset_replay` 模式，不加载模型权重；输出来自 AVCS 帧、音频和逐帧 caption 标注，并通过 `provenance` 字段声明来源。详见 [API.md](./API.md)。

## 公开界面演示

https://yixuan-wang1.github.io/auralis-multimodal-annotation/

GitHub Pages 只承载静态前端，无法直接读取浏览者电脑的 `G:\\avcs`。完整接口联调请使用本地服务地址。
