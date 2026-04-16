---
title: 文献智读
emoji: 📘
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.45.0
app_file: app.py
pinned: false
---

# 文献智读

《文献智读》是一个面向大学生课程写作场景的学习辅助 Web 工具。当前版本基于 `Python + Streamlit` 实现“上传单篇论文 PDF -> 提取关键信息 -> 生成课程写作辅助内容 -> 导出结果”的闭环，重点服务于课程汇报、课程论文前期整理和文献综述基础框架搭建。

## 当前已实现功能

- 上传单篇 PDF 并提取可复制文本
- 识别论文标题、作者、关键词
- 主摘要优先显示中文摘要；无中文摘要时回退英文摘要
- 提取研究问题、研究方法、核心结论
- 生成课程写作向的 AI 解读内容：
  - 通俗摘要
  - 研究方法说明
  - 创新点分析
  - 不足分析
- 生成三类写作输出：
  - 课程汇报提纲
  - 课程论文提纲
  - 文献综述基础框架
- 导出 `.md` / `.txt` 解析结果
- 在模型服务不可用时自动回退到本地保守生成，保证主流程可运行

## 本次迭代新增内容

- 接通作者字段，并统一进入标准结果、页面展示和导出
- 修正摘要语言选择逻辑，避免英文摘要误顶替主摘要
- 收紧关键词去重与数量控制，并在低置信度场景下增加 LLM 关键词兜底
- 新增课程写作辅助层，基于摘要、结构化字段和正文预览生成更适合大学生使用的解读和提纲
- 结果页升级为三段式阅读布局：
  - 左侧：基础信息卡片
  - 中间：摘要、结构化内容、AI 解读
  - 右侧：写作输出区
- 导出文件同步加入新增字段和三类提纲
- 新增 `.env` 文件支持，无需每次手动设置环境变量
- 新增同会话内存缓存，重复上传同一文件时直接复用解析结果，跳过 LLM 调用
- 新增文件大小校验，超过 20 MB 时给出明确提示
- 调试日志统一改用 `logging` 模块，不再输出到标准输出

## 项目结构

```text
literature-smart-reader/
├─ app.py
├─ requirements.txt
├─ README.md
├─ assets/
│  └─ styles.css
├─ config/
│  └─ settings.py
├─ models/
│  └─ paper_result.py
├─ services/
│  ├─ document_parse_service.py
│  ├─ export_service.py
│  ├─ llm_service.py
│  ├─ metadata_service.py
│  ├─ paper_parse_service.py
│  ├─ pdf_service.py
│  ├─ structure_service.py
│  ├─ structured_rewrite_service.py
│  └─ summary_service.py
├─ tests/
│  └─ test_keyword_extraction.py
├─ utils/
│  ├─ session.py
│  └─ text_utils.py
└─ views/
   ├─ home_view.py
   ├─ result_view.py
   └─ upload_view.py
```

## 运行方式

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 可选：配置中转站环境变量

如果你希望启用结构化重写、关键词兜底和课程写作辅助的 LLM 增强层，复制 `.env.example` 为 `.env` 并填入实际值：

```bash
cp .env.example .env
```

```ini
RELAY_API_KEY=你的中转站密钥
RELAY_BASE_URL=你的中转站兼容地址
RELAY_MODEL=你的模型名
```

未配置时，系统会自动回退为本地规则和保守生成结果，不影响主流程使用。

3. 启动应用

```bash
streamlit run app.py
```

4. 打开浏览器中的本地地址，依次完成：

- 首页进入
- 上传 PDF
- 开始解析
- 查看结果页与导出结果

## 结果页说明

结果页采用三段式布局，适合课堂展示截图：

- 左侧：展示标题、作者、关键词、文件名、摘要语言、解析状态和解析提示
- 中间：展示主摘要、通俗摘要、研究方法说明、研究问题/方法/结论，以及创新点和不足分析
- 右侧：展示课程汇报提纲、课程论文提纲和文献综述基础框架

## 当前限制

- 当前版本仍以单篇、可复制文本的 PDF 为主，文件大小不超过 20 MB
- 扫描版 PDF 未集成 OCR，文本过少时可能解析失败
- 作者、关键词和结构化字段仍受原始版式质量影响，建议结合原文核对
- 内存缓存仅在同一会话内有效，关闭浏览器后缓存清空
- 多文献对比、批量处理和数据库持久化暂未纳入当前版本

## 测试

当前测试位于 `tests/test_keyword_extraction.py`，覆盖：

- 关键词显式标记区提取
- 关键词去重与数量控制
- 中文摘要优先于英文摘要
- 作者字段进入标准结果
- 课程写作辅助在信息不足时的稳定 fallback

如果环境未安装 `pytest`，可以直接运行：

```bash
python -m unittest discover -s tests -v
```
