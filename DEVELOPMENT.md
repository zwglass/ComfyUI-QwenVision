# ComfyUI-QwenVision 开发规范（面向 openclaw 自动开发）

本文档参考 `kijai/ComfyUI-KJNodes` 的工程化方式，并结合本仓库的最小可用结构制定。目标是让 openclaw 在无人值守场景下稳定地产生可合并改动。

## 1. 目标与边界

- 目标: 维护一个稳定、可复用的 ComfyUI Qwen Vision 自定义节点插件。
- 边界: 本仓库只负责节点定义、推理流程、模型缓存与图像预处理，不承担 Web 服务或训练逻辑。
- 原则: 小步提交、最小改动面、可验证输出、可回滚。

## 2. 目录职责（必须遵守）

- `__init__.py`
  - 仅导出 `NODE_CLASS_MAPPINGS` 与 `NODE_DISPLAY_NAME_MAPPINGS`。
- `nodes/`
  - 节点接口层: `INPUT_TYPES`、`RETURN_TYPES`、`FUNCTION`、`CATEGORY`、节点调度。
  - 每个节点独立文件，避免单文件膨胀。
- `qwenvision/cache_manager.py`
  - GGUF/mmproj 源解析、下载、缓存、卸载。
  - 必须线程安全（当前使用 `Lock`）。
- `qwenvision/inference.py`
  - `llama-mtmd-cli` 子进程推理封装、超时控制、输出解析。
- `qwenvision/image_utils.py`
  - ComfyUI `IMAGE` 到 PIL 的转换与格式兜底。
- `dev_doc/`
  - 设计草稿、开发提纲、实验记录，不作为运行时依赖。

## 3. 代码风格与静态检查（参考 KJNodes）

统一采用以下约束（与 KJNodes 习惯对齐）:

- 格式化: `black`，`line-length = 79`
- import 排序: `isort`（`profile = "black"`，`line_length = 79`）
- Lint: `ruff`，至少启用:
  - `E`, `F`, `W`, `I`
  - 可按需忽略 `E501`（行长）以降低自动修复冲突
- Python 环境与依赖管理: 统一使用 `uv`（禁止使用裸 `pip` 直接改全局环境）

推荐命令:

```bash
uv run black .
uv run isort .
uv run ruff check .
```

常用环境命令:

```bash
uv venv
uv sync
```

## 4. 节点接口规范

- 新增节点必须同时更新:
  - `NODE_CLASS_MAPPINGS`
  - `NODE_DISPLAY_NAME_MAPPINGS`
- 节点 `CATEGORY` 统一使用 `Qwen/Vision`（如需扩展，先在 PR 描述说明理由）。
- `INPUT_TYPES` 中:
  - 所有参数必须提供合理默认值与范围。
  - 可选参数放入 `optional`，避免破坏已有 workflow。
- 返回值约定:
  - 用户可见异常优先转为字符串状态返回，避免直接中断 ComfyUI 图。

## 5. 推理与缓存规范

- 模型加载策略:
  - 仅支持 GGUF + mmproj 配套加载。
  - 远程源必须下载到 `ComfyUI/models/qwenvision/` 后再使用。
- 缓存键必须包含:
  - `model_source|mmproj_source|cli_path`
- 推理统一通过 `llama-mtmd-cli` 执行，不直接在节点内加载 Transformers 模型。
- 第一阶段默认单图推理；批量逻辑需单独设计，不得隐式改变现有行为。

## 6. openclaw 自动开发流程（强制）

每次自动开发任务按以下顺序执行:

1. 读取任务并定位影响文件。
2. 仅改动必要文件，避免无关格式化噪音。
3. 优先保持向后兼容:
   - 不随意修改节点名、返回类型、参数名。
4. 完成后执行最小验证:
   - 语法检查与导入检查。
5. 输出结构化结果:
   - 改动文件列表
   - 行为变化说明
   - 验证命令与结果
   - 风险与后续建议

## 7. 提交前最小验证清单

至少执行:

```bash
PYTHONPYCACHEPREFIX=.pycache uv run python -m compileall .
PYTHONPYCACHEPREFIX=.pycache uv run python -m py_compile __init__.py nodes/*.py qwenvision/*.py
```

如环境已安装开发工具，再执行:

```bash
uv run ruff check .
uv run black --check .
uv run isort --check-only .
```

## 8. 变更分级规则

- 小改动:
  - 注释、文档、错误文案、非接口内部重构。
- 中改动:
  - 新增可选参数、新增节点但不影响旧节点。
- 大改动:
  - 修改节点输入输出契约、缓存键策略、模型加载主路径。
  - 必须补充迁移说明和兼容策略。

## 9. 禁止事项

- 禁止在节点文件中堆积大型业务逻辑（保持“每节点单职责”）。
- 禁止引入未使用依赖。
- 禁止无验证直接提交影响模型加载路径的改动。
- 禁止将调试临时代码（如硬编码模型路径）保留在主分支。

## 10. PR 描述模板（openclaw 输出可复用）

```text
## Summary
- What changed:
- Why:

## Files
- path/to/file.py: brief

## Validation
- command: result

## Compatibility
- Breaking change: yes/no
- Migration needed: yes/no
```
