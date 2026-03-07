# DeepMT 环境变量文档

本文档列出了 DeepMT 项目支持的所有环境变量及其说明。

## 配置相关

### DEEPMT_CONFIG_PATH

**说明**: 指定配置文件的路径

**类型**: 字符串（文件路径或目录路径）

**默认值**: 无（使用默认配置文件查找顺序）

**示例**:
```bash
# 指定配置文件
export DEEPMT_CONFIG_PATH="/path/to/config.yaml"

# 指定配置目录（将查找该目录下的 config.yaml）
export DEEPMT_CONFIG_PATH="/path/to/config/dir"
```

**说明**:
- 如果指定的是文件路径，将直接使用该文件
- 如果指定的是目录路径，将在该目录下查找 `config.yaml`
- 如果未设置此环境变量，系统将按以下顺序查找配置文件：
  1. 当前工作目录的 `config.yaml`
  2. 项目根目录的 `config.yaml`
  3. 用户配置目录（`~/.config/deepmt/config.yaml`）

---

## 日志相关

### DEEPMT_LOG_DIR

**说明**: 指定日志文件存储目录

**类型**: 字符串（目录路径）

**默认值**: `data/logs`

**示例**:
```bash
export DEEPMT_LOG_DIR="/var/log/deepmt"
```

**说明**:
- 日志文件将以 `deepmt_YYYYMMDD.log` 格式命名
- 日志文件会按天自动轮转，保留最近 14 天的日志
- 如果目录不存在，系统会自动创建

---

### DEEPMT_LOG_LEVEL

**说明**: 设置日志输出级别

**类型**: 字符串（枚举值）

**可选值**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**默认值**: `INFO`

**示例**:
```bash
# 开发环境：输出详细调试信息
export DEEPMT_LOG_LEVEL="DEBUG"

# 生产环境：只输出重要信息
export DEEPMT_LOG_LEVEL="WARNING"
```

**说明**:
- `DEBUG`: 输出所有日志，包括详细的调试信息
- `INFO`: 输出一般信息、警告和错误
- `WARNING`: 只输出警告和错误
- `ERROR`: 只输出错误和严重错误
- `CRITICAL`: 只输出严重错误
- 此设置仅影响终端输出，文件日志始终记录 DEBUG 级别及以上的所有日志

---

### DEEPMT_LOG_CONSOLE_STYLE

**说明**: 设置终端日志输出样式

**类型**: 字符串（枚举值）

**可选值**: `colored`, `file`

**默认值**: `colored`

**示例**:
```bash
# 使用彩色输出（默认）
export DEEPMT_LOG_CONSOLE_STYLE="colored"

# 使用与文件日志相同的格式（纯文本，无颜色和图标）
export DEEPMT_LOG_CONSOLE_STYLE="file"
```

**说明**:
- `colored`: 终端输出使用彩色格式，包含图标和颜色标记，便于快速识别日志类型
  - 格式示例: `🚀 INIT    | Loaded config from: /path/to/config.yaml`
- `file`: 终端输出使用与文件日志相同的详细格式，包含时间戳、日志级别、模块名、文件名和行号
  - 格式示例: `2026-01-23 20:00:00 INFO [ConfigLoader] config_loader.py:134 - [INIT] Loaded config from: /path/to/config.yaml`
- 推荐在开发环境使用 `colored`，在生产环境或需要日志分析时使用 `file`

---

## 第三方服务相关

### OPENAI_API_KEY

**说明**: OpenAI API 密钥（或兼容 OpenAI API 的服务密钥）

**类型**: 字符串

**默认值**: 无（必须在配置文件或环境变量中提供）

**示例**:
```bash
export OPENAI_API_KEY="sk-..."
```

**说明**:
- 此环境变量为标准 OpenAI SDK 环境变量
- 如果同时在配置文件和环境变量中设置，配置文件中的值优先
- 支持兼容 OpenAI API 格式的其他服务（如百度千帆、文心一言等）

---

## XDG Base Directory 规范

### XDG_CONFIG_HOME

**说明**: XDG 配置目录（标准 Linux 环境变量）

**类型**: 字符串（目录路径）

**默认值**: `~/.config`

**示例**:
```bash
export XDG_CONFIG_HOME="$HOME/.config"
```

**说明**:
- 这是 XDG Base Directory 规范定义的标准环境变量
- DeepMT 会在 `$XDG_CONFIG_HOME/deepmt/` 目录下查找配置文件
- 通常不需要手动设置，使用系统默认值即可

---

## 使用示例

### 开发环境配置

```bash
# 开发环境：详细日志 + 彩色输出
export DEEPMT_LOG_LEVEL="DEBUG"
export DEEPMT_LOG_CONSOLE_STYLE="colored"
export DEEPMT_LOG_DIR="./logs"
```

### 生产环境配置

```bash
# 生产环境：关键日志 + 文件格式输出
export DEEPMT_LOG_LEVEL="WARNING"
export DEEPMT_LOG_CONSOLE_STYLE="file"
export DEEPMT_LOG_DIR="/var/log/deepmt"
export DEEPMT_CONFIG_PATH="/etc/deepmt/config.yaml"
```

### Docker 环境配置

```dockerfile
# Dockerfile
ENV DEEPMT_LOG_LEVEL=INFO
ENV DEEPMT_LOG_CONSOLE_STYLE=file
ENV DEEPMT_LOG_DIR=/app/logs
```

### 临时调试

```bash
# 临时启用调试模式运行
DEEPMT_LOG_LEVEL=DEBUG python main.py
```

---

## 注意事项

1. **环境变量优先级**: 环境变量的优先级高于配置文件中的设置
2. **大小写**: 环境变量名称区分大小写，必须使用大写
3. **路径**: 所有路径支持相对路径和绝对路径，支持 `~` 展开为用户主目录
4. **日志级别**: 文件日志始终记录所有级别的日志，`DEEPMT_LOG_LEVEL` 仅影响终端输出
5. **配置重载**: 修改环境变量后需要重启应用程序才能生效
