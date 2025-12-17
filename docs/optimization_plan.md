# DeepMT 项目优化计划

## 📋 优化目标

基于用户反馈，进行以下三个方面的优化：

1. **重新定义用户使用方式和结果**
2. **IR对用户隐藏，作为内部表示**
3. **MR生成与测试分离，MR可持久化**

---

## 🎯 优化1：重新定义用户使用方式

### 当前问题
- 用户需要手动创建IR对象
- 用户需要了解IR的内部结构
- 使用方式复杂，需要了解多个组件

### 目标设计

#### 用户视角的使用方式

**方式1：命令行工具**
```bash
# 测试PyTorch的Add算子
deepmt test operator --name Add --inputs 1.0 2.0 --framework pytorch

# 测试多个算子
deepmt test operators --config tests/operators.yaml --framework pytorch

# 测试模型
deepmt test model --path model.py --framework pytorch
```

**方式2：Python API**
```python
from deepmt import DeepMT

# 创建测试客户端
client = DeepMT()

# 测试算子
result = client.test_operator(
    name="Add",
    inputs=[1.0, 2.0],
    framework="pytorch"
)

# 查看结果
print(result.summary())  # 测试摘要
print(result.report())    # 详细报告
result.save_report("report.html")  # 保存报告
```

**方式3：配置文件**
```yaml
# config.yaml
tests:
  - type: operator
    name: Add
    inputs: [1.0, 2.0]
    framework: pytorch
  - type: operator
    name: Multiply
    inputs: [3.0, 4.0]
    framework: pytorch
```

### 用户看到的结果

1. **测试摘要**（控制台输出）
   ```
   ========================================
   DeepMT 测试结果
   ========================================
   算子: Add
   框架: PyTorch
   总测试数: 3
   通过: 3
   失败: 0
   耗时: 0.12s
   ========================================
   ```

2. **详细报告**（HTML/PDF）
   - 每个MR的测试结果
   - 失败的MR详情
   - 缺陷分类和统计
   - 可视化图表

3. **数据库存储**
   - 所有测试结果自动保存
   - 支持查询历史结果
   - 支持对比不同框架的结果

---

## 🔒 优化2：IR对用户隐藏

### 当前问题
- 用户需要手动创建`OperatorIR`对象
- 用户需要了解IR的数据结构

### 目标设计

#### IR自动生成层

**用户输入** → **IR转换器** → **IR对象**（内部）

```python
# 用户只需要提供简单的输入
user_input = {
    "name": "Add",
    "inputs": [1.0, 2.0],
    "framework": "pytorch"
}

# 系统内部自动转换为IR
ir = IRConverter.from_user_input(user_input)
# 用户不需要知道IR的存在
```

#### 实现方案

1. **创建用户输入转换器** (`ir/converter.py`)
   - `from_operator_name()` - 从算子名称创建IR
   - `from_framework_code()` - 从框架代码创建IR
   - `from_config()` - 从配置文件创建IR

2. **创建用户友好的API层** (`api/`)
   - `DeepMT` 类作为主入口
   - 隐藏所有IR相关细节
   - 提供简洁的方法接口

---

## 💾 优化3：MR生成与测试分离

### 当前问题
- MR生成和测试耦合在一起
- 每次测试都要重新生成MR
- MR无法重用和共享

### 目标设计

#### MR知识库系统

**MR生成阶段**（独立）
```python
# 生成MR并保存
mr_generator = OperatorMRGenerator()
mrs = mr_generator.generate(operator_ir)
mr_repository.save(operator_name="Add", mrs=mrs)
```

**MR测试阶段**（重用）
```python
# 从知识库加载MR
mrs = mr_repository.load(operator_name="Add")
# 直接使用已生成的MR进行测试
scheduler.run_with_mrs(ir, mrs, framework)
```

#### 实现方案

1. **MR知识库** (`mr_generator/mr_repository.py`)
   - 保存MR到数据库/文件
   - 支持按算子名称查询
   - 支持MR版本管理

2. **分离的测试执行器** (`core/test_runner.py`)
   - 接收IR和MR列表
   - 执行测试
   - 不负责MR生成

3. **MR管理工具**
   - 批量生成MR
   - MR知识库维护
   - MR有效性验证

---

## 📝 详细实现计划

### 阶段1：用户API层（优先级最高）

#### 1.1 创建用户友好的API
- [ ] 创建 `api/deepmt.py` - 主API类
- [ ] 实现 `test_operator()` 方法
- [ ] 实现 `test_model()` 方法
- [ ] 实现结果报告方法

#### 1.2 IR自动转换
- [ ] 扩展 `ir/converter.py`
- [ ] 实现 `from_operator_name()`
- [ ] 实现 `from_user_input()`

#### 1.3 命令行工具
- [ ] 创建 `cli/` 目录
- [ ] 实现 `deepmt` 命令
- [ ] 支持配置文件

### 阶段2：MR知识库系统

#### 2.1 MR知识库
- [ ] 创建 `mr_generator/mr_repository.py`
- [ ] 实现MR保存/加载
- [ ] 支持数据库存储

#### 2.2 MR生成工具
- [ ] 创建 `scripts/generate_mrs.py`
- [ ] 批量生成MR
- [ ] MR验证和测试

#### 2.3 测试执行器分离
- [ ] 创建 `core/test_runner.py`
- [ ] 修改 `TaskScheduler` 支持预生成MR
- [ ] 实现MR缓存机制

### 阶段3：结果展示优化

#### 3.1 报告生成
- [ ] 实现 `analysis/report_generator.py`
- [ ] HTML报告模板
- [ ] 可视化图表

#### 3.2 结果查询API
- [ ] 实现结果查询接口
- [ ] 支持历史结果对比
- [ ] 支持多框架结果对比

---

## 🏗️ 架构变化

### 优化前
```
用户 → 手动创建IR → MRGenerator → Scheduler → 结果
```

### 优化后
```
用户 → DeepMT API → IR转换器 → IR（隐藏）
                              ↓
                    MR知识库 ← MRGenerator（独立）
                              ↓
                    TestRunner → 结果 → 报告
```

---

## 📊 实施优先级

1. **P0（立即）**：用户API层 + IR自动转换
2. **P1（本周）**：MR知识库系统
3. **P2（本月）**：命令行工具 + 报告优化

---

## 🎯 成功标准

1. 用户可以在3行代码内完成测试
2. 用户不需要了解IR的存在
3. MR可以独立生成和重用
4. 测试结果清晰易懂

