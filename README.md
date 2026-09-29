# reversible-matrix-assembly

本地稀疏矩阵装配引擎（纯 Python 标准库实现，无任何第三方运行依赖）。

多个计算模块以**唯一批次 ID** 提交 `(row, col, value)` 三元组贡献：

- 行列在创建时给定正整数，之后固定不变；
- 行、列从零开始编号；
- 同批次内与跨批次的重复坐标全部累加；
- 支持新增（`Add`）、整批替换（`Replace`）、撤销（`Revoke`）；
- 多项操作可放入一个**原子事务**，按顺序解释，全部成功才提交；
- 版本号从 0 开始，每个成功的非空事务只 +1，空事务不改版本；
- 新增已存在 ID、替换/撤销不存在 ID、坐标越界、非有限数值都会让整个事务失败，
  矩阵、批次记录和版本保持不变（先在工作副本上执行，最后一次性发布）；
- 查询返回只读快照（版本 + 批次清单）、按 `(行, 列)` 排序的非零坐标和矩阵向量乘法；
- 可配置有限、非负的消零阈值：`|最终值| <= threshold` 的条目不参与读出和乘法，
  但批次贡献不会丢失，替换/撤销后剩余贡献会重新体现；
- 旧快照不受后续提交影响（快照隔离）；向量长度不匹配抛出 `AssemblyError` 且不改状态。

## 目录结构

- `sparse_assembly/engine.py` — 可复用引擎接口（`SparseAssembler`、`Add/Replace/Revoke`、`Snapshot`、`Entry`、`AssemblyError`）
- `demo.py` — 自结束的命令行业务演示，打印输入、操作、结果与断言，失败时退出码非零
- `tests/` — `unittest` 自动化测试
- `atk-demo.json` — 服务器录屏演示配置（`kind=terminal`，无 setup，直接运行 demo）
- `requirements.txt` — 空文件：无需安装任何第三方包（主要第三方依赖数 = 0，<= 2）

## 快速开始

```bash
python3 demo.py
```

演示在 1 秒内自行结束，退出码为 0 表示所有断言通过。

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

当前共 50 个测试，覆盖：装配与乘法（对照稠密参考实现）、批次替换/撤销、
事务回滚（重复 ID、未知 ID、越界、负数/非有限数值、格式错误）、
阈值抵消与替换/撤销后贡献重现、快照隔离、非法构造/向量输入，以及 demo 子进程集成测试。

## 接口示例

```python
from sparse_assembly import SparseAssembler, Add, Replace, Revoke

engine = SparseAssembler(nrows=4, ncols=4, threshold=1e-12)

# 单个事务包含多个操作，原子提交；版本只增加一次
snap = engine.transaction([
    Add("stiffness", [(0, 0, 2.0), (0, 1, -1.0), ...]),
    Add("load", [(0, 0, 1.0), (3, 3, 1.0)]),
])

snap.version            # 1
snap.batch_ids          # ('stiffness', 'load')
snap.nonzeros()         # 按 (row, col) 排序的 Entry 元组（已应用阈值）
snap.matvec([1.0]*4)    # 矩阵向量乘法
snap.batch("load")      # 批次原始贡献（不受阈值影响）

engine.replace_batch("load", [(1, 1, 4.0)])
engine.revoke_batch("load")
engine.transaction([])  # 空事务：成功但版本不变
```

## 服务器演示（atk-demo.json）

```json
{"kind": "terminal", "setup": [], "start": ["{python}", "demo.py"]}
```

无需构建与依赖安装；`start` 是命令参数数组（不是 shell 字符串），
程序自行结束、不启动常驻服务，因此无需额外清理。
