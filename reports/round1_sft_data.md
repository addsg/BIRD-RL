# 第一轮 SFT 数据集

## 数据来源

- 教师轨迹范围：TRAIN_0 至 TRAIN_299
- 教师模型：GPT-5.4
- 候选轨迹数量：300
- 通过 Evaluator 的轨迹数量：281
- 因 preprocess 失败而剔除的轨迹数量：8
- 最终采用的轨迹数量：273

## 数据划分

- 训练集：225 条样本，来自 11 个数据库
- 验证集：48 条样本，来自 2 个数据库
- 验证集数据库：movie_3 和 public_review_platform
- 训练集和验证集之间没有数据库重叠

## 数据格式约定

两个 Parquet 文件都只包含一个名为 messages 的字段：

    messages: list<struct<content: string, role: string>>

消息按照以下顺序排列：

    system -> user -> assistant -> user observation -> assistant ...

最后一条 assistant 消息包含 submit_solution。工具返回的 observation 使用 user
角色，因此 veRL MultiTurnSFTDataset 的损失掩码只会训练 assistant 轮次。

## 数据文件

以下文件位于本地 SFT 数据目录（本文记为 SFT_DATA_DIR），数据文件本身不提交到 Git：

- 训练集：SFT_DATA_DIR/train.parquet
- 验证集：SFT_DATA_DIR/validation.parquet
- 数据划分清单：SFT_DATA_DIR/split_manifest.jsonl
- preprocess 后的 schema 快照：SFT_DATA_DIR/train_schema_300.jsonl
- 清理后的轨迹：SFT_DATA_DIR/trajectories_sft_clean.jsonl
- 验证报告：SFT_DATA_DIR/validation_report.json
- 汇总信息：SFT_DATA_DIR/summary.json

## 数据验证

已使用 veRL MultiTurnSFTDataset 加载这些文件，最大序列长度设置为 16384。
所有样本均通过角色顺序、最终提交、教师答案泄漏、schema 和 token 长度检查。

- 训练集最大 token 长度：9452
- 验证集最大 token 长度：8654
- 超过 16384 tokens 的样本数量：0
