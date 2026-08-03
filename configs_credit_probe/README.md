# Credit-Card probe only

本目录保留一次单数据集方向性探测所需的配置和 model policy。它是辅助研究材料，不属于正式 ICOMP 2026 协议：

- 只覆盖 `credit_card_fraud`，不是六个正式数据集范围；
- pipeline YAML 缺少当前正式 schema 的部分字段和 stage scopes；
- `model_policy.json` 不能传给正式或 smoke campaign；
- 本目录中的输出不得并入正式汇总或论文。

正式运行必须使用 `../configs/`，并在新的 bootstrap run 中重新生成完整 model policy。
