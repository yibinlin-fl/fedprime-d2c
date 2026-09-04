# P3-A Artifact Availability After Clean Completion

Updated: 2026-09-04

P3-A 所需 16/16 matched contexts 已补齐。新增数据仅为 Phase-A1a 同一 1000 张 CIFAR-10
clean source 在 H0/H9/L0/L9 × 四个 round-40 checkpoint 上的 pre-softmax logits 和
probabilities，shape 均为 `4 x 4 x 1000 x 10`。

Clean export 没有训练、backward、corruption generation、PRIME generation 或 checkpoint
修改。关闭 RTX 3050 TF32 后，smoke 与封存 V100 参考的最大概率误差为 `2.98e-7`；正式四个
可交叉验证 context 的最大误差为 `5.36e-7`，argmax agreement 全部为 1.0。

```text
clean output sha256:
4D24CFC1E6A610F721C473CFAED879FAD1751D7151DAC0BF11FE70F7AB0A7A7F
```

Phase-A1a corrupted probabilities、clean outputs、source order、labels、client/arm order和
checkpoint lineage 均匹配，因此原先的 `INSUFFICIENT_EXISTING_ARTIFACTS` 已解除。
