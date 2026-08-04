# RAHFL 源码精读理解记录

更新时间：2026-07-07

本文记录对 `RAHFL-master` 源码的重新精读结论。重点不是复述论文，而是说明源码里真正发生了什么训练和通信。

## 1. 源码入口和整体流程

核心入口：

```text
RAHFL-master/Network/pretrain.py
RAHFL-master/HHF/RAHFL.py
RAHFL-master/loss.py
RAHFL-master/Dataset/dataaug.py
RAHFL-master/Dataset/utils.py
```

原始 RAHFL 的完整流程是：

```text
1. pretrain.py 对每个客户端模型做 40 epoch 本地 CE 预训练。
2. RAHFL.py 加载预训练模型。
3. 每个通信轮先评估所有客户端。
4. 用 public CIFAR-100 做 HFL / AsymHFL public-logit 蒸馏。
5. 再用 private CIFAR-10-C 做 AugMix + JSD + DCL 本地鲁棒训练。
6. 最后一轮再次测试并保存模型。
```

原始超参数中比较关键的是：

```text
TrainBatchSize = 256
Pretrain_Epoch = 40
CommunicationEpoch = 40
AugMix_Module = 'jsd'
CL_Module = 'dcl'
Col_Module = 'asymhfl'
Private_Corrupt_Rate = 1
Test_Corrupt_Rate = 1
Public_Corrupt_Rate = 0
Nets_Name_List = ['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2']
```

我们统一 runner 里的 RAHFL 数字不是论文最强复现，因为默认没有独立 40 epoch 预训练，public batch 数也更小。这一点解释了为什么当前 fair baseline 约 56.41，而论文报告更高。

## 2. RAHFL 的 AugMix 不是单独原版 AugMix

`RAHFL-master/Dataset/augmentations.py` 基本沿用了 AugMix 论文的基础增强算子：

```text
autocontrast, equalize, posterize, rotate, solarize,
shear_x, shear_y, translate_x, translate_y
```

`dataaug.py` 中的 `aug(image, preprocess)` 是标准 AugMix 思路：

```text
mixture_width = 3
mixture_depth = -1  # 每条增强链随机深度 1 到 3
aug_severity = 3
ws ~ Dirichlet([1, 1, 1])
m  ~ Beta(1, 1)
mixed = (1 - m) * preprocess(image) + m * sum_i ws_i * preprocess(aug_chain_i(image))
```

但是 RAHFL 的关键不只是这个 AugMix。它在 `Dataset/utils.py` 里构造了一个四视图样本：

```text
images[0] = preprocess(base crop)
images[1] = AugMix(base crop)
images[2] = AugMix(base crop)
images[3] = preprocess(second transformed crop)
```

其中 `images[3]` 来自 `transform_train_weak`，实际包含：

```text
RandomResizedCrop
ColorJitter
RandomGrayscale
GaussianBlur
RandomHorizontalFlip
```

所以源码里的 RAHFL local base 更准确地说是：

```text
AugMix two strong views
+ JSD prediction consistency
+ an extra transformed view for DCL feature learning
```

这比“直接使用 AugMix 原论文”更强。

## 3. 本地训练损失的真实组成

在 `HHF/RAHFL.py` 的 `update_model_via_private_data()` 中，若 `AugMix_Module == 'jsd'`，一个 batch 的本地损失是：

```text
L_local = CE(clean)
        + 12 * JSD(clean, aug1, aug2)
        + DCL(clean_feature, weak_feature, strong_feature)
```

CE 只作用在 clean/base view：

```python
loss = criterion(logits_clean, labels)
```

JSD 约束 clean、AugMix1、AugMix2 的预测分布一致：

```python
p_mixture = log((p_clean + p_aug1 + p_aug2) / 3)
jsd_loss = (KL(p_mixture, p_clean)
          + KL(p_mixture, p_aug1)
          + KL(p_mixture, p_aug2)) / 3
loss += 12 * jsd_loss
```

DCL 分支单独再前向三份图像到 backbone：

```python
images_cont = cat(images[0], images[1], images[3])
features = normalize(network.module.backbone(images_cont))
fclean1, f1, fclean2 = split(features)

DCLLoss(
    original_feature=fclean1.unsqueeze(1),
    weak_feature=fclean2.unsqueeze(1),
    strong_feature=f1.unsqueeze(1),
    labels=labels,
)
```

注意这里的命名关系：

```text
original_feature = images[0] 的 clean/base 特征
strong_feature   = images[1] 的 AugMix 特征
weak_feature     = images[3] 的额外 transformed crop 特征
```

## 4. DCLLoss 的真实逻辑

`loss.py` 里的 `DCLLoss` 有两部分。

第一部分是 supervised contrastive loss，但只使用：

```text
original_feature + weak_feature
```

即：

```python
features = torch.cat([original_feature, weak_feature], dim=1)
```

它根据 label mask，把同类样本视为 positives，不同类样本视为 negatives。

第二部分是 DDM / relation alignment。它不是直接把 strong view 拉到 clean view，而是让 strong view 模仿 weak view 的关系分布：

```python
wo_features = cat([weak_feature, original_feature])
w_sim = weak_feature   @ wo_features.T
s_sim = strong_feature @ wo_features.T

w_logits = softmax(exp(w_sim) / T)
s_logits = softmax(exp(s_sim) / T)

dmm_loss = KL(log(s_logits), detach(w_logits))
```

因此 DCL 的方向是：

```text
clean + weak 负责形成监督对比空间；
weak view 作为关系教师；
strong AugMix view 作为学生；
strong view 被要求模仿 weak view 对 clean/weak 特征库的相似度分布。
```

这点非常关键。RAHFL 并不是粗暴地把强增强也放进 SupCon 正样本里，而是让强增强学习一个相对温和视图的关系结构。这个设计天然适合 corruption robustness，因为强增强可能已经很难，但它不被直接当成稳定 anchor，而是被关系蒸馏约束。

## 5. AsymHFL 通信逻辑

RAHFL 的通信只依赖 public CIFAR-100 logits，因此天然支持模型异构。不同客户端模型 backbone 维度可以不同，只要分类 logits 都是 CIFAR-10 的 10 维即可。

每轮通信前，源码会先在 private CIFAR-10-C test set 上测试每个客户端：

```python
matrix_update_acc_list.append(accuracy)
```

随后在 public CIFAR-100 batch 上收集每个客户端输出：

```python
p_logits    = softmax(logits).detach()   # teacher target
plog_logits = log_softmax(logits)        # student log prob
```

如果 `Col_Module == 'asymhfl'`，客户端只向“测试准确率不低于自己”的客户端学习：

```python
if matrix_update_acc_list[participant] <= matrix_update_acc_list[i]:
    loss += KL(student_log_prob, teacher_prob_i)
loss /= learn_from_client_num
```

所以 AsymHFL 的本质是：

```text
用 private test accuracy 做客户端级强弱排序；
弱客户端在 public images 上蒸馏强客户端的 softmax 分布；
蒸馏粒度是完整 10 类 softmax；
不区分类别、不区分样本、不区分缺失类。
```

严格实验协议下，这个“用测试集准确率做路由”存在测试信息泄漏风险。我们当前为了公平复现 RAHFL baseline 保留了这个行为，但论文写作时要非常谨慎。

## 6. 数据处理和 Non-IID 划分

`CIFAR_C` 不是读取普通图片文件夹，而是读取 RAHFL 预生成的 `.npy`：

```text
train/random_corrupt_1.npy
train/labels.npy
test/random_corrupt_1.npy
test/labels.npy
```

`make_cifar_c.py` 的逻辑是：

```text
对原始 CIFAR 图像随机选一部分；
对被选中的图像随机选择 corruption 类型；
随机选择 severity；
保存成 random_corrupt_{rate}.npy。
```

Non-IID 使用 `sampling.py` 的 Dirichlet 划分：

```python
proportions = np.random.dirichlet(np.repeat(dirichlet_beta, num_users))
```

alpha / beta 越小，客户端类别越偏；越大越接近 IID。

## 7. 模型异构如何被支持

四个模型都返回：

```python
return linear_output, embedding_output
```

其中：

```text
linear_output   = 10 类 logits，用于 public-logit KD 通信；
embedding_output = projector 输出，但 RAHFL DCL 实际没有用它。
```

RAHFL DCL 分支直接调用：

```python
network.module.backbone(images_cont)
```

这意味着每个客户端的 DCL 在本地自己的特征维度里做，不需要跨模型对齐。跨客户端通信只发生在 logits 空间，因此模型异构被绕开了。

## 8. 我对 RAHFL 强点的重新判断

RAHFL 的强度主要来自三件事：

```text
1. 本地四视图鲁棒训练：
   clean + AugMix1 + AugMix2 + extra transformed crop。

2. JSD 约束预测一致性：
   让模型在强扰动下仍输出稳定分类分布。

3. DCL 的关系蒸馏：
   strong AugMix view 不直接做 SupCon anchor，而是模仿 weak view 的关系分布。
```

AsymHFL 的贡献在我们当前统一 runner 下相对有限，但它确实提供了额外的 public-logit 知识交互。真正不能低估的是本地 AugMix/JSD/DCL 这套组合。

## 9. 可借鉴的创新方向

从源码看，后续如果要围绕 RAHFL 改进，最有价值的切入点不是简单替换 AugMix，也不是盲目改 public logits，而是：

```text
1. 改 DCL 的关系教师：
   原始 DCL 固定 weak -> strong。
   可以考虑在 Non-IID 下做类别均衡、类间关系重标定、tail-aware relation。

2. 改 DCL 的关系库：
   原始关系库只包含当前 batch 的 weak + clean。
   Non-IID mini-batch 下 tail 类正样本不足，关系估计会很偏。

3. 改通信和本地 DCL 的接口：
   不是把 public logits 当万能知识，而是让 public logits 只辅助那些本地 DCL 已经建立稳定关系的类别或样本。

4. 改 AsymHFL 路由：
   原始路由是客户端整体准确率，粒度很粗且依赖 test accuracy。
   可以改为验证集、客户端自估计、类别/样本级可靠性，但必须证明不是只增加复杂度。
```

一句话总结：

```text
RAHFL 的核心不是“用了 AugMix”，而是把 AugMix prediction consistency 和 DCL feature relation consistency 叠在一起。
如果我们的创新不能触碰这个局部鲁棒表征机制，就很难正面大幅超过它。
```
