dyGF 全量实验协议审计与离线分析

范围
本模块只读取协议、已有预测、逐样本指标和测量记录，不执行模型训练、
推理、选模或参数更新。支持原始 3269/340/60 划分的协议核对；不会用
归档包中的 380/18 子集替代历史全量数据，也不会把开发集当独立测试集。

已经实现
1. 全量划分 ID、共同训练祖先、阶段预算、逐轮日志及已有结果的审计。
2. 四格交互效应、条件效应、平均主效应与成组配对 bootstrap。
3. 已有外部基线指标的统一导入；不同方法种子数不一致时按复合物配对。
4. 从已保存主链导出距离；接触阈值、连续距离误差及逐层接触得失分析。
5. 长度分组、逐组删除敏感性、已导出的重复推理漂移和成本记录汇总。

缺失实验的现状（2026-10 追加，详见包根 REPORT_missing_experiments_v1.md）
训练与模型改动已在 code/ 下补齐并通过 CPU 冒烟，但完整预算的重训在本机
不可执行（无可用 GPU；单种子训练成本见 results/geometry_and_cost_v1/）：
  code/interface_ablation_refiner.py        保持容量的输入屏蔽与静态/动态邻域
  code/train/train_matched_ablation_v3.py   双去除（no_pair_no_binding）等 8 个 matched 变体
  code/eval/test_matched_ablations_v3.py    已训臂的统一锁定打分（四格需齐备）
  code/eval/evaluate_external_ablations_v1.py  外部队列上的推理期屏蔽/静态邻域评估
  code/scripts/run_missing_ablations_v1.sh  GPU 训练/打分/交互分析运行手册
已执行：
  * 已有 2x2 的协议审计：9/12 格存在且一致，base 三格缺失；交互效应记为
    not_estimable（results/trained_arm_marginals_v1/partial_factorial_2x2.json）。
  * 外部 20 例上的推理期屏蔽与静态邻域评估（3 种子）、逐层接触得失、
    阈值敏感性、几何复算、CPU 重复漂移与成本汇总。
  * 外部基线 ours/RAAD/MEAN 同口径打分与成组配对比较
    （results/external_baselines_v2/）；DiffAb 见该目录说明。
仍缺：显存峰值从未记录，报告为 null；CUDA 非确定性未测量，CPU 重复漂移
为 0 只约束本机。几何质量沿用手臂自带的 cn_mae/planarity/clash 指标，
本模块仍未新增全原子几何评估器。

依赖与测试
Python 3.9+、NumPy。仅 export-distances 需要 PyTorch 来读取可信本地数据。
pytest 仅用于合成数据测试。以下命令从归档包根目录执行：

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=code python -m pytest code/tests/test_research_audit.py -q -p no:cacheprovider
  PYTHONPATH=code python -m research_audit --help

以下示例的 /tmp 路径是待提供输入或新输出，不是已生成的实验结果。
所有输出拒绝覆盖已有文件。无网络访问，无模型执行。

全量历史协议
  PYTHONPATH=code python -m research_audit audit-history \
    --workspace /media/xd/code1/dyGF \
    --output /tmp/dygf_protocol_audit.json

加 --verify-artifacts 可核对现有检查点、冻结生成器缓存与代码的文件哈希，
不会反序列化检查点。默认检查三个种子 2026/2027/2028，可用 --seeds 指定。
报告完整列出缺失配置，不会把只去除序列输入的 no_sequence 当成双去除。

审计使用历史约定目录：
  full:         outputs/full_pipeline_seed{seed}_v1
  only_pair:    outputs/matched_ablation_v2_no_binding_seed{seed}
  only_binding: outputs/matched_ablation_v2_no_pair_seed{seed}
  base:         outputs/matched_ablation_v2_no_pair_no_binding_seed{seed}

base 是所期待的目录名，不代表存在该训练实现或结果。不同命名的未来
实验可用下面的显式指标清单导入，不应为了通过审计改写历史 plan。
full/only_binding 继承 binding 阶段，only_pair/base 从 pre-binding 祖先
出发，需对照完整阶段历史。协议字符串与文件哈希不能证明运行时保留权重、
优化器状态相等，因此报告 protocol_verified 始终为 false。
完整性、已有记录一致性和实际运行等价性是不同结论。

指标导入
输入清单 JSON 的相对路径以清单所在目录为基准；每个输入必须给出 SHA256：
  {
    "groups": {"group_1": ["sample_1"], "group_2": ["sample_2"]},
    "metrics": ["aligned_rmsd", "contact_f1", "cn_mae"],
    "runs": [
      {"arm": "full", "seed": 2026,
       "metrics_path": "full_seed2026.json", "sha256": "实际文件SHA256"}
    ]
  }
每个指标文件格式为 {"per_complex": [{"sample_id": ..., 指标名: 数值, ...}]}。
每个运行必须完整覆盖 groups 中的全部样本，不能仅保留成功预测。

  PYTHONPATH=code python -m research_audit collect \
    --input /tmp/score_manifest.json --output /tmp/paired_scores.csv

CSV 必填列：sample_id,group,seed,arm，以及指定指标。length 为可选列。
group 必须来自已有依赖关系定义，不能从实验结果重新聚类。

2×2 交互分析
  PYTHONPATH=code python -m research_audit factorial \
    --input /tmp/paired_scores.csv --output /tmp/interaction.json \
    --metric aligned_rmsd:min --metric interface_fape:min --metric contact_f1:max

四格 arm 必须恰为 base、only_pair、only_binding、full。每个样本、每个
种子都须有四格；缺格、重复、非有限值、组归属不一致均报错。
误差指标乘 -1，统一使正值表示收益。报告
  I = direction * (full - only_pair - only_binding + base)
并附两种条件效应、两项平均主效应与 full-base 对照。

主区间：先在每个样本内平均成对种子的差，再按整个依赖组重采样，保留
组内全部样本，最终仍为样本等权估计。另报组等权均值。三个种子不会
扩充成三倍独立样本。交叉种子/组 bootstrap 仅作少种子敏感性分析。
区间未作多重比较校正，不能自动解释为确认性显著性或显著协同。
只有一个依赖组时，采样区间为 null。交互效应依赖所选指标尺度。
可附 --protocol-audit；不完整审计会报错，完整审计也不会自动认证运行等价。

外部基线
  PYTHONPATH=code python -m research_audit baselines \
    --input /tmp/external_scores.csv --output /tmp/external_mean_comparison.json \
    --reference ours --comparator mean \
    --metric aligned_rmsd:min --metric contact_f1:max

允许 ours 三种子而 MEAN/DiffAb 单种子。每个方法内部先按样本平均种子，
再按同一批复合物和关联组配对，不强行配对不同方法的随机种子。
区间以已有检查点为条件，不为单种子方法虚构训练方差。
greedy AAR、first_sample_aar、oracle_aar 必须作为不同指标保留。

接触与逐层诊断
export-distances 的输入清单：
  {
    "ids": ["sample_1", "sample_2"],
    "targets_sha256": "可信本地评估数据.pt的SHA256",
    "runs": [
      {"arm": "ours", "seed": 2026,
       "predictions_path": "generated_seed2026.json", "sha256": "实际文件SHA256",
       "stage_index": 0, "stage_name": "final"}
    ]
  }
已有预测采用归档格式 [{"sample_id": ..., "backbone": L×4×3 数组, ...}]。
目标 .pt 采用 [(sample_id, x, y), ...]，只读取评估坐标及掩码。
torch.load 会反序列化本地数据，因此只传入可信归档文件。

  PYTHONPATH=code python -m research_audit export-distances \
    --input /tmp/prediction_manifest.json --targets data/external_test/inputs.pt \
    --output /tmp/saved_distances.json
  PYTHONPATH=code python -m research_audit contacts \
    --input /tmp/saved_distances.json --output /tmp/contact_diagnostics.json \
    --thresholds 6 7 8 9 10

直接输入 contacts 的格式为数组，每行包含 arm, seed, sample_id,
stage_index, reference_distances, predicted_distances；可给布尔 pair_mask。
距离为同一 Cα 对应关系下的二维 Å 数组。接触采用 <= 阈值。输出 TP/FP/FN、
F1、precision/recall、连续距离 MAE/P95，以及 native<=12 Å 子集的距离 MAE。
双方接触均为空时沿用历史 F1=0 约定，并显式标记 both_empty。

若有逐层坐标，同一运行的 stage_index 必须从零连续编号，原生距离和掩码
不得变化；分别输出新增正确、丢失正确、新增错误和移除错误接触。
第一阶段无前驱，变化记为 null。只有逐层 F1 标量的历史 CSV 无法恢复
接触得失集合，必须提供实际保存的逐层坐标/距离，不能从标量反推。
这些结果是事后诊断，不是分支因果消融，也不是结合实验。

长度分组与关联组稳健性
  PYTHONPATH=code python -m research_audit paired \
    --input /tmp/two_arm_scores.csv --output /tmp/length_sensitivity.json \
    --reference full --comparator base --long-min 16 \
    --metric aligned_rmsd:min --metric contact_f1:max --metric cn_mae:min

paired 要求同种子配对；CSV 只放这两个 arm。length 一旦提供，须全样本一致。
报告 all、length<16、length>=16，并逐个删除整个关联组；删除敏感性范围
不是置信区间。16 是可配置的描述性阈值，不是生物学边界或预注册声明。

复算漂移
  PYTHONPATH=code python -m research_audit drift \
    --input /tmp/repeated_metrics.json --output /tmp/numerical_drift.json \
    --metric aligned_rmsd --metric contact_f1

输入为行数组，必填 arm,sample_id,repeat_id,checkpoint_sha256,input_sha256,
inference_seed,device,precision,runtime_id，以及所选指标。
runtime_id 应固定软件版本、执行配置及线程等条件。同样本的这些条件须
完全一致；随机采样种子或检查点不同会拒绝归为数值复算漂移。
每个样本至少两个不同 repeat_id。可附 backbone 或 sequence_logits，
报告最大成对坐标/数值差和 RMSE。该命令只分析保存的复算结果。

成本汇总
  PYTHONPATH=code python -m research_audit costs \
    --input /tmp/cost_measurements.csv --output /tmp/cost_report.json

CSV 必填 run,seed,device,precision,timing_scope；可选 parameter_count,
trainable_parameter_count,training_seconds,inference_seconds,peak_memory_bytes,
n_inference_examples。缺失测量保持 null，不能用检查点大小替代显存峰值。
历史 epoch.seconds 包含训练、验证和保存耗时，不能称为纯前向时间。

归档完整性
这些新增文件已加入包级 MANIFEST.json。原始训练/评估结果与代码保持不变。
新增报告建议写到包外 /tmp；若存回归档，应重新登记清单，不能宣称旧清单
已经覆盖新输出。

本次验证：code/tests 全量 51 项测试通过（含 19 项 research_audit 合成测试与
新增的 interface_ablation_refiner 测试）；原目录只读协议审计确认 3269/340/60
划分，已有三组元数据一致，三个种子的 base 结果缺失。包级校验没有未登记文件
或哈希不符，但原清单中的 7 个 provenance 文件在当前包中缺失（本次追加的
清单重建保留了这些条目，未以重建掩盖该缺口）；因此
不能声称整个历史归档已通过完整性验证。新增代码没有删除这些文件。
