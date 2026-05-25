import pandas as pd

# 50个模块完整数据
modules = [
    # 一、数据导入与预处理模块 (1-8)
    (1, "被动声呐水声目标数据导入", "Data Import", "数据导入", "支持WAV/RAW/Mat/NPY/NC等多种水声数据格式批量导入"),
    (2, "被动声呐水声目标数据标准化处理", "Data Standardization", "标准化", "统一采样率、统一时长、幅度归一化、带通滤波"),
    (3, "评价数据集划分配置", "Dataset Split Config", "数据集划分", "按比例/按类别/按时序/按航次智能划分训练/验证/测试集"),
    (4, "数据标注校验与修正", "Label Validation", "标注校验", "标注格式校验、标注一致性检查、异常标注修正"),
    (5, "数据质量预评估", "Quality Assessment", "质量评估", "信噪比计算、数据完整性检查、可用性评分"),
    (6, "数据增强与扩充", "Data Augmentation", "数据增强", "9种基础增强+高级增强方法配置与应用"),
    (7, "数据版本管理", "Data Version Control", "版本管理", "数据集版本控制、处理链路追溯、变更记录"),
    (8, "多源数据融合", "Multi-source Fusion", "数据融合", "多传感器数据融合、时间对齐、空间配准"),
    
    # 二、混淆矩阵分析模块 (9-15)
    (9, "混淆矩阵计算", "Confusion Matrix", "混淆矩阵", "标准混淆矩阵、归一化混淆矩阵、加权混淆矩阵"),
    (10, "混淆矩阵可视化", "Confusion Matrix Viz", "可视化", "热力图展示、数字标注、颜色映射自定义"),
    (11, "类别级混淆分析", "Class-level Analysis", "类别分析", "每类TP/TN/FP/FN详细统计、混淆模式识别"),
    (12, "混淆矩阵对比", "Matrix Comparison", "对比分析", "多模型/多版本混淆矩阵对比分析"),
    (13, "混淆矩阵导出", "Matrix Export", "数据导出", "导出为CSV/JSON/Excel/图片格式"),
    (14, "混淆矩阵统计检验", "Statistical Test", "统计检验", "McNemar检验、卡方检验、一致性检验"),
    (15, "错误样本定位", "Error Localization", "错误定位", "从混淆矩阵定位具体误分类样本"),
    
    # 三、分类指标归纳统计模块 (16-23)
    (16, "分类指标归纳统计", "Classification Metrics", "基础指标", "准确率、精确率、召回率、F1分数、特异度、马修斯系数"),
    (17, "宏平均与微平均计算", "Macro/Micro Average", "平均指标", "宏平均F1、微平均F1、加权平均F1"),
    (18, "多分类ROC曲线", "ROC Curve", "ROC分析", "One-vs-Rest ROC、One-vs-One ROC、AUC计算"),
    (19, "PR曲线计算", "PR Curve", "PR分析", "精确率-召回率曲线、平均精度(AP)计算"),
    (20, "阈值优化分析", "Threshold Optimization", "阈值优化", "F1最优阈值、Youden指数、平衡点分析"),
    (21, "置信度分析", "Confidence Analysis", "置信度", "置信度分布、置信度校准、可靠性曲线"),
    (22, "分类报告生成", "Classification Report", "报告生成", "每类详细指标报告、汇总统计表"),
    (23, "多标签分类指标", "Multi-label Metrics", "多标签指标", "海明损失、子集准确率、精确匹配率、带权F1"),
    
    # 四、模型稳定性指标集计算模块 (24-30)
    (24, "模型稳定性指标集计算", "Stability Metrics", "稳定性", "交叉验证标准差、置信区间、方差分析"),
    (25, "鲁棒性指标计算", "Robustness Metrics", "鲁棒性", "噪声鲁棒性、干扰鲁棒性、环境变化鲁棒性"),
    (26, "泛化能力评估", "Generalization", "泛化性", "训练/验证差异、过拟合检测、泛化误差估计"),
    (27, "一致性评估", "Consistency", "一致性", "多次运行一致性、随机种子敏感性分析"),
    (28, "敏感性分析", "Sensitivity", "敏感性", "超参数敏感性、输入扰动敏感性"),
    (29, "训练动态稳定性", "Training Stability", "训练稳定性", "损失曲线平滑度、收敛稳定性分析"),
    (30, "设备兼容性测试", "Compatibility", "兼容性", "CPU/GPU/不同硬件平台一致性验证"),
    
    # 五、定量评价指标存储与管理模块 (31-37)
    (31, "定量评价指标存储与管理", "Metric Storage", "存储管理", "指标数据库存储、版本管理、历史记录追溯"),
    (32, "指标版本对比", "Metric Version Compare", "版本对比", "多版本指标对比、趋势分析、性能退化检测"),
    (33, "指标导出与分享", "Metric Export", "导出分享", "导出CSV/JSON/Excel、报告生成、共享链接"),
    (34, "自定义指标配置", "Custom Metrics", "自定义", "用户自定义指标公式、指标组合计算"),
    (35, "指标阈值告警", "Threshold Alert", "告警", "性能阈值设置、自动告警通知"),
    (36, "指标统计分析", "Statistical Analysis", "统计分析", "均值/中位数/标准差、分布直方图、箱线图"),
    (37, "指标生命周期管理", "Lifecycle Management", "生命周期", "指标创建、审批、发布、归档"),
    
    # 六、评价结果多维度可视化模块 (38-42)
    (38, "评价结果多维度可视化", "Multi-dimension Viz", "多维度可视化", "雷达图、平行坐标图、热力图、散点图矩阵"),
    (39, "交互式仪表板", "Interactive Dashboard", "仪表板", "动态筛选、钻取分析、自定义视图"),
    (40, "时空分布可视化", "Spatio-temporal Viz", "时空可视化", "地理位置标注、时间序列变化、航迹叠加"),
    (41, "性能退化趋势图", "Performance Trend", "趋势分析", "版本演进趋势、时间衰减分析"),
    (42, "对比可视化", "Comparison Visualization", "对比可视化", "箱线图对比、柱状图对比、折线图对比"),
    
    # 七、多模型性能对比分析模块 (43-46)
    (43, "多模型性能对比分析", "Model Comparison", "模型对比", "多模型并列对比、排行榜生成、统计显著性检验"),
    (44, "模型排序与筛选", "Model Ranking", "排序筛选", "按指标排序、条件筛选、最优模型推荐"),
    (45, "模型融合效果分析", "Ensemble Analysis", "融合分析", "融合前后对比、融合权重优化、提升幅度计算"),
    (46, "模型差异分析", "Model Difference", "差异分析", "预测差异样本识别、决策边界对比"),
    
    # 八、日志输出与模版管理模块 (47-50)
    (47, "定量分析评价结果分析日志输出", "Log Output", "日志输出", "结构化日志输出、分析过程记录、异常日志"),
    (48, "定量分析评价模版管理", "Template Management", "模版管理", "评价模版创建、存储、加载、共享"),
    (49, "自动化报告生成", "Auto Report", "报告生成", "PDF/HTML/Markdown报告、自定义报告模版"),
    (50, "评价工作流管理", "Workflow Management", "工作流", "评价流程编排、定时评价、批量评价"),
]

# 创建DataFrame
df = pd.DataFrame(modules, columns=["模块编号", "模块名称", "英文名称", "子分类", "功能描述"])

# 添加分类列
categories = ["数据导入与预处理"] * 8 + ["混淆矩阵分析"] * 7 + ["分类指标归纳统计"] * 8 + \
             ["模型稳定性指标"] * 7 + ["指标存储与管理"] * 7 + ["多维度可视化"] * 5 + \
             ["多模型对比分析"] * 4 + ["日志与模版管理"] * 4
df["分类"] = categories

# 保存为Excel
output_file = "passive_sonar_quantitative_evaluation_system_50modules.xlsx"
df.to_excel(output_file, index=False, sheet_name="50模块清单")

print(f"✅ Excel文件已生成: {output_file}")
print(f"📊 共 {len(df)} 个模块，分 {df['分类'].nunique()} 个类别")
print("\n📈 分类统计:")
for cat, count in df['分类'].value_counts().items():
    bar = "█" * (count // 2)
    print(f"  • {cat}: {count}个模块 {bar}")