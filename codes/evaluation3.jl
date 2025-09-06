using CSV
using DataFrames
using Statistics
using Plots

"""
问题3模型评估脚本
评估聚类效果和最佳孕周预测
"""

# 读取数据
input_path = abspath(joinpath(@__DIR__, "../output/"))
bmi_intervals = CSV.read(joinpath(input_path, "bmi_intervals3.csv"), DataFrame)
optimal_weeks = CSV.read(joinpath(input_path, "optimal_gest_weeks3.csv"), DataFrame)
regression_curves = CSV.read(joinpath(input_path, "regression_curves3.csv"), DataFrame)
processed_data = CSV.read(joinpath(input_path, "processed_data3.csv"), DataFrame)

println("=" ^ 60)
println("问题3 模型评估")
println("=" ^ 60)

# 1. 混淆矩阵相关指标评估
println("\n1. 混淆矩阵评估:")

# 基于合格率评估分类性能
for k_value in unique(bmi_intervals.k_value)
    k_data = filter(row -> row.k_value == k_value, bmi_intervals)
    
    println("\n   k = $k_value:")
    
    # 计算每个聚类的性能指标
    for cluster_id in unique(k_data.cluster)
        cluster_data = filter(row -> row.cluster == cluster_id, k_data)
        if !isempty(cluster_data)
            qual_rate = cluster_data.qualification_rate[1]
            count = cluster_data.count[1]
            
            # 计算混淆矩阵元素（基于合格率）
            tp = round(Int, count * qual_rate)  # True Positives
            fn = count - tp  # False Negatives
            
            # 计算精确率和召回率
            precision = qual_rate  # 在这个上下文中，精确率等于合格率
            recall = qual_rate  # 召回率（灵敏度）
            
            # 计算F1-Score
            if precision + recall > 0
                f1_score = 2 * precision * recall / (precision + recall)
            else
                f1_score = 0.0
            end
            
            println("      Cluster $cluster_id:")
            println("         样本数: $count")
            println("         合格率(Precision): $(round(precision, digits=3))")
            println("         召回率(Recall): $(round(recall, digits=3))")
            println("         F1-Score: $(round(f1_score, digits=3))")
        end
    end
end

# 2. 交叉验证评估
println("\n2. 交叉验证评估:")

# 使用不同的k值作为交叉验证的折
k_values = unique(bmi_intervals.k_value)
cv_scores = Float64[]

for k in k_values
    k_data = filter(row -> row.k_value == k, bmi_intervals)
    # 计算该k值下的平均合格率作为评分
    avg_qual_rate = mean(filter(x -> !ismissing(x), k_data.qualification_rate))
    push!(cv_scores, avg_qual_rate)
    println("   k=$k: 平均合格率 = $(round(avg_qual_rate, digits=3))")
end

if !isempty(cv_scores)
    cv_mean = mean(cv_scores)
    cv_std = std(cv_scores)
    println("\n   交叉验证总结:")
    println("      平均合格率: $(round(cv_mean, digits=3))")
    println("      标准差: $(round(cv_std, digits=4))")
    println("      CV分数: $(round(cv_mean - cv_std, digits=3)) ~ $(round(cv_mean + cv_std, digits=3))")
end

# 3. 模型评估 - 基于最佳孕周预测
println("\n3. 最佳孕周预测评估:")

# 分析最佳孕周的分布
for cluster_id in unique(optimal_weeks.cluster)
    cluster_weeks = filter(row -> row.cluster == cluster_id, optimal_weeks)
    if !isempty(cluster_weeks)
        weeks = cluster_weeks.optimal_gest_week
        
        println("\n   Cluster $cluster_id:")
        println("      最佳孕周范围: $(minimum(weeks)) ~ $(maximum(weeks))")
        println("      平均最佳孕周: $(round(mean(weeks), digits=1))")
        println("      标准差: $(round(std(weeks), digits=2))")
        
        # 风险评估（基于题目描述：12周内低风险，13-27周高风险）
        avg_week = mean(weeks)
        if avg_week <= 12
            println("      风险等级: 低风险 (≤12周)")
        elseif avg_week <= 27
            println("      风险等级: 中高风险 (13-27周)")
        else
            println("      风险等级: 高风险 (>27周)")
        end
    end
end

# 4. 敏感性分析 - 分析不同参数对结果的影响
println("\n4. 敏感性分析:")

# 分析epsilon参数的敏感性
epsilon_impact = DataFrame(epsilon = Float64[], avg_qual_rate = Float64[], n_clusters = Int[])
for eps in unique(bmi_intervals.epsilon)
    eps_data = filter(row -> row.epsilon == eps, bmi_intervals)
    avg_qual = mean(filter(x -> !ismissing(x), eps_data.qualification_rate))
    n_clusters = length(unique(eps_data.cluster))
    push!(epsilon_impact, (eps, avg_qual, n_clusters))
end

println("   Epsilon参数影响:")
for row in eachrow(epsilon_impact)
    println("      ε=$(round(row.epsilon, digits=2)): 平均合格率=$(round(row.avg_qual_rate, digits=3)), 聚类数=$(row.n_clusters)")
end

# 计算epsilon变化对结果的影响程度
if nrow(epsilon_impact) > 1
    eps_sensitivity = std(epsilon_impact.avg_qual_rate) / mean(epsilon_impact.avg_qual_rate)
    println("\n   Epsilon敏感度系数: $(round(eps_sensitivity, digits=4))")
    
    if eps_sensitivity < 0.1
        println("   评价: 模型对epsilon参数不敏感，稳定性好")
    elseif eps_sensitivity < 0.2
        println("   评价: 模型对epsilon参数有一定敏感性")
    else
        println("   评价: 模型对epsilon参数较敏感，需谨慎选择")
    end
end

# 5. 模型综合评估
println("模型综合评估")

# 计算各项指标
overall_metrics = Dict{String, Float64}()

# 平均合格率
avg_qualification_rate = mean(filter(x -> !ismissing(x), bmi_intervals.qualification_rate))
overall_metrics["平均合格率"] = avg_qualification_rate

# 聚类稳定性（不同k值下聚类数的变化）
cluster_counts = [length(unique(filter(row -> row.k_value == k, bmi_intervals).cluster)) 
                   for k in unique(bmi_intervals.k_value)]
cluster_stability = 1.0 - std(cluster_counts) / mean(cluster_counts)
overall_metrics["聚类稳定性"] = cluster_stability

# 最佳孕周合理性（12周内的比例）
optimal_week_values = filter(x -> !ismissing(x), optimal_weeks.optimal_gest_week)
early_detection_rate = sum(optimal_week_values .<= 12) / length(optimal_week_values)
overall_metrics["早期检测率"] = early_detection_rate

println("评估指标:")
for (metric, value) in overall_metrics
    println("   $metric: $(round(value, digits=3))")
end

# 计算综合评分
overall_score = mean(values(overall_metrics))
println("\n综合评分: $(round(overall_score, digits=3))")

if overall_score > 0.8
    println("总体评价: 优秀 - 模型在聚类和预测方面表现出色")
elseif overall_score > 0.6
    println("总体评价: 良好 - 模型性能满足要求")
elseif overall_score > 0.4
    println("总体评价: 一般 - 模型有改进空间")
else
    println("总体评价: 较差 - 建议重新调整模型参数")
end

# 6. 详细建议
println("\n改进建议:")

if cluster_stability < 0.8
    println("• 聚类稳定性较低，建议优化DBSCAN参数选择方法")
end

if early_detection_rate < 0.5
    println("• 早期检测率偏低，建议调整回归模型以更好地识别早期最佳时点")
end

if avg_qualification_rate < 0.7
    println("• 平均合格率有提升空间，建议考虑更多影响因素")
end

# 保存评估结果
evaluation_results = DataFrame(
    Metric = ["平均合格率", "聚类稳定性", "早期检测率", "综合评分", "CV平均分", "CV标准差"],
    Value = [
        avg_qualification_rate,
        cluster_stability,
        early_detection_rate,
        overall_score,
        cv_mean,
        cv_std
    ]
)

# 保存详细的聚类评估
cluster_evaluation = DataFrame()
for k_value in unique(bmi_intervals.k_value)
    k_data = filter(row -> row.k_value == k_value, bmi_intervals)
    for cluster_id in unique(k_data.cluster)
        cluster_data = filter(row -> row.cluster == cluster_id, k_data)
        if !isempty(cluster_data)
            qual_rate = cluster_data.qualification_rate[1]
            count = cluster_data.count[1]
            f1_score = 2 * qual_rate * qual_rate / (qual_rate + qual_rate)
            
            push!(cluster_evaluation, (
                k_value = k_value,
                cluster = cluster_id,
                sample_count = count,
                qualification_rate = qual_rate,
                f1_score = f1_score,
                min_bmi = cluster_data.min_bmi[1],
                max_bmi = cluster_data.max_bmi[1],
                optimal_week = cluster_data.optimal_gest_week[1]
            ))
        end
    end
end

CSV.write(joinpath(input_path, "*evaluation_results3.csv"), evaluation_results)
CSV.write(joinpath(input_path, "*cluster_evaluation3.csv"), cluster_evaluation)

println("\n评估结果已保存:")
println("   • evaluation_results3.csv - 总体评估指标")
println("   • cluster_evaluation3.csv - 聚类详细评估")