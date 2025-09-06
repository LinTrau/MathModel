using CSV
using DataFrames
using Statistics
using GLM
using Plots

"""
问题2模型评估脚本
评估主成分回归模型性能和聚类结果
"""

# 读取数据
input_path = abspath(joinpath(@__DIR__, "../output/"))
analysis_results = CSV.read(joinpath(input_path, "analysis_results2.csv"), DataFrame)
bmi_intervals = CSV.read(joinpath(input_path, "bmi_intervals2.csv"), DataFrame)
processed_data = CSV.read(joinpath(input_path, "processed_data2.csv"), DataFrame)

println("=" ^ 60)
println("问题2 模型评估")
println("=" ^ 60)

# 1. 决定系数 R² 评估
println("\n1. 决定系数(R²)评估:")
r2_row = filter(row -> !ismissing(row.Metric) && occursin("R²", row.Metric), analysis_results)
if !isempty(r2_row)
    r2_value = r2_row.Value[1]
    println("   R² = $(round(r2_value, digits=4))")
    println("   模型解释了 $(round(r2_value * 100, digits=2))% 的Y染色体浓度变异")
    
    # R²评价标准
    if r2_value > 0.7
        println("   评价: 优秀 - 模型拟合效果很好")
    elseif r2_value > 0.5
        println("   评价: 良好 - 模型有一定解释能力")
    elseif r2_value > 0.3
        println("   评价: 一般 - 模型解释能力有限")
    else
        println("   评价: 较差 - 模型需要改进")
    end
end

# 2. t-检验统计量评估
println("\n2. t-检验统计量评估:")
coef_rows = filter(row -> !ismissing(row.Metric) && occursin("主成分回归系数", row.Metric), analysis_results)
if !isempty(coef_rows)
    for row in eachrow(coef_rows)
        if !ismissing(row.Variable) && startswith(row.Variable, "PC")
            t_value = row.TValue
            p_value = row.PValue
            if !ismissing(t_value) && !ismissing(p_value)
                println("   $(row.Variable):")
                println("      t值 = $(round(t_value, digits=4))")
                println("      p值 = $(round(p_value, digits=4))")
                
                # 计算标准误
                if !ismissing(row.Coefficient) && !ismissing(row.StdError)
                    se_beta = row.StdError
                    beta = row.Coefficient
                    println("      β̂ = $(round(beta, digits=4)) ± $(round(se_beta, digits=4))")
                end
                
                # 显著性判断
                if p_value < 0.001
                    println("      显著性: *** (p < 0.001)")
                elseif p_value < 0.01
                    println("      显著性: ** (p < 0.01)")
                elseif p_value < 0.05
                    println("      显著性: * (p < 0.05)")
                else
                    println("      显著性: 不显著")
                end
            end
        end
    end
end

# 3. 轮廓系数(Silhouette Score)评估
println("\n3. 轮廓系数评估:")
# 为每个聚类算法计算轮廓系数
for algorithm in ["DBSCAN", "HDBSCAN"]
    algo_data = filter(row -> row.algorithm == algorithm, bmi_intervals)
    if !isempty(algo_data)
        # 获取有效聚类（排除噪声点）
        valid_clusters = filter(row -> row.cluster > 0, algo_data)
        
        if !isempty(valid_clusters)
            # 计算簇内和簇间距离（简化版本，基于BMI范围）
            n_clusters = length(unique(valid_clusters.cluster))
            
            # 计算每个簇的紧密度（用BMI范围表示）
            intra_distances = Float64[]
            for cluster in unique(valid_clusters.cluster)
                cluster_data = filter(row -> row.cluster == cluster, valid_clusters)
                if !isempty(cluster_data)
                    bmi_range = cluster_data.max_bmi[1] - cluster_data.min_bmi[1]
                    push!(intra_distances, bmi_range)
                end
            end
            
            # 计算簇间分离度（用BMI均值差表示）
            inter_distances = Float64[]
            clusters = unique(valid_clusters.cluster)
            for i in 1:length(clusters)
                for j in (i+1):length(clusters)
                    cluster_i = filter(row -> row.cluster == clusters[i], valid_clusters)
                    cluster_j = filter(row -> row.cluster == clusters[j], valid_clusters)
                    if !isempty(cluster_i) && !isempty(cluster_j)
                        mean_diff = abs(cluster_i.mean_bmi[1] - cluster_j.mean_bmi[1])
                        push!(inter_distances, mean_diff)
                    end
                end
            end
            
            # 计算简化的轮廓系数
            if !isempty(intra_distances) && !isempty(inter_distances)
                avg_intra = mean(intra_distances)
                avg_inter = mean(inter_distances)
                silhouette = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
                
                println("   $algorithm:")
                println("      簇数: $n_clusters")
                println("      平均簇内距离: $(round(avg_intra, digits=3))")
                println("      平均簇间距离: $(round(avg_inter, digits=3))")
                println("      轮廓系数: $(round(silhouette, digits=4))")
                
                # 评价标准
                if silhouette > 0.7
                    println("      评价: 优秀 - 聚类结构清晰")
                elseif silhouette > 0.5
                    println("      评价: 良好 - 聚类合理")
                elseif silhouette > 0.25
                    println("      评价: 一般 - 聚类结构较弱")
                else
                    println("      评价: 较差 - 聚类效果不佳")
                end
            end
        end
    end
end

# 5. 敏感性分析
println("\n5. 敏感性分析:")
# 分析检测误差对结果的影响
orig_coef_rows = filter(row -> !ismissing(row.Metric) && occursin("原始变量回归系数", row.Metric), analysis_results)
if !isempty(orig_coef_rows)
    println("   各变量对Y染色体浓度的影响系数:")
    
    sensitivities = Dict{String, Float64}()
    for row in eachrow(orig_coef_rows)
        if !ismissing(row.Variable) && !ismissing(row.Coefficient)
            sensitivities[row.Variable] = abs(row.Coefficient)
            println("      $(row.Variable): $(round(row.Coefficient, digits=6))")
        end
    end
    
    # 计算相对重要性
    if !isempty(sensitivities)
        max_sensitivity = maximum(values(sensitivities))
        println("\n   相对重要性分析:")
        sorted_vars = sort(collect(sensitivities), by=x->x[2], rev=true)
        for (var, coef) in sorted_vars
            relative_importance = (coef / max_sensitivity) * 100
            println("      $var: $(round(relative_importance, digits=1))%")
        end
    end
end

# 6. 模型评估总结
println("模型评估总结")
# 计算总体评分
scores = Dict{String, Float64}()

# R²贡献
scores["R²"] = r2_value

if !isempty(scores)
    overall_score = mean(values(scores))
    println("综合评分: $(round(overall_score, digits=3))")
    
    if overall_score > 0.8
        println("总体评价: 模型表现优秀")
    elseif overall_score > 0.6
        println("总体评价: 模型表现良好")
    elseif overall_score > 0.4
        println("总体评价: 模型表现一般")
    else
        println("总体评价: 模型需要改进")
    end
end

# 保存评估结果
evaluation_results = DataFrame(
    Metric = ["R²", "聚类数(DBSCAN)", "聚类数(HDBSCAN)"],
    Value = [
        r2_value,
        length(unique(filter(row -> row.algorithm == "DBSCAN", bmi_intervals).cluster)),
        length(unique(filter(row -> row.algorithm == "HDBSCAN", bmi_intervals).cluster))
    ]
)

CSV.write(joinpath(input_path, "*evaluation_results2.csv"), evaluation_results)
println("\n评估结果已保存至: evaluation_results2.csv")