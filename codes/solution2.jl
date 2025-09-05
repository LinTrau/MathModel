using CSV
using DataFrames
using Clustering
using Loess
using GLM
using Distances
using Plots
using StatsBase
using Statistics
using StatsPlots

input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data2.csv"))
output_file_path = abspath(joinpath(@__DIR__, "../output/"))
default(fontfamily="LXGWWenKai-Medium")
df = CSV.read(input_file_path, DataFrame)
df.is_z_abnormal = abs.(df."Y染色体的Z值") .>= 3
selected_indicators = select(df, "唯一比对的读段数", "被过滤掉读段数的比例", "重复读段的比例", "在参考基因组上比对的比例", "Y染色体的Z值")
variables = ["唯一比对的读段数", "被过滤掉读段数的比例", "重复读段的比例", "在参考基因组上比对的比例", "Y染色体的Z值"]
plots_list = []
indicators_matrix = Matrix(selected_indicators)
indicators_names = names(selected_indicators)

epsilon = 3.34  # 聚类半径
min_pts = 3  # 聚类点数
weight = 100  # 惩罚权重
k_data = Matrix(select(df, "孕妇BMI", "检测孕周"))
data_matrix = Matrix(select(df, "孕妇BMI", "检测孕周", :is_z_abnormal))
data_matrix[:, 3] = data_matrix[:, 3] .* weight

function positivize(col::Vector, type::Symbol)
    if type == :benefit
        return col
    elseif type == :cost
        return maximum(col) .- col
    end
end

# 测量误差处理
for j in 1:size(indicators_matrix, 2)
    col_data = indicators_matrix[:, j]
    if occursin("比例", indicators_names[j]) || occursin("重复", indicators_names[j])
        indicators_matrix[:, j] = positivize(col_data, :cost)
    elseif occursin("Z值", indicators_names[j])
        indicators_matrix[:, j] = positivize(abs.(col_data), :cost)
    end
end
for j in 1:size(indicators_matrix, 2)
    if indicators_names[j] in variables[1:3]
        indicators_matrix[:, j] = log1p.(indicators_matrix[:, j])
    end
end
for j in 1:size(indicators_matrix, 2)
    min_val = minimum(indicators_matrix[:, j])
    max_val = maximum(indicators_matrix[:, j])
    if max_val != min_val
        indicators_matrix[:, j] = (indicators_matrix[:, j] .- min_val) ./ (max_val - min_val)
    end
end

# 熵权法
n = size(indicators_matrix, 1)
k = 1.0 / log(n)
entropy = zeros(size(indicators_matrix, 2))
for j in 1:size(indicators_matrix, 2)
    p_ij = indicators_matrix[:, j] ./ sum(indicators_matrix[:, j])
    entropy[j] = -k * sum(p_ij .* log.(p_ij .+ 1e-6))
end
weights = (1.0 .- entropy) ./ sum(1.0 .- entropy)

# TOPSIS
ideal_positive = maximum(indicators_matrix, dims=1)
ideal_negative = minimum(indicators_matrix, dims=1)
positive_distance = zeros(n)
negative_distance = zeros(n)
for i in 1:n
    positive_distance[i] = sqrt(sum(weights .* (indicators_matrix[i, :]' .- ideal_positive) .^ 2))
    negative_distance[i] = sqrt(sum(weights .* (indicators_matrix[i, :]' .- ideal_negative) .^ 2))
end
topsis_score = negative_distance ./ (positive_distance .+ negative_distance)
df[!, :topsis_score] = topsis_score
entropy_topsis = DataFrame("孕妇代码" => df."孕妇代码", "唯一比对的读段数权重" => weights[1], "被过滤掉读段数的比例权重" => weights[2], "重复读段的比例权重" => weights[3], "在参考基因组上比对的比例权重" => weights[4], "Y染色体的Z值权重" => weights[5], "TOPSIS总得分" => topsis_score)

# 遍历每个变量并绘制直方图和分布曲线
for var in variables
    p = plot(title="变量分布: $(var)", xlabel=var, ylabel="密度", legend=false, size=(600, 400))
    stephist!(p, df[!, var], normalize=true, label="直方图", fillalpha=0.3, bins=50)
    density!(p, df[!, var], label="分布曲线", linewidth=2)
    push!(plots_list, p)
end
distribution_plot = plot(plots_list..., layout=(2, 3), size=(1200, 800), plot_title="各变量分布总览")

# 聚类
# 计算k-距离
n = size(k_data, 1)
distances = pairwise(Euclidean(), k_data, dims=1)
k_distances = zeros(n)
for i in 1:n
    sorted_distances = sort(distances[:, i])
    k_distances[i] = sorted_distances[min_pts+1]
end
sorted_k_distances = sort(k_distances, rev=true)
p_k_distance = plot(sorted_k_distances, title="k-距离图 (k = $min_pts)", xlabel="数据点索引（按距离排序）", ylabel="到第 k 个邻居的距离", legend=false, size=(800, 600))
result = dbscan(data_matrix', epsilon; min_neighbors=min_pts)
df[!, :cluster] = assignments(result)
# DBCSAN方法
cluster_summary = combine(groupby(df, :cluster), "孕妇BMI" => mean => :mean_bmi, "检测孕周" => mean => :mean_gestational_week, :is_z_abnormal => sum => :abnormal_count, nrow => :count)
cluster_summary[!, :abnormal_ratio] = cluster_summary.abnormal_count ./ cluster_summary.count
grouped_by_cluster = groupby(df, :cluster)
bmi_intervals = combine(grouped_by_cluster, "孕妇BMI" => minimum => :min_bmi, "孕妇BMI" => maximum => :max_bmi, "孕妇BMI" => mean => :mean_bmi, nrow => :count)
println("各聚类中孕妇BMI的区间统计：", bmi_intervals)

# BMI-孕周的回归
cluster_plot = plot(title="BMI-孕周聚类与局部加权回归", xlabel="孕妇BMI", ylabel="检测孕周", legend=:topright, size=(800, 600))
for (i, grp) in enumerate(groupby(df, :cluster))
    # 筛选掉噪声点
    if grp.cluster[1] == 0
        scatter!(cluster_plot, grp."孕妇BMI", grp."检测孕周", label="噪声点", markershape=:x, markercolor=:black)
        continue
    end
    # 拟合局部加权回归模型
    x_data = grp."孕妇BMI"
    y_data = grp."检测孕周"
    if length(x_data) > 1
        model = loess(x_data, y_data)
        x_pred = collect(range(minimum(x_data), stop=maximum(x_data), length=100))
        y_pred = predict(model, x_pred)
        scatter!(cluster_plot, x_data, y_data, label="聚类 $(grp.cluster[1])", markercolor=i)
        plot!(cluster_plot, x_pred, y_pred, label="回归曲线 $(grp.cluster[1])", linewidth=2, linestyle=:dash, linecolor=i)
        mid_idx = Int(floor(length(x_pred) / 2))
        annotate!(x_pred[mid_idx], y_pred[mid_idx] + 0.5,
            text("outlier_concentration: $(round(cluster_summary[cluster_summary.cluster .== grp.cluster[1], :abnormal_ratio][1] * 100, digits=8))%", 8, :left, :bottom, :black))
    end
end
df_abnormal = filter(:is_z_abnormal => x -> x == true, df)
scatter!(cluster_plot, df_abnormal."孕妇BMI", df_abnormal."检测孕周", markercolor=:cyan, markershape=:rtriangle, label="Z值异常点", markersize=6)

# Y染色体浓度-其他因素的回归
X = hcat(df."检测孕周", df."孕妇BMI", df.topsis_score)
y = df."Y染色体浓度"
regression_df = DataFrame("检测孕周" => df."检测孕周", "孕妇BMI" => df."孕妇BMI", "TOPSIS评分" => df.topsis_score, "Y染色体浓度" => df."Y染色体浓度")
# 多元线性回归
model_lm = lm(@formula(Y染色体浓度 ~ 检测孕周 + 孕妇BMI + TOPSIS评分), regression_df)
y_pred_lm = GLM.predict(model_lm)
residuals_lm = df."Y染色体浓度" .- y_pred_lm
p_residuals_lm = plot(y_pred_lm, residuals_lm, seriestype=:scatter, xlabel="预测值", ylabel="残差", title="多元线性回归：残差图", legend=false, markeralpha=0.5)
hline!(p_residuals_lm, [0], linestyle=:dash, color=:red)
p_predicted_lm = plot(y_pred_lm, df."Y染色体浓度", seriestype=:scatter, xlabel="预测值", ylabel="实际值", title="多元线性回归：预测值-实际值", legend=false, markeralpha=0.5)
plot!(p_predicted_lm, [minimum(y_pred_lm), maximum(y_pred_lm)], [minimum(y_pred_lm), maximum(y_pred_lm)], linestyle=:dash, color=:red)
regression_plot_lm = plot(p_residuals_lm, p_predicted_lm, layout=(1, 2), size=(1000, 500))
# 广义线性模型
regression_df_glm = DataFrame("检测孕周" => df."检测孕周", "孕妇BMI" => df."孕妇BMI", "TOPSIS评分" => df.topsis_score, "Y染色体浓度" => df."Y染色体浓度" .+ 1e-9)
model_glm = glm(@formula(Y染色体浓度 ~ 检测孕周 + 孕妇BMI + TOPSIS评分), regression_df_glm, Gamma(), LogLink())
y_pred_glm = GLM.predict(model_glm)
residuals_glm = df."Y染色体浓度" .- y_pred_glm
p_residuals_glm = plot(y_pred_glm, residuals_glm, seriestype=:scatter, xlabel="预测值", ylabel="残差", title="广义线性模型: 残差图", legend=false, markeralpha=0.5)
hline!(p_residuals_glm, [0], linestyle=:dash, color=:red)
p_predicted_glm = plot(y_pred_glm, df."Y染色体浓度", seriestype=:scatter, xlabel="预测值", ylabel="实际值", title="广义线性模型: 预测值 - 实际值", legend=false, markeralpha=0.5)
plot!(p_predicted_glm, [minimum(y_pred_glm), maximum(y_pred_glm)], [minimum(y_pred_glm), maximum(y_pred_glm)], linestyle=:dash, color=:red)
regression_plot_glm = plot(p_residuals_glm, p_predicted_glm, layout=(1, 2), size=(1000, 500))

# 绘图、打表
savefig(p_k_distance, joinpath(output_file_path, "k_distance_graph.png"))
savefig(cluster_plot, joinpath(output_file_path, "bmi_gestational_week_dbscan_loess.png"))
savefig(distribution_plot, joinpath(output_file_path, "variable_distributions.png"))
savefig(regression_plot_lm, joinpath(output_file_path, "regression_diagnostic_lm.png"))
savefig(regression_plot_glm, joinpath(output_file_path, "regression_diagnostic_glm.png"))
CSV.write(joinpath(output_file_path, "topsis_scores.csv"), entropy_topsis)