using CSV
using DataFrames
using Clustering
using Distances
using Plots
using StatsBase
using Statistics
using StatsPlots
using MultivariateStats
using GLM
using LinearAlgebra

# 设置文件路径
input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data3.csv"))
output_file_path = abspath(joinpath(@__DIR__, "../output/"))
default(fontfamily="LXGWWenKai-Medium")

df = CSV.read(input_file_path, DataFrame)
df.is_z_abnormal = abs.(df."Y染色体的Z值") .>= 3

punish_weight = 10000
# 多元化：增加年龄、身高、体重作为聚类变量
k_data = Matrix(select(df, "孕妇BMI", "年龄", "身高", "体重"))
data_matrix = Matrix(select(df, "孕妇BMI", "年龄", "身高", "体重", :is_z_abnormal))
# 聚类数据标准化，现在处理4个连续变量
scaler = fit(ZScoreTransform, data_matrix[:, 1:4], dims=1)
data_matrix[:, 1:4] = StatsBase.transform(scaler, data_matrix[:, 1:4])
data_matrix[:, 5] = data_matrix[:, 5] .* punish_weight

# --- 检测误差因子处理与PCA降维 ---
detection_error_vars = ["原始读段数", "唯一比对的读段数", "被过滤掉读段数的比例", "重复读段的比例", "在参考基因组上比对的比例"]
error_data = Matrix(select(df, detection_error_vars))
error_scaler = fit(ZScoreTransform, error_data, dims=1)
error_data_scaled = StatsBase.transform(error_scaler, error_data)
M = fit(PCA, error_data_scaled'; maxoutdim=2)
error_pcs = predict(M, error_data_scaled')'
df[!, :pc1] = error_pcs[:, 1]
df[!, :pc2] = error_pcs[:, 2]
# --- PCA部分结束 ---

# --- 新增函数：多变量局部加权回归预测 ---
function loess_predict(x_train, y_train, x_pred)
    # x_train: 训练数据的自变量矩阵
    # y_train: 训练数据的因变量向量
    # x_pred: 预测点的自变量矩阵

    n_pred = size(x_pred, 1)
    y_pred = zeros(n_pred)

    # 定义高斯核函数作为权重函数
    function gaussian_kernel(dist, bandwidth)
        return exp.(-0.5 * (dist / bandwidth) .^ 2)
    end

    # 带宽参数，可根据数据调整
    bandwidth = 2.0

    # 遍历每个预测点
    for i in 1:n_pred
        # 计算预测点与所有训练点的欧式距离
        dists = [norm(x_pred[i, :] .- x_train[j, :]) for j in 1:size(x_train, 1)]

        # 计算权重向量
        weights = Diagonal(gaussian_kernel(dists, bandwidth))

        # 准备加权最小二乘法数据
        X = hcat(ones(size(x_train, 1)), x_train)
        W = weights
        y = y_train

        # 使用正规方程解加权最小二乘 (X'WX)^-1 * X'Wy
        # 确保矩阵可逆，否则跳过
        try
            beta = (X' * W * X) \ (X' * W * y)

            # 预测值 = beta[0] + beta[1]*x1 + ...
            y_pred[i] = [1; x_pred[i, :]]' * beta
        catch e
            println("加权最小二乘求解失败，跳过预测点 $i. 错误: $e")
            y_pred[i] = NaN
        end
    end
    return y_pred
end
# --- 新增函数结束 ---

# DBSCAN聚类
n = size(k_data, 1)
distances = pairwise(Euclidean(), k_data', dims=2)
k_distance_plots = []
all_dbscan_intervals = DataFrame()
all_optimal_gest_weeks = DataFrame(cluster=Int[], min_pts=Int[], epsilon=Float64[], optimal_gest_week=Union{Missing,Float64}[])
num_dims = size(k_data, 2)
all_regression_curves = DataFrame() # 新增：用于收集所有回归数据的总表

for min_pts in (num_dims+1):(num_dims+8)
    # 计算k-距离... (此部分代码未变)
    k_distances = zeros(n)
    for i in 1:n
        sorted_distances = sort(distances[:, i])
        k_distances[i] = sorted_distances[min_pts+1]
    end
    sorted_k_distances = sort(k_distances, rev=true)
    len = length(sorted_k_distances)
    second_diff = zeros(len - 2)
    for i in 2:len-1
        second_diff[i-1] = sorted_k_distances[i-1] - 2 * sorted_k_distances[i] + sorted_k_distances[i+1]
    end
    elbow_idx = argmax(abs.(second_diff)) + 1
    epsilon = sorted_k_distances[elbow_idx]

    p_k_distance = plot(sorted_k_distances, title="k-距离图 (k = $min_pts)", xlabel="数据点索引", ylabel="距离", legend=false, size=(400, 300))
    hline!([epsilon], linestyle=:dash, color=:red, label="选择的ε = $(round(epsilon, digits=2))")
    push!(k_distance_plots, p_k_distance)

    dbscan_result = dbscan(data_matrix', epsilon; min_neighbors=min_pts)
    cluster_labels = assignments(dbscan_result)
    temp_df = copy(df)
    temp_df[!, :cluster] = cluster_labels
    temp_df[!, :is_qualified] = temp_df."Y染色体浓度" .>= 0.04

    cluster_dfs = groupby(temp_df, :cluster)
    best_times = DataFrame(cluster=Int[], optimal_gest_week=Union{Missing, Float64}[])

    for grp in cluster_dfs
        if nrow(grp) < 10
            push!(best_times, (first(grp.cluster), missing))
            continue
        end

        x_train = Matrix(select(grp, "检测孕周", "孕妇BMI", "年龄", "身高", "体重", :pc1))
        y_train = grp."Y染色体浓度"

        gest_week_range = collect(10.0:0.1:25.0)
        x_pred = [gest_week_range fill(mean(grp."孕妇BMI"), length(gest_week_range)) fill(mean(grp."年龄"), length(gest_week_range)) fill(mean(grp."身高"), length(gest_week_range)) fill(mean(grp."体重"), length(gest_week_range)) fill(mean(grp.pc1), length(gest_week_range))]
        y_pred = loess_predict(x_train, y_train, x_pred)

        best_week_idx = findfirst(y -> y >= 0.04, y_pred)
        if best_week_idx !== nothing
            best_week = gest_week_range[best_week_idx]
            push!(best_times, (first(grp.cluster), best_week))
        else
            push!(best_times, (first(grp.cluster), missing))
        end

        max_conc_idx = argmax(y_pred)
        optimal_gest_week = gest_week_range[max_conc_idx]
        push!(all_optimal_gest_weeks, (first(grp.cluster), min_pts, epsilon, optimal_gest_week))

        regression_df = DataFrame(
            "孕周" => gest_week_range,
            "预测Y染色体浓度" => y_pred,
            "平均BMI" => fill(mean(grp."孕妇BMI"), length(gest_week_range)),
            "平均年龄" => fill(mean(grp."年龄"), length(gest_week_range)),
            "平均身高" => fill(mean(grp."身高"), length(gest_week_range)),
            "平均体重" => fill(mean(grp."体重"), length(gest_week_range)),
            "平均PC1" => fill(mean(grp.pc1), length(gest_week_range)),
            "cluster" => fill(first(grp.cluster), length(gest_week_range)),
            "min_pts" => fill(min_pts, length(gest_week_range)),
            "epsilon" => fill(epsilon, length(gest_week_range))
        )
        global all_regression_curves = vcat(all_regression_curves, regression_df)
    end

    bmi_intervals = combine(groupby(temp_df, :cluster),
        "孕妇BMI" => minimum => :min_bmi,
        "孕妇BMI" => maximum => :max_bmi,
        "孕妇BMI" => mean => :mean_bmi,
        nrow => :count,
        :is_qualified => mean => :qualification_rate
    )
    final_intervals = leftjoin(bmi_intervals, best_times, on=:cluster)
    final_intervals.k_value = fill(min_pts, nrow(final_intervals))
    final_intervals.epsilon = fill(epsilon, nrow(final_intervals))

    global all_dbscan_intervals = vcat(all_dbscan_intervals, final_intervals)
end

k_distance_com_plot = plot(k_distance_plots..., layout=(2, 4), size=(1600, 800))

# 画图打表
savefig(k_distance_com_plot, joinpath(output_file_path, "k_distance_comparison3.png"))
CSV.write(joinpath(output_file_path, "bmi_intervals3.csv"), all_dbscan_intervals)
CSV.write(joinpath(output_file_path, "regression_curves3.csv"), all_regression_curves)
CSV.write(joinpath(output_file_path, "optimal_gest_weeks3.csv"), all_optimal_gest_weeks)