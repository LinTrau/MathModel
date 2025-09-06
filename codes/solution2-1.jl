using CSV
using PyCall
using DataFrames
using Clustering
using Loess
using Distances
using Plots
using StatsBase
using Statistics
using StatsPlots
@pyimport hdbscan

input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data2.csv"))
output_file_path = abspath(joinpath(@__DIR__, "../output/"))
default(fontfamily="LXGWWenKai-Medium")

df = CSV.read(input_file_path, DataFrame)
df.is_z_abnormal = (abs.(df."Y染色体的Z值") .>= 3) .| (df."GC含量" .< 0.4) .| (df."GC含量" .> 0.6)

punish_weight = 10000
k_data = Matrix(select(df, "孕妇BMI", "检测孕周"))
data_matrix = Matrix(select(df, "孕妇BMI", "检测孕周", :is_z_abnormal))
# 聚类数据标准化
scaler = fit(ZScoreTransform, data_matrix[:, 1:2], dims=1)
data_matrix[:, 1:2] = StatsBase.transform(scaler, data_matrix[:, 1:2])
data_matrix[:, 3] = data_matrix[:, 3] .* punish_weight


# DBSCAN聚类
n = size(k_data, 1)
distances = pairwise(Euclidean(), k_data, dims=1)
k_distance_plots = []
dbscan_plots = []
all_dbscan_intervals = DataFrame()
for min_pts in 3:10
    # 计算k-距离
    k_distances = zeros(n)
    for i in 1:n
        sorted_distances = sort(distances[:, i])
        k_distances[i] = sorted_distances[min_pts+1]
    end
    sorted_k_distances = sort(k_distances, rev=true)
    len = length(sorted_k_distances)
    # 找拐点
    second_diff = zeros(len - 2)
    for i in 2:len-1
        second_diff[i-1] = sorted_k_distances[i-1] - 2 * sorted_k_distances[i] + sorted_k_distances[i+1]
    end
    elbow_idx = argmax(abs.(second_diff)) + 1
    epsilon = sorted_k_distances[elbow_idx]
    p_k_distance = plot(sorted_k_distances, title="k-距离图 (k = $min_pts)", xlabel="数据点索引", ylabel="距离", legend=false, size=(400, 300))
    hline!([epsilon], linestyle=:dash, color=:red, label="选择的ε = $(round(epsilon, digits=2))")
    push!(k_distance_plots, p_k_distance)
    # 打表
    dbscan_result = dbscan(data_matrix', epsilon; min_neighbors=min_pts)
    cluster_labels = assignments(dbscan_result)
    temp_df = copy(df)
    temp_df[!, :cluster] = cluster_labels
    bmi_intervals = combine(groupby(temp_df, :cluster), "孕妇BMI" => minimum => :min_bmi, "孕妇BMI" => maximum => :max_bmi, "孕妇BMI" => mean => :mean_bmi, nrow => :count)
    bmi_intervals.k_value = fill(min_pts, nrow(bmi_intervals))
    bmi_intervals.epsilon = fill(epsilon, nrow(bmi_intervals))
    global all_dbscan_intervals = vcat(all_dbscan_intervals, bmi_intervals)
    # 绘图
    cluster_plot = plot(title="k=$min_pts, ε=$(round(epsilon, digits=2))", xlabel="孕妇BMI", ylabel="检测孕周", legend=false, size=(400, 300))
    for (i, grp) in enumerate(groupby(temp_df, :cluster))
        if grp.cluster[1] == 0
            scatter!(cluster_plot, grp."孕妇BMI", grp."检测孕周", markershape=:x, markercolor=:black, markersize=2)
            continue
        end
        x_data = grp."孕妇BMI"
        y_data = grp."检测孕周"
        model = loess(x_data, y_data)
        x_pred = collect(range(minimum(x_data), stop=maximum(x_data), length=50))
        y_pred = predict(model, x_pred)
        # 寻找回归最小值
        min_idx = argmin(y_pred)
        min_x = x_pred[min_idx]
        min_y = y_pred[min_idx]
        scatter!(cluster_plot, [min_x], [min_y], markercolor=:red, markershape=:star5, markersize=4, markerstrokewidth=1, markerstrokecolor=:black)
        annotate!(min_x, min_y + 0.8, text("Min($(round(min_x, digits=1)), $(round(min_y, digits=1)))", 6, :center, :bottom, :red))
        scatter!(cluster_plot, x_data, y_data, markercolor=i, markersize=2)
        plot!(cluster_plot, x_pred, y_pred, linewidth=1, linestyle=:dash, linecolor=i)

    end
    df_abnormal = filter(:is_z_abnormal => x -> x == true, temp_df)
    scatter!(cluster_plot, df_abnormal."孕妇BMI", df_abnormal."检测孕周", markercolor=:cyan, markershape=:rtriangle, label="Z值异常点", markersize=5)
    push!(dbscan_plots, cluster_plot)
end
k_distance_com_plot = plot(k_distance_plots..., layout=(2, 4), size=(1600, 800))
dbscan_com_plot = plot(dbscan_plots..., layout=(2, 4), size=(1600, 800))
# HDBSCAN聚类
hdbscan_plots = []
all_hdbscan_intervals = DataFrame()
for k in 5:12
    clusterer = hdbscan.HDBSCAN(min_cluster_size=k)
    hdbscan_clusters = clusterer.fit_predict(data_matrix[:, 1:2])
    df.hdbscan_cluster = hdbscan_clusters
    p_hdbscan = plot(title="HDBSCAN 聚类结果 (MinClusterSize=$k)", xlabel="孕妇BMI", ylabel="检测孕周", legend=false, size=(1000, 800))
    df_abnormal = filter(:is_z_abnormal => x -> x == true, df)
    scatter!(p_hdbscan, df_abnormal."孕妇BMI", df_abnormal."检测孕周", markercolor=:cyan, markershape=:rtriangle, label="Z值异常点", markersize=3)
    for (i, grp) in enumerate(groupby(df, :hdbscan_cluster))
        if grp.hdbscan_cluster[1] == -1
            scatter!(p_hdbscan, grp."孕妇BMI", grp."检测孕周", markershape=:x, markercolor=:black, markersize=2, label="噪声点")
            continue
        end
        x_data = grp."孕妇BMI"
        y_data = grp."检测孕周"
        if length(x_data) > k
            model = loess(x_data, y_data)
            x_pred = collect(range(minimum(x_data), stop=maximum(x_data), length=50))
            y_pred = predict(model, x_pred)
            min_idx = argmin(y_pred)
            min_x = x_pred[min_idx]
            min_y = y_pred[min_idx]
            scatter!(p_hdbscan, [min_x], [min_y], markercolor=:red, markershape=:star5, markersize=8, markerstrokewidth=1, markerstrokecolor=:black, label=false)
            annotate!(p_hdbscan, min_x, min_y + 0.8, text("Min($(round(min_x, digits=1)), $(round(min_y, digits=1)))", 8, :center, :bottom, :red))
            plot!(p_hdbscan, x_pred, y_pred, linewidth=2, linestyle=:dash, linecolor=i, label=false)
        end
        scatter!(p_hdbscan, x_data, y_data, markercolor=i, markersize=2, label="Cluster $(grp.hdbscan_cluster[1])")
    end
    # 打表
    cluster_labels = assignments(hdbscan_clusters)
    temp_df = copy(df)
    temp_df[!, :cluster] = cluster_labels
    bmi_intervals = combine(groupby(temp_df, :cluster), "孕妇BMI" => minimum => :min_bmi, "孕妇BMI" => maximum => :max_bmi, "孕妇BMI" => mean => :mean_bmi, nrow => :count)
    bmi_intervals.min_cluster_size = fill(k, nrow(bmi_intervals))
    global all_hdbscan_intervals = vcat(all_hdbscan_intervals, bmi_intervals)
    df_abnormal = filter(:is_z_abnormal => x -> x == true, temp_df)
    scatter!(p_hdbscan, df_abnormal."孕妇BMI", df_abnormal."检测孕周", markercolor=:cyan, markershape=:rtriangle, label="Z值异常点", markersize=5)
    push!(hdbscan_plots, p_hdbscan)
end
hdbscan_com_plot = plot(hdbscan_plots..., layout=(2, 4), size=(1600, 800))
all_dbscan_intervals.algorithm .= "DBSCAN"
all_hdbscan_intervals.algorithm .= "HDBSCAN"
all_bmi_intervals = vcat(all_dbscan_intervals, all_hdbscan_intervals, cols=:union)

# 绘图、打表
savefig(k_distance_com_plot, joinpath(output_file_path, "k_distance_comparison2.png"))
savefig(dbscan_com_plot, joinpath(output_file_path, "dbscan_clusters2.png"))
savefig(hdbscan_com_plot, joinpath(output_file_path, "hdbscan_clusters2.png"))
CSV.write(joinpath(output_file_path, "bmi_intervals2.csv"), all_bmi_intervals)
