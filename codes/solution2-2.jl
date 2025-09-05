using CSV
using DataFrames
using Plots
using GLM
using MultivariateStats
using Statistics
using StatsPlots


input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data2.csv"))
df = CSV.read(input_file_path, DataFrame)
default(fontfamily="LXGWWenKai-Medium")

X_vars = ["原始读段数", "唯一比对的读段数", "被过滤掉读段数的比例", "重复读段的比例", "在参考基因组上比对的比例"]
Y_var = "Y染色体浓度"
X = Matrix(select(df, Symbol.(X_vars)))
Y = vec(df[!, Symbol(Y_var)])
correlation_matrix = cor(X)
# Z标准化
X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X_norm = (X .- X_mean) ./ X_std
# 主成分回归
pca_model = fit(PCA, X_norm'; maxoutdim=3)
loadings = projection(pca_model)
X_pca = MultivariateStats.transform(pca_model, X_norm')'
pcr_df = DataFrame(X_pca, Symbol.("PC" .* string.(1:3)))
pcr_df[!, :Y] = Y
pcr_formula = @formula(Y ~ PC1 + PC2 + PC3)
pcr_model = lm(pcr_formula, pcr_df)
pcr_coeftable = coeftable(pcr_model)
r_squared = r2(pcr_model)
pcr_coeffs_pc = coef(pcr_model)[2:end]
pcr_coeffs_orig = loadings * pcr_coeffs_pc
Y_pcr_pred = MultivariateStats.predict(pcr_model, pcr_df[:, 1:3])
pcr_residuals = Y - Y_pcr_pred
# 作图
p_pcr_res = scatter(Y_pcr_pred, pcr_residuals, xlabel="预测值", ylabel="残差", title="残差图", legend=false)
hline!(p_pcr_res, [0], color=:red, linestyle=:dash)
p_pcr_act_pred = scatter(Y, Y_pcr_pred, xlabel="实际值", ylabel="预测值", title="实际值—预测值", legend=false)
plot!(p_pcr_act_pred, [minimum(Y), maximum(Y)], [minimum(Y), maximum(Y)], color=:red, linestyle=:dash)
p_corr = heatmap(correlation_matrix, xlabel="各因素", ylabel="各因素", title="各因素相关性热力图", xtick=(1:size(X, 2), X_vars), ytick=(1:size(X, 2), X_vars), xrotation=45, yrotation=0, aspect_ratio=:equal, colorbar_title="相关系数", right_margin=10Plots.mm)
# 打表
all_results_df = DataFrame(
    Metric = Union{String, Missing}[],
    Value = Union{Float64, Missing}[],
    Variable = Union{String, Missing}[],
    Coefficient = Union{Float64, Missing}[],
    PC1 = Union{Float64, Missing}[],
    PC2 = Union{Float64, Missing}[],
    PC3 = Union{Float64, Missing}[],
    StdError = Union{Float64, Missing}[],
    TValue = Union{Float64, Missing}[],
    PValue = Union{Float64, Missing}[])
explained_variance = principalvars(pca_model) ./ sum(principalvars(pca_model))
pca_results_df = DataFrame(Metric=["PCA 解释方差贡献率"], PC1=explained_variance[1], PC2=explained_variance[2], PC3=explained_variance[3])
append!(all_results_df, pca_results_df, cols=:union)
push!(all_results_df, fill(missing, size(all_results_df, 2)))
loadings_df = DataFrame(Metric = fill("PCA载荷", 5), Variable = X_vars, PC1 = loadings[:, 1], PC2 = loadings[:, 2], PC3 = loadings[:, 3])
append!(all_results_df, loadings_df, cols=:union)
push!(all_results_df, fill(missing, size(all_results_df, 2)))
r_squared_df = DataFrame(Metric=["主成分回归模型的R²值"], Value=[r2(pcr_model)])
append!(all_results_df, r_squared_df, cols=:union)
push!(all_results_df, fill(missing, size(all_results_df, 2)))
coeftable_df = DataFrame(Metric = fill("主成分回归系数表", 4),Variable = ["(截距)", "PC1", "PC2", "PC3"],Coefficient = coeftable(pcr_model).cols[1],StdError = coeftable(pcr_model).cols[2],TValue = coeftable(pcr_model).cols[3],PValue = coeftable(pcr_model).cols[4])
append!(all_results_df, coeftable_df, cols=:union)
push!(all_results_df, fill(missing, size(all_results_df, 2)))
pcr_coeffs_orig_df = DataFrame(Metric = fill("原始变量回归系数", 5), Variable = X_vars, Coefficient = pcr_coeffs_orig)
append!(all_results_df, pcr_coeffs_orig_df, cols=:union)

savefig(p_corr, joinpath(@__DIR__, "../output/correlation_heatmap.png"))
savefig(p_pcr_res, joinpath(@__DIR__, "../output/pcr_residual_plot.png"))
savefig(p_pcr_act_pred, joinpath(@__DIR__, "../output/pcr_actual_vs_predicted.png"))
CSV.write(joinpath(@__DIR__, "../output/analysis_results2.csv"), all_results_df)