using CSV
using DataFrames
using Plots
using GLM
using MultivariateStats
using Statistics


input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data2.csv")) 
df = CSV.read(input_file_path, DataFrame)
default(fontfamily="LXGWWenKai-Medium")

X_vars = ["原始读段数", "唯一比对的读段数", "被过滤掉读段数的比例", "重复读段的比例", "在参考基因组上比对的比例"]
Y_var = "Y染色体浓度"
X = Matrix(select(df, Symbol.(X_vars)))
Y = vec(df[!, Symbol(Y_var)]) 
# Z标准化
X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X_norm = (X .- X_mean) ./ X_std
# 主成分回归
pca_model = fit(PCA, X_norm'; maxoutdim=3)
X_pca = MultivariateStats.transform(pca_model, X_norm')'
pcr_df = DataFrame(X_pca, Symbol.("PC" .* string.(1:3)))
pcr_df[!, :Y] = Y
pcr_formula = @formula(Y ~ PC1 + PC2 + PC3)
pcr_model = lm(pcr_formula, pcr_df)
loadings = projection(pca_model)
pcr_coeffs_pc = coef(pcr_model)[2:end]
pcr_coeffs_orig = loadings * pcr_coeffs_pc
Y_pcr_pred = MultivariateStats.predict(pcr_model, pcr_df[:, 1:3])
pcr_residuals = Y - Y_pcr_pred
# 作图
p_pcr_res = scatter(Y_pcr_pred, pcr_residuals, xlabel="预测值", ylabel="残差", title="残差图", legend=false)
hline!(p_pcr_res, [0], color=:red, linestyle=:dash)
p_pcr_act_pred = scatter(Y, Y_pcr_pred, xlabel="实际值", ylabel="预测值", title="实际值—预测值", legend=false)
plot!(p_pcr_act_pred, [minimum(Y), maximum(Y)], [minimum(Y), maximum(Y)], color=:red, linestyle=:dash)
savefig(p_pcr_res, joinpath(@__DIR__, "../output/pcr_residual_plot.png"))
savefig(p_pcr_act_pred, joinpath(@__DIR__, "../output/pcr_actual_vs_predicted.png"))
