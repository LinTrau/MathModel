using CSV
using DataFrames
using Plots

input_file_path = abspath(joinpath(@__DIR__, "../output/processed_data2.csv"))
default(fontfamily="LXGWWenKai-Medium")
df = CSV.read(input_file_path, DataFrame)
scatter(df."孕妇BMI", df."检测孕周", xlabel="孕妇BMI", ylabel="检测孕周", title="BMI-孕周散点图", legend=false, size=(800, 600))
savefig(abspath(joinpath(@__DIR__, "../output/bmi_gestational_week_scatter.png")))