using JuMP, GLPK, DataFrames, CSV, Plots, XLSX

# --- 1. 数据准备与处理 ---
# 您指定的自定义文件路径
df_suppliers_supply = CSV.read("供货.csv", DataFrame)
df_suppliers_order = CSV.read("订货.csv", DataFrame)
df_transporter_loss = CSV.read("转运商相关数据.csv", DataFrame)
df_supplier_ranking = CSV.read("供应商排序结果.csv", DataFrame)

# --- 筛选出排名前50的供应商ID ---
top_50_suppliers = first(df_supplier_ranking, 50).suppliers

# --- 计算模型参数 ---
supplier_ids = df_suppliers_order[:, 1]
reliability_rates = Dict{String,Float64}()
max_supplies = Dict{String,Float64}()
supplier_materials = Dict{String,String}()

for supplier_id in top_50_suppliers
    row_idx = findfirst(isequal(supplier_id), supplier_ids)
    order_data = df_suppliers_order[row_idx, 3:end]
    supply_data = df_suppliers_supply[row_idx, 3:end]
    total_order = sum(skipmissing(order_data))
    total_supply = sum(skipmissing(supply_data))

    if total_order > 0
        reliability_rates[supplier_id] = total_supply / total_order
    else
        reliability_rates[supplier_id] = 0.0
    end
    max_supplies[supplier_id] = maximum(skipmissing(supply_data))
    supplier_materials[supplier_id] = df_suppliers_order[row_idx, 2]
end

# 转运商损耗率
transporter_ids = df_transporter_loss[:, 1]
loss_rates = Dict{Tuple{String,Int},Float64}()
for j in 1:8
    transporter_id = transporter_ids[j]
    for t in 1:240
        rate = df_transporter_loss[j, t+1]
        loss_rates[(transporter_id, t)] = rate / 100
    end
end

# --- 定义模型常量 ---
const weeks = 1:24
const suppliers = top_50_suppliers
const transporters = df_transporter_loss[:, 1]
const C_trans = 1.0
const C_storage = 0.5
const C_A = 1.2 * 100
const C_B = 1.1 * 100
const C_C = 100.0
const avg_weekly_demand = 2.82e4 * 0.72

# --- 2. 经济订购与最小损耗综合模型 ---
model = Model(GLPK.Optimizer)

# 决策变量
@variables(model, begin
    X[suppliers, weeks] >= 0      # 订购量
    W[suppliers, transporters, weeks] >= 0 # 转运量
    Y_received[weeks] >= 0        # 总接收量
    Z[weeks] >= 0                 # 总消耗量
    I[0:24] >= 0                  # 库存量
    Z_A[weeks] >= 0
    Z_B[weeks] >= 0
    Z_C[weeks] >= 0 # 各类原材料消耗量
end)

# 目标函数
@objective(model, Min,
    sum(
        (supplier_materials[i] == "A" ? C_A :
         supplier_materials[i] == "B" ? C_B : C_C) * X[i, t]
        for i in suppliers, t in weeks
    ) +
    sum(C_trans * Y_received[t] for t in weeks) +
    sum(C_storage * I[t] for t in weeks)
)

# 约束条件
# 产能约束 (产品量)
@constraint(model, production_demand[t in weeks],
    Z_A[t] / 0.6 + Z_B[t] / 0.66 + Z_C[t] / 0.72 == 2.82e4
)

# 总消耗量约束 
@constraint(model, total_consumption[t in weeks], Z[t] == Z_A[t] + Z_B[t] + Z_C[t])

# 库存平衡约束
@constraint(model, inventory_balance[t in weeks], I[t] == I[t-1] + Y_received[t] - Z[t])
@constraint(model, initial_inventory_is_zero, I[0] == 0)

# 修正后的最小库存约束
@constraint(model, final_min_inventory, I[24] >= 2 * avg_weekly_demand)

# 供货量与订货量关系 & 转运量与供货量关系
@constraint(model, supply_link[i in suppliers, t in weeks],
    sum(W[i, transporters[j], t] for j in 1:8) == reliability_rates[i] * X[i, t]
)

# 转运商运力约束
@constraint(model, transporter_capacity[j in 1:8, t in weeks],
    sum(W[i, transporters[j], t] for i in suppliers) <= 6000
)

# 总接收量计算（考虑损耗）
@constraint(model, total_received_volume[t in weeks],
    Y_received[t] == sum(W[i, transporters[j], t] * (1 - loss_rates[(transporters[j], t)]) for i in suppliers, j in 1:8)
)

# 供应商供货能力约束
@constraint(model, supplier_limit[i in suppliers, t in weeks], X[i, t] <= max_supplies[i])

# --- 求解模型 ---
optimize!(model)

# --- 3. 结果提取与可视化 ---

if termination_status(model) == MOI.OPTIMAL
    println("模型已成功求解。")
    
    # 提取结果
    total_cost = objective_value(model)
    ordering_plan = value.(X)
    transfer_plan = value.(W)
    inventory_levels = [value(I[t]) for t in 0:24]
    total_consumption = [value(Z[t]) for t in weeks]
    total_received = [value(Y_received[t]) for t in weeks]
    
    # 将结果打印到控制台
    println("\n--- 结果摘要 ---")
    println("总成本: ", round(total_cost, digits=2), " 元")
    println("最终库存量: ", round(inventory_levels[end], digits=2), " m³")
    println("最小库存底线: ", round(2 * avg_weekly_demand, digits=2), " m³")

    # --- 将结果写入到原始的 XLSX 文件中 ---
    # 1. 订购方案结果
    XLSX.openxlsx("附件A 订购方案数据结果.xlsx", mode="rw") do xf
        sheet_name = "问题2的订购方案结果"
        if sheet_name in XLSX.sheetnames(xf)
            sheet = xf[sheet_name]
            
            # 使用循环找到包含数据的最后一行，避免兼容性问题
            last_data_row = 6
            while true
                cell = XLSX.getcell(sheet, last_data_row + 1, 1)
                if isnothing(cell) || isa(cell, XLSX.EmptyCell)
                    break
                end
                last_data_row += 1
            end

            # 读取模板中的供应商ID，获取正确的写入顺序
            template_suppliers = [sheet[i, 1] for i in 7:last_data_row]
            
            # 遍历模板中的每个供应商，根据其ID精确写入结果
            for i in 1:length(template_suppliers)
                supplier_id_read = template_suppliers[i]
                if ismissing(supplier_id_read)
                    continue
                end
                supplier_id = String7(supplier_id_read)
                for t in weeks
                    col_idx = t + 1
                    try
                        cell_value = round(ordering_plan[supplier_id, t], digits=2)
                        sheet[6 + i, col_idx] = cell_value
                    catch e
                        if isa(e, BoundsError) || isa(e, KeyError)
                            sheet[6 + i, col_idx] = 0.0
                        else
                            rethrow(e)
                        end
                    end
                end
            end
            println("\n已成功将订购方案结果写入到 '附件A 订购方案数据结果.xlsx' 的 '", sheet_name, "' 工作表中。")
        else
            println("\n警告: 未找到 '", sheet_name, "' 工作表，请检查文件名。")
        end
    end

    # 2. 转运方案结果
    XLSX.openxlsx("附件B 转运方案数据结果.xlsx", mode="rw") do xf
        sheet_name = "问题2的转运方案结果"
        if sheet_name in XLSX.sheetnames(xf)
            sheet = xf[sheet_name]
            
            # 使用循环找到包含数据的最后一行，避免兼容性问题
            last_data_row = 5
            while true
                cell = XLSX.getcell(sheet, last_data_row + 1, 1)
                if isnothing(cell) || isa(cell, XLSX.EmptyCell)
                    break
                end
                last_data_row += 1
            end
            
            # 读取模板中的供应商ID，获取正确的写入顺序
            template_suppliers = [sheet[i, 1] for i in 6:last_data_row]
            
            # 遍历模板中的每个供应商，根据其ID精确写入结果
            for i in 1:length(template_suppliers)
                supplier_id_read = template_suppliers[i]
                if ismissing(supplier_id_read)
                    continue
                end
                supplier_id = String7(supplier_id_read)
                for t in weeks
                    for j in 1:length(transporters)
                        transporter_id = transporters[j]
                        
                        # 计算正确的列索引，注意模板格式是每个周有8列（8个转运商）
                        col_idx = 1 + (t - 1) * 8 + j
                        
                        # 检查该供应商和转运商的组合是否在模型结果中，并进行类型转换
                        try
                            cell_value = round(transfer_plan[supplier_id, transporter_id, t], digits=2)
                            sheet[5 + i, col_idx] = cell_value
                        catch e
                            if isa(e, BoundsError) || isa(e, KeyError)
                                sheet[5 + i, col_idx] = 0.0
                            else
                                rethrow(e)
                            end
                        end
                    end
                end
            end
            println("已成功将转运方案结果写入到 '附件B 转运方案数据结果.xlsx' 的 '", sheet_name, "' 工作表中。")
        else
            println("警告: 未找到 '", sheet_name, "' 工作表，请检查文件名。")
        end
    end

    # --- 可视化 ---
    # 1. 库存与生产走势图
    p1 = plot(0:24, inventory_levels, label="库存量", xlabel="周数", ylabel="数量 (m³)", 
              title="24周库存与生产走势", linewidth=2, fontfamily="LXGWWenKai-Medium")
    plot!(p1, 1:24, total_received, label="总接收量", markershape=:circle, fontfamily="LXGWWenKai-Medium")
    plot!(p1, 1:24, total_consumption, label="总消耗量", markershape=:star, fontfamily="LXGWWenKai-Medium")
    hline!([2 * avg_weekly_demand], label="最小库存底线", linestyle=:dash, linecolor=:red, linewidth=2, fontfamily="LXGWWenKai-Medium")
    savefig(p1, "inventory_production_plot.png")

    # 2. 供应商贡献排行榜 (生成条形图并输出CSV)
    total_orders = [sum(value(X[i, t]) for t in weeks) for i in suppliers]
    df_orders = DataFrame(Supplier = suppliers, TotalOrders = total_orders)
    sort!(df_orders, :TotalOrders, rev=true)
    
    # 将完整的供应商排名数据保存到CSV文件
    CSV.write("supplier_ranking.csv", df_orders)
    println("已生成详细供应商订购量排名文件：supplier_ranking.csv")

    p2 = bar(df_orders.Supplier, df_orders.TotalOrders, 
             label="总订购量", 
             title="供应商订购量贡献排行榜（全部50家）", 
             xlabel="供应商", 
             ylabel="总订购量 (m³)",
             legend=false,
             xrotation=90,
             size=(1200, 600),
             bottom_margin=10Plots.mm,
             fontfamily="LXGWWenKai-Medium")
    savefig(p2, "supplier_contribution.png")

    # 3. 转运商任务分配图
    total_transfers = [sum(value(W[i, transporters[j], t]) for i in suppliers, t in weeks) for j in 1:8]
    p3 = bar(transporters, total_transfers, label="总转运量", xlabel="转运商ID", ylabel="数量 (m³)", title="转运商任务分配", fontfamily="LXGWWenKai-Medium")
    hline!([24 * 6000], label="总运力上限", linestyle=:dash, linecolor=:red, fontfamily="LXGWWenKai-Medium")
    savefig(p3, "transporter_allocation.png")

else
    println("求解失败，状态: ", termination_status(model))
end
