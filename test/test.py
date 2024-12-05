import gurobipy as gp

# 确保你的 .lp 文件路径是正确的
lp_file_path = '../ErrorModel.lp'

# 尝试加载模型并求解
try:
    # 加载 .lp 文件
    m = gp.read(lp_file_path)

    # 设置参数（如果需要）
    # m.Params.method = 2  # 例如，设置求解器为 barrier 方法

    # 求解模型
    m.optimize()

    # 输出结果
    if m.status == gp.GRB.OPTIMAL:
        print('Optimal objective value:', m.objVal)
        for v in m.getVars():
            print(f'{v.varName}: {v.x}')
    else:
        print(m.getConstrs())  # 查看所有约束
        print(m.getVars())  # 查看所有变量

        print('Optimization was stopped with status:', m.status)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))

except AttributeError:
    print('Encountered an attribute error')
