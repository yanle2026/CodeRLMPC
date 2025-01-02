height = [2,0,1,0,2,1,0,1,3,2,1,2,1]
print(height.index(max(height)))
a = height.index(max(height))
height[a] = 0
print(height)
br = height.index(max(height[:a]))
bl = height.index(max(height[a:]))
sumbl = 0
while True:
    # 这里是 do-while 循环的循环体
    # 执行你想要的操作
    sumbl = sum(height[bl] - height[_] for _ in height[bl:a])
    # 在循环体的末尾检查条件
    # 如果条件不满足，则退出循环
    if bl is 0:
        break


height[a] = 0
print(height)