# 脚本 14 逐行学习笔记

对应脚本：`scripts_archive/14_grasp_yellow_tube_cap_up.py`

这份笔记面向 Python 初学者。目标不是要求你第一次就记住所有公式，而是让你知道：

1. 每一部分代码接收什么；
2. 它做了什么计算；
3. 它输出什么；
4. 为什么机械臂需要这一步。

> 建议先运行 `python scripts_archive/14_grasp_yellow_tube_cap_up.py --help`，再按本文的阅读顺序看代码。
> 不要一开始就从第一行顺序读到底，因为数学工具函数只有放到主流程中才容易理解。

## 一、推荐阅读顺序

按下面顺序阅读，比从第一行硬啃更容易：

1. `main()`：先看程序总入口。
2. `detect_and_lock()`：看相机如何得到黄色试管。
3. `make_solution()`：看像素如何变成机械臂位置和 yaw。
4. `build_cap_up_plan()`：看程序如何搜索一条可达路径。
5. `execute_plan()`：看真正的动作顺序。
6. `SimpleCArm`：再理解 IK、FK 和运动命令。
7. 四元数函数：最后研究姿态旋转的数学细节。

整个数据流可以压缩成一句话：

```text
RGB 图像
  -> YOLO 分割 y-body / y-cap
  -> 图像中心、B 点、C 点
  -> T_arm2cam 转换为机械臂 XY 和 B->C yaw
  -> 搜索 IK 路径
  -> 下降抓取
  -> 45°、90° 两段转姿
  -> 核对固定凹槽高空位和释放位
  -> 依次移动到两个固定凹槽关节位
  -> 打开夹爪
  -> 移动到固定撤离关节位
  -> 回到六关节零位
```

## 二、运行模式

脚本有三种模式。

### 1. 只做视觉预览

```bash
python scripts_archive/14_grasp_yellow_tube_cap_up.py
```

脚本14在 `model.predict()` 中固定使用 FP32。

相机画面稳定后按 `C`，程序只打印目标位置，不连接机械臂。

实时画面只保留分割掩膜，不显示 bounding box、类别文字和置信度。蓝色 `B`
表示试管无盖端，红色 `C` 表示靠近黄色盖子的管身端点，黄色线表示 B→C。
画面中的 `IMAGE B->C ANGLE` 使用图像像素坐标系：向右是 `+u`，向下是 `+v`。
它用于检查分割得到的图像方向，不能直接发送给机械臂。目标稳定后，画面还会
显示机械臂基座 XY 平面内的 `BASE B->C YAW`；它与后续机械臂路径规划使用的
`tube_yaw` 是同一个角度。把两个角度同时显示，有助于检查手眼坐标变换是否符合
预期。

### 2. 只读检查 IK

```bash
python scripts_archive/14_grasp_yellow_tube_cap_up.py \
  --ip 10.42.0.101 \
  --check-ik
```

程序连接控制器并读取关节状态，计算 IK/FK，但不调用 `ready()`，也不发送运动命令。

### 3. 真实执行

```bash
python scripts_archive/14_grasp_yellow_tube_cap_up.py \
  --ip 10.42.0.101 \
  --execute
```

这会真实使能并运动。应先完成同一摆放状态下的 `--check-ik`，确认完整扫掠区域无障碍并准备急停。

## 三、最重要的变量、单位和坐标系

### 1. 单位

| 内容 | 单位 |
|---|---|
| 图像点 `u, v` | 像素 px |
| 标定矩阵中的平移 | 毫米 mm |
| CArm Pose 的 `x, y, z` | 米 m |
| Python 三角函数的角度 | 弧度 rad |
| 打印给人看的角度 | 度 ° |
| 四元数 | 无单位 |

代码里最容易犯错的是毫米和米混用。因此 `pixel_to_arm()` 最后明确除以 `1000.0`。

### 2. 两个坐标系

- 相机坐标系：三维射线最初所在的坐标系。
- 机械臂基座坐标系：CArm 的目标 Pose 所在坐标系。

工作台不再单独建立坐标系，而是在机械臂基座坐标系中直接表示为固定平面：

```text
Z = 68.124833 mm
```

`pixel_to_arm()` 完成：

```text
像素 -> 相机射线 -> 机械臂射线 -> 固定工作台平面交点
```

### 3. B 点和 C 点

- `B`：试管无盖的一端。
- `C`：黄色管盖的一端。
- `B -> C`：从无盖端指向盖子端的有向向量。

有了方向而不只是“无方向长轴”，程序才知道最后应该把哪一端转到基座 `+Z`，也就是盖子朝上。

## 四、Python 语法知识

## 4.1 文件入口和模块说明

```python
#!/usr/bin/env python3
"""模块说明。"""
```

- 第一行叫 shebang。在 Linux 中直接执行文件时，它告诉系统使用 Python 3。
- 三引号字符串放在文件开头会成为模块文档字符串，可通过 `__doc__` 读取。

```python
if __name__ == "__main__":
    ...
```

- 直接运行该文件时，`__name__` 等于 `"__main__"`。
- 其他文件 `import` 它时，条件不成立，因此不会自动启动相机或机械臂。

## 4.2 导入

```python
import math
import numpy as np
from pathlib import Path
```

- `import math`：使用 `math.sin()`、`math.pi`。
- `as np`：给 NumPy 一个常用短名。
- `from ... import ...`：只导入模块中的指定名字。

脚本还在函数内部导入 `YOLO`、`pyrealsense2` 和 `Carm`，这叫延迟导入。好处是只运行 `--help` 或只做视觉时，不会过早加载不需要的硬件库。

## 4.3 列表、元组、字典和 NumPy 数组

```python
list_value = [1, 2, 3]
tuple_value = (1, 2, 3)
dict_value = {"x": 1, "y": 2}
array_value = np.array([1, 2, 3])
```

- 列表 `list` 可修改，适合逐步 `append()`。
- 元组 `tuple` 通常表示不应修改的固定数据。
- 字典 `dict` 用键名保存一组相关结果。
- NumPy 数组支持向量和矩阵运算。

例如：

```python
ROTATE_XY = (0.240, -0.010)
```

这个二元素元组直接保存固定高空转姿点的机械臂基座 `(x, y)`，单位为米。规划时使用：

```python
rotate_x, rotate_y = ROTATE_XY
```

## 4.4 类型提示

```python
def normalize_quat(quaternion: Sequence[float]) -> tuple[float, ...]:
```

- `quaternion: Sequence[float]`：参数应是浮点数序列，列表或元组都可以。
- `-> tuple[float, ...]`：返回任意长度的浮点元组；本函数实际固定返回四个数。
- 类型提示帮助阅读器和编辑器，不会自动替你验证所有运行时类型。

```python
serial: str | None
```

表示 `serial` 可以是字符串，也可以是 `None`。

```python
Any
```

表示这里兼容多种类型，通常用于第三方库可能返回不同格式的地方。

## 4.5 函数、默认参数和返回值

```python
def tilt_angles() -> list[float]:
    return [45.0, 90.0]
```

- `def` 定义函数。
- 空括号表示这个函数不需要接收参数。
- `return` 立即结束函数并返回结果。

## 4.6 序列解包和字典复制

```python
x, y, _ = solution["center_arm"]
```

这是序列解包。三个元素分别赋值给 `x`、`y`、`_`。下划线表示第三个值有意不用。

规划器需要在原字典上增加 Pose 和关节解。现在使用容易跟踪的复制和赋值：

```python
step = item.copy()
step["pose"] = target_pose
step["joints"] = joints
```

`copy()` 产生一个新字典，后面的赋值不会修改原来的 `item`。

## 4.7 用普通循环建立列表

脚本避免使用较紧凑的列表推导式，直接写出循环过程：

```python
ordered_names = []
for class_name in names:
    ordered_names.append(str(class_name))
```

`append()` 每次向列表尾部添加一个元素。这种写法行数更多，但便于初学者逐步观察变量。

## 4.8 直接执行并检查 SDK 命令

```python
response = self.arm.set_tool_index(TOOL)
self.check_command("set_tool_index", response)
```

第一行真实调用 CArm SDK；第二行检查返回值。当前脚本不再使用 `lambda` 隐藏命令调用时机。

## 4.9 条件、真值和普通 if/else

```python
if twist_deg:
```

数字 `0` 视为 False，非零数字视为 True。

以前可以用一行条件表达式选择状态文字；现在展开成普通分支：

```python
if solution is not None:
    status_text = "STABLE - C: lock"
else:
    status_text = "Waiting for y-body/y-cap"
```

## 4.10 普通循环和手动编号

```python
yaw_index = 0
for yaw in candidate_yaws:
    yaw_index += 1
```

每处理一个 yaw，编号增加 1。它比 `enumerate()` 更长，但执行过程更直观。

路径搜索使用多层循环，是在依次尝试：

1. 高度；
2. 两个 yaw；
3. 转姿点；
4. 腕部 twist；
5. 45°、90° 两个转姿目标。

内层任何一步发生 `RuntimeError`，`continue` 会跳到下一个候选。

## 4.11 类和 self

```python
class RealSenseColor:
    def __init__(self, serial):
        self.pipeline = ...
```

- `class` 把数据和相关操作组合起来。
- `__init__` 在创建对象时执行。
- `self.pipeline` 属于当前对象，其他方法也能使用。

`SimpleCArm` 同样把连接、IK、FK、运动和夹爪操作包在一个对象中。

## 4.12 异常和资源清理

```python
try:
    ...
except RuntimeError as exc:
    ...
finally:
    robot.close()
```

- `try`：尝试执行可能失败的代码。
- `except`：捕获指定错误。
- `as exc`：把错误对象保存到变量。
- `finally`：无论成功还是失败都会执行，适合关闭相机和网络连接。

```python
raise ValueError("说明")
```

主动停止当前流程，并把原因告诉调用者。

```python
raise RuntimeError("新说明") from exc
```

把底层异常作为原因保留下来，形成异常链。

## 4.13 with 文件上下文

```python
with path.open(encoding="utf-8") as file:
    camera = yaml.safe_load(file)
```

离开 `with` 代码块时文件会自动关闭，即使读取失败也一样。

## 4.14 f-string

```python
f"x={x:.3f}, yaw={yaw:.1f}"
```

- `{x:.3f}`：保留三位小数。
- `{yaw:.1f}`：保留一位小数。

脚本使用完整确认字符串，是为了让操作者再次看见本次目标位置和姿态。

## 五、图像与分割数学

## 5.1 为什么要找最大连通区域

YOLO 掩膜可能包含少量孤立噪点。`connectedComponentsWithStats()` 会给每一块相连区域编号并计算面积，脚本只保留面积最大的物体区域。

这不是重新做识别，只是清理掩膜。

## 5.2 为什么 `np.nonzero()` 写成 `ys, xs`

图像数组的索引顺序是 `[行, 列]`，也就是 `[y, x]`。因此：

```python
ys, xs = np.nonzero(mask)
points = np.column_stack((xs, ys))
```

转成几何点时又改回常见的 `[x, y]` 顺序。

## 5.3 PCA 如何求试管长轴

管身掩膜由很多二维像素点组成。先求中心：

```text
center = 所有像素点的平均值
```

再把每个点减去中心，并计算协方差矩阵：

```text
            [ var(x)    cov(x,y) ]
Covariance =[                     ]
            [ cov(x,y)  var(y)   ]
```

对称协方差矩阵有两个互相垂直的特征向量：

- 较大特征值对应像素分布最伸展的方向；
- 对细长试管来说，这就是长轴方向。

代码：

```python
values, vectors = np.linalg.eigh(covariance)
axis = vectors[:, np.argmax(values)]
```

然后把全部像素投影到主轴：

```text
projection_i = (point_i - center) dot axis
```

脚本取 1% 和 99% 分位点作为两个端点，而不是直接取最小/最大值，以减弱孤立噪点影响。

PCA 只能得到“轴”，不能自己判断哪端是盖子。脚本另外计算 y-cap 中心，离管盖最近的端点定义为 C，另一端定义为 B。

## 5.4 多帧稳定与圆周平均

程序保存最近 10 帧结果：

```python
deque(maxlen=10)
```

中心位置使用中位数，并要求每帧中心到中位中心不超过 3 px。

角度不能直接用普通平均。例如：

```text
+179° 和 -179°
```

它们实际几乎同方向，普通平均却是 0°。正确的圆周平均是：

```text
mean_angle = atan2(mean(sin(angle_i)), mean(cos(angle_i)))
```

## 六、像素到机械臂坐标的数学

## 6.1 相机内参矩阵

针孔相机内参通常写成：

```text
    [ fx  0  cx ]
K = [ 0  fy  cy ]
    [ 0   0   1 ]
```

- `fx, fy`：以像素为单位的焦距。
- `cx, cy`：主点，通常靠近图像中心。

像素 `[u,v]` 对应的相机射线方向满足：

```text
K * ray_camera = [u, v, 1]^T
```

脚本用：

```python
ray_camera = np.linalg.solve(K, [u, v, 1.0])
```

这与 `inv(K) @ pixel` 数学上等价，但直接解方程通常更合适。

## 6.2 齐次变换矩阵

三维刚体变换写成 4×4 矩阵：

```text
T = [ R  t ]
    [ 0  1 ]
```

- `R` 是 3×3 旋转矩阵。
- `t` 是 3×1 平移向量。

三维点需要补一个 1：

```text
[x, y, z, 1]^T
```

这样旋转和平移可以在一次矩阵乘法中完成。

## 6.3 用 T_arm2cam 直接求工作台交点

`T_arm2cam.txt` 在当前标定工程中的实际作用是把相机坐标变换到机械臂基座坐标。先把相机射线变换到机械臂坐标系：

```text
origin_arm = T_arm2cam 的平移部分
ray_arm = T_arm2cam 的旋转部分 * ray_camera
```

射线方程是：

```text
p(s) = origin_arm + s * ray_arm
```

工作台平面是 `z=68.124833 mm`，所以：

```text
s = (table_z - origin_arm_z) / ray_arm_z
```

把 `s` 代回射线方程，就直接得到机械臂基座坐标，不再经过工作台坐标系。

这个方法的关键假设是：用于定位的中心点确实可以投影到固定工作台平面。如果物体所在平面高度改变，XY 也会产生系统误差。

## 七、yaw、四元数和盖朝上

## 7.1 机械臂平面 yaw

把 B、C 都变换到机械臂坐标系后：

```text
direction = C - B
yaw = atan2(direction_y, direction_x)
```

`atan2(y,x)` 能正确区分四个象限，输出范围通常为 `[-pi, pi]`。

## 7.2 为什么有两个 yaw 候选

对称夹爪沿一条轴抓取时，下面两个方向通常都能夹住：

```text
yaw_2 = yaw_1 + 180°
```

但机械臂的关节可达性可能不同，所以让 IK 分别尝试两个方向。

## 7.3 为什么机械臂姿态使用四元数

CArm Pose 是：

```text
[x, y, z, qx, qy, qz, qw]
```

前三个数是位置，后四个数是单位四元数。单位四元数满足：

```text
qx² + qy² + qz² + qw² = 1
```

它比直接叠加欧拉角更适合连续组合三维旋转，也能避免某些万向节锁问题。

## 7.4 绕 Z 轴的 yaw 四元数

绕 Z 轴转 `yaw`：

```text
q_yaw = [0, 0, sin(yaw/2), cos(yaw/2)]
```

注意使用的是半角。脚本再做 Hamilton 乘积：

```text
q_target = q_yaw * q_down
```

这样既保留夹爪向下，又让夹爪在水平面内与试管对齐。

## 7.5 轴角转四元数

绕单位轴 `a=[ax,ay,az]` 旋转角度 `theta`：

```text
q = [ax*sin(theta/2),
     ay*sin(theta/2),
     az*sin(theta/2),
     cos(theta/2)]
```

`axis_angle_quat()` 就实现这个公式。

## 7.6 为什么翻转轴用叉积

初始 B→C 方向是工作台内的水平向量：

```text
d = [cos(yaw), sin(yaw), 0]
```

目标方向是基座竖直向上：

```text
z = [0, 0, 1]
```

垂直于二者的旋转轴可由叉积得到：

```text
axis = d × z
```

绕该轴旋转 90°，理论上就能把 B→C 从水平转到 `+Z`。脚本把动作分成 45° 和 90° 两段：先求并执行 45° 姿态，再把它的关节解作为 90° IK 的 seed。这样 IK 求解过程与真机实际运动顺序一致。

## 7.7 twist 是什么

`twist` 是绕试管自身长轴旋转夹爪：

- 不改变 B→C 指向；
- 不改变盖子最终朝上这一目标；
- 会改变腕部关节姿态，因此可能把无解路径变成可达路径。

规划器按 `0, +15, -15, +30, -30, ...` 依次尝试，找到第一条完整可达路径就停止。

## 7.8 四元数乘法顺序

三维旋转组合不满足交换律：

```text
q1 * q2 通常不等于 q2 * q1
```

脚本里的约定是用左乘附加新的基座坐标系旋转。不要在不验证几何意义和机械臂结果的情况下随意交换乘法顺序。

## 7.9 用四元数旋转向量

理论公式：

```text
v_rotated = q * [v,0] * conjugate(q)
```

`rotate_vector()` 使用它的展开形式，避免临时构造向量四元数。

单位四元数的共轭是：

```text
conjugate([qx,qy,qz,qw]) = [-qx,-qy,-qz,qw]
```

它等于该旋转的逆。

## 八、IK、FK 和路径搜索

## 8.1 FK 是什么

正运动学 Forward Kinematics：

```text
六个关节角 -> TCP 位置与姿态
```

脚本用 `forward_kine()` 得到 `[x,y,z,qx,qy,qz,qw]`。

## 8.2 IK 是什么

逆运动学 Inverse Kinematics：

```text
TCP 位置与姿态 + 初始关节 seed -> 一组六关节角
```

一个 Pose 可能：

- 没有解；
- 有一组解；
- 有多组不同肘部/腕部构型的解。

因此 seed 很重要。脚本每段使用上一段关节角作为下一段 seed，追求连续解。

## 8.3 `ik()` 做了哪些检查

脚本不是只看 SDK 是否返回数字，还检查：

1. 返回值是不是 6 个有限关节角；
2. 是否距离每个关节限位至少 2°；
3. 单段最大关节变化是否超过当前 90° 限制；
4. 把 IK 解代回 FK 后，位置是否在 2 mm 内；
5. 姿态是否在 1° 内。

四元数姿态误差使用：

```text
angle_error = 2 * acos(|q_actual dot q_target|)
```

取绝对值是因为 `q` 和 `-q` 表示完全相同的旋转。

这些是规划目标一致性检查，不等同于完整的碰撞规划。脚本仍依赖 CArm 控制器的碰撞功能以及操作者对真实扫掠区域的检查。

## 8.4 路径搜索顺序

`build_cap_up_plan()` 做只读枚举：

```text
固定高度
  -> 两个抓取 yaw
    -> 试管更高上方避障点
    -> 固定高空转姿点
      -> 多个 twist
        -> 45° IK
          -> 90° IK
            -> 核对固定凹槽高空位
              -> 核对固定凹槽释放位
```

凹槽高空位关节角是
`[-0.986911, 1.985130, -0.395399, -0.727665, -1.554320, -0.039102] rad`；
新的释放位关节角是
`[-0.988061, 2.029240, -0.378233, -0.729191, -1.570800, -0.081827] rad`。
松爪后的固定撤离位关节角是
`[-1.075880, 1.808340, -0.193980, -0.729572, -1.570800, -0.081827] rad`。
规划器不为凹槽搜索 IK，只检查盖朝上姿态到固定凹槽高空位的关节变化。
简单版不再计算或限制固定释放位的盖子倾角。

## 8.5 关节运动与直线运动

- `move_joints()`：关节空间插补，TCP 轨迹不保证是直线。
- `move_line()`：要求 TCP 在笛卡尔空间沿直线移动。

本脚本只在试管抓取位置使用直线下降、抬高；准备、转移、转姿、凹槽高空位
和释放位都使用关节运动。凹槽阶段直接使用现场记录的关节角，不重新求 IK。

## 九、主流程逐步解释

`main()` 的实际逻辑如下：

1. `parse_args()` 读取命令行。
2. `load_calibration()` 读取相机内参和 `T_arm2cam.txt`。
3. `detect_and_lock()` 打开相机，分割 y-body/y-cap。
4. 最近 10 帧稳定后，用户按 `C` 锁定。
5. `pixel_to_arm()` 计算中心、B、C 的机械臂坐标。
6. `make_solution()` 得到中心 XY、试管 yaw 和两个抓取 yaw。
7. 若只是预览，打印后退出。
8. 若 `--check-ik` 或 `--execute`，连接 CArm 并确认起点接近零位。
9. `build_cap_up_plan()` 只读搜索完整路径。
10. `--check-ik` 到此结束，不使能、不运动。
11. `--execute` 进入 `execute_plan()`，要求输入完整确认文字。
12. 到准备位。
13. 到试管更高上方 `APPROACH_Z_M=0.350 m`，张开夹爪。
14. 从更高上方直接直线下降到抓取高度，等待，闭爪。
15. 直线抬回更高上方，移动到固定高空转姿点。
16. 经过 45°、90° 两个姿态，把 C 端转向基座 `+Z`。
17. 移动到现场记录的固定凹槽高空位。
18. 移动到最新截图对应的固定凹槽释放位。
19. 打开夹爪。
20. 移动到固定撤离关节位。
21. 回到六关节零位。
22. `finally` 断开机械臂连接。

## 十、当前简单版的边界

为了让代码容易学习，这个版本没有实现完整工业运动规划器。需要明确：

- 只处理画面里恰好一个 `y-body` 和一个 `y-cap`。
- 用工作台固定平面求三维点，没有使用深度流。
- PCA 得到的是掩膜几何长轴，结果依赖分割质量。
- 固定高空转姿点来自当前工作空间的只读 IK 验证，不代表任意环境都无碰撞。
- IK/FK 通过只说明目标在运动学上可求解，不证明真实扫掠路径无障碍。
- `check_command()` 只确认当前 SDK 返回 `True` 或 `Task_Recieve`，没有逐动作读取实际 TCP 再做闭环到位验证。
- 当前调试版到达固定凹槽释放位后会打开夹爪、移动到固定撤离位，并回到六关节零位。
- 相机、夹爪、工具 TCP、手眼矩阵或工作台位置改变后，需要重新验证。

## 十一、适合初学者的无运动练习

### 练习 1：理解角度归一化

不要连接相机和机械臂，单独在 Python 中试：

```python
import math

def wrap_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

for degree in (0, 180, 190, 370, -190):
    print(degree, "->", math.degrees(wrap_pi(math.radians(degree))))
```

### 练习 2：理解圆周平均

```python
import math
import numpy as np

angles = np.radians([179, -179])
mean = math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
print(math.degrees(mean))
```

结果应接近 ±180°，而不是 0°。

### 练习 3：理解矩阵乘法和元素乘法

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("元素乘法：\n", A * B)
print("矩阵乘法：\n", A @ B)
```

### 练习 4：只做视觉

把黄色试管换几个随机平面方向，每次锁定后观察打印的 `B→C yaw` 是否随摆放方向连续变化。这个过程不会连接机械臂。

### 练习 5：只读 IK

视觉结果稳定后运行 `--check-ik`，记录程序选择的：

- yaw 候选编号；
- 转姿高度；
- twist。

改变试管平面方向后重复，理解为什么随机摆放会得到不同的 IK 路径。

## 十二、术语速查

| 术语 | 本脚本中的意思 |
|---|---|
| RGB | RealSense 彩色图像 |
| mask | YOLO 分割出的像素区域 |
| PCA | 从管身掩膜估计长轴 |
| B | 无盖端 |
| C | 管盖端 |
| yaw | 基座 XY 平面内绕 Z 的方向角 |
| Pose | `[x,y,z,qx,qy,qz,qw]` |
| TCP | 工具中心点，这里对应 tool=1 夹爪 |
| FK | 关节角求末端 Pose |
| IK | 末端 Pose 求关节角 |
| seed | IK 求解使用的初始关节角 |
| twist | 绕试管长轴转腕 |
| 齐次坐标 | 用四维向量统一表达三维旋转和平移 |
| 圆周平均 | 正确平均跨越 ±180° 的方向角 |

学习时不必一次记住 Hamilton 积的四行展开公式。先掌握数据流、单位、坐标系、PCA、`atan2`、FK/IK 的输入输出，再回头研究四元数，会容易很多。


在项目根目录运行。
先预览检测结果，不连接机械臂：
python scripts_archive/14_grasp_yellow_tube_cap_up.py


只连接机械臂检查IK，不运动：
python scripts_archive/14_grasp_yellow_tube_cap_up.py \
  --ip 10.42.0.101 \
  --check-ik


确认IK通过后，真实执行：
python scripts_archive/14_grasp_yellow_tube_cap_up.py \
  --ip 10.42.0.101 \
  --execute
  
建议严格按照“预览 → --check-ik → --execute”的顺序运行。
