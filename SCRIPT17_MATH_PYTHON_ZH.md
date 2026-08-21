# SCRIPT17 数学理论和 Python 语法详细说明

对应脚本：`scripts/17_demo_two_tube_branch_yaw.py`

这份文档和 `SCRIPT17_LEARNING_ZH.md` 的定位不同：

- `SCRIPT17_LEARNING_ZH.md`：讲脚本17的整体流程，适合先看。
- 本文档：专门讲脚本17里面用到的数学知识、机器人理论、图像几何、以及 Python 语法。

建议阅读顺序：

```text
先读第 1～18 节：数学和机器人理论
再读第 19～45 节：Python 语法
最后读第 46 节：把整条链路串起来
```

---

## 1. 脚本17的理论主线

脚本17的核心不是“纯粹识别图片”，也不是“单纯发送机械臂关节”。它做的是一条完整链路：

```text
图像里的试管
  -> 分割掩膜
  -> B 点、C 点、中心点
  -> 像素坐标
  -> 机械臂基座坐标
  -> 抓取 yaw
  -> POS/NEG 固定分支
  -> 绑定 yaw 候选
  -> IK 检查抓取前半段
  -> 固定关节完成放槽
```

从数学上看，这里面主要有六类知识：

```text
1. 图像坐标系
2. 掩膜和连通区域
3. PCA 主轴
4. 相机模型和射线-平面交点
5. 角度、yaw、wrap_pi、圆周平均
6. 四元数、IK、FK、关节空间
```

---

## 2. 图像坐标系：为什么图像 y 轴是向下

OpenCV 图像的坐标原点在左上角：

```text
(0, 0) -----------------> x / u
  |
  |
  |
  v
y / v
```

所以一个像素点通常写成：

```python
[x, y]
```

其中：

```text
x：横向，向右增大
y：纵向，向下增大
```

这和数学课上的平面坐标不同。数学课常见的是：

```text
y 向上
```

但图像里是：

```text
y 向下
```

这就是为什么脚本里反复提醒：

```text
图像角度不能直接当机械臂 yaw。
```

---

## 3. B 点和 C 点为什么要区分方向

脚本里定义：

```text
B：试管无盖端
C：试管盖子端
B→C：从无盖端指向盖子端的方向
```

如果只知道“试管是一条线”，机械臂只能知道它的轴线方向，但不知道盖子在哪一端。

举例：

```text
B -------- C
```

和：

```text
C -------- B
```

这两条线的几何轴线一样，但“盖子端方向”相反。脚本17后续要让盖子朝上、再放到凹槽，所以必须知道 B 和 C。

---

## 4. YOLO 分割结果是什么

YOLO 实例分割输出的不只是检测框，还包括每个目标的掩膜。

可以简单理解为：

```text
检测框：目标大概在哪里
掩膜：目标每个像素具体覆盖哪里
```

脚本17使用的类别顺序是：

```python
MODEL_CLASSES = ("p-body", "p-cap", "y-body", "y-cap")
```

对应 ID：

```text
0 -> p-body  紫色管身
1 -> p-cap   紫色管盖
2 -> y-body  黄色管身
3 -> y-cap   黄色管盖
```

所以：

```python
yellow_geometry(result, image_shape)
```

实际找的是：

```text
y-body + y-cap
```

而：

```python
purple_geometry(result, image_shape)
```

实际找的是：

```text
p-body + p-cap
```

---

## 5. 掩膜 mask 是什么

掩膜可以理解为一张只有“是/不是”的图。

例如：

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

其中：

```text
1：属于这个物体
0：不属于这个物体
```

脚本中：

```python
mask = cv2.resize(masks[index], (width, height)) > 0.5
```

意思是：

```text
先把 YOLO 掩膜缩放回原图大小；
再用 0.5 做阈值；
大于 0.5 的像素认为属于物体。
```

---

## 6. 为什么要找最大连通区域

YOLO 掩膜可能有小噪点，比如：

```text
真正试管区域：一大块
误检噪点：几个小点
```

脚本使用：

```python
cv2.connectedComponentsWithStats(...)
```

把所有白色区域分成不同块。

连通区域就是：

```text
上下左右或斜方向连在一起的一块像素。
```

然后脚本只保留面积最大的那块：

```python
index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
return labels == index
```

这样做是为了：

```text
不要让零碎噪点影响试管中心和长轴方向。
```

---

## 7. np.nonzero 为什么返回 y、x

脚本里：

```python
ys, xs = np.nonzero(largest_component(mask))
```

对二维图像来说，NumPy 数组下标顺序是：

```python
image[row, column]
```

也就是：

```text
row    -> y
column -> x
```

所以 `np.nonzero()` 返回的是：

```text
第一个数组：哪些行不是 0，也就是 y
第二个数组：哪些列不是 0，也就是 x
```

之后脚本用：

```python
np.column_stack((xs, ys))
```

把它重新整理成更直观的：

```text
[
  [x1, y1],
  [x2, y2],
  ...
]
```

---

## 8. PCA 主轴：为什么能找到试管方向

试管管身是一条细长形状。把管身所有像素点看成点云，它在长轴方向上分布最宽。

PCA 做的事情就是：

```text
找出点云分布最宽的方向。
```

脚本核心代码：

```python
center = np.mean(body, axis=0)
covariance = np.cov((body - center).T)
values, vectors = np.linalg.eigh(covariance)
axis = vectors[:, int(np.argmax(values))]
```

逐步解释：

```python
center = np.mean(body, axis=0)
```

求所有管身像素点的平均位置，也就是中心点。

```python
body - center
```

把点云平移到中心附近，方便统计分布方向。

```python
np.cov((body - center).T)
```

计算协方差矩阵。协方差矩阵描述：

```text
点云在 x 方向分散多少；
点云在 y 方向分散多少；
x 和 y 是否一起变化。
```

```python
values, vectors = np.linalg.eigh(covariance)
```

求特征值和特征向量。

可以把它理解为：

```text
特征向量：候选方向
特征值：这个方向上点云展开得多宽
```

最大特征值对应的方向就是试管长轴。

---

## 9. 为什么用 1% 和 99% 分位数找端点

脚本不是直接用投影最小值和最大值，而是：

```python
end_1 = center + np.percentile(projections, 1) * axis
end_2 = center + np.percentile(projections, 99) * axis
```

原因是最边缘的像素可能有噪声。

如果使用绝对最小/最大：

```text
一个孤立噪点就可能被当成端点。
```

用 1% 和 99%：

```text
忽略极少数最边缘点；
端点更稳定。
```

这是一个很实用的视觉工程技巧。

---

## 10. atan2 是什么

脚本里多次用：

```python
math.atan2(y, x)
```

它的作用是：

```text
根据一个向量 (x, y)，计算这个向量相对 x 轴正方向的角度。
```

为什么不用普通的：

```python
math.atan(y / x)
```

因为 `atan(y/x)` 分不清象限，而且 x=0 时会出问题。

`atan2(y, x)` 可以正确处理四个象限：

```text
(+x, +y)
(-x, +y)
(-x, -y)
(+x, -y)
```

在脚本里：

```python
image_angle_deg = math.degrees(math.atan2(delta_v, delta_u))
```

表示图像里的 B→C 角度。

```python
tube_yaw = math.atan2(direction[1], direction[0])
```

表示机械臂基座 XY 平面里的 B→C yaw。

---

## 11. 图像角度 image_angle_deg

脚本中：

```python
delta_u = float(cap[0] - bottom[0])
delta_v = float(cap[1] - bottom[1])
image_angle_deg = math.degrees(math.atan2(delta_v, delta_u))
```

这里：

```text
bottom 是 B 点
cap 是 C 点
delta_u 是 C 相对 B 的图像横向变化
delta_v 是 C 相对 B 的图像纵向变化
```

由于图像 y 轴向下，所以这个角度只适合描述“图像里看起来的方向”。

脚本17用它判断：

```text
image_angle_deg > 0 -> POS
image_angle_deg < 0 -> NEG
```

但它不直接控制机械臂 yaw。

---

## 12. 机械臂 yaw tube_yaw

脚本先把 B、C 两点从像素坐标转换到机械臂基座坐标：

```python
bottom_arm = pixel_to_arm(bottom, calibration)
cap_arm = pixel_to_arm(cap, calibration)
```

然后：

```python
direction = cap_arm - bottom_arm
tube_yaw = math.atan2(direction[1], direction[0])
```

这里的 `direction` 是机械臂基座坐标系中的 B→C 向量。

所以 `tube_yaw` 才是真正用于夹爪对齐试管方向的 yaw。

一句话：

```text
image_angle_deg 负责选择分支；
tube_yaw 负责计算抓取姿态。
```

---

## 13. 相机内参矩阵 K

脚本读取：

```python
intrinsic = load_matrix("intrinsic.txt", (3, 3))
```

相机内参矩阵通常长这样：

```text
K = [ fx   0  cx
       0  fy  cy
       0   0   1 ]
```

含义：

```text
fx, fy：焦距，单位像素
cx, cy：主点，也就是相机光轴落在图像上的位置
```

相机模型大致是：

```text
[u, v, 1] = K * [X/Z, Y/Z, 1]
```

反过来，从一个像素点可以得到一条相机射线。

---

## 14. 像素点为什么只能得到一条射线

单个 RGB 像素没有深度。

同一个像素点可能对应空间里一整条线上的点：

```text
相机光心 ---- 近处点 ---- 中间点 ---- 远处点
```

它们投影到图像上都是同一个像素。

所以脚本不能只靠一个像素直接知道三维位置，必须引入额外条件：

```text
试管在工作台平面上，工作台 Z 高度已知。
```

这个已知高度就是：

```python
TABLE_Z_ARM_MM = 68.12483333
```

---

## 15. 像素到机械臂坐标：射线和平面求交

对应函数：

```python
pixel_to_arm(pixel, calibration)
```

核心步骤：

```python
ray_camera = np.linalg.solve(calibration["intrinsic"], [u, v, 1.0])
arm_from_camera = calibration["arm_from_camera"]
ray_origin_arm = arm_from_camera[:3, 3]
ray_direction_arm = arm_from_camera[:3, :3] @ ray_camera
scale = (TABLE_Z_ARM_MM - ray_origin_arm[2]) / ray_direction_arm[2]
arm_mm = ray_origin_arm + scale * ray_direction_arm
```

理论解释：

```text
origin：相机原点在机械臂坐标系中的位置
direction：像素射线在机械臂坐标系中的方向
scale：沿射线走多远会碰到工作台平面
```

射线公式：

```text
p = origin + scale * direction
```

要求交点在工作台平面上：

```text
p_z = TABLE_Z_ARM_MM
```

所以：

```text
scale = (TABLE_Z_ARM_MM - origin_z) / direction_z
```

最后除以 1000：

```python
return arm_mm[:3] / 1000.0
```

因为标定矩阵用毫米，而 CArm Pose 用米。

---

## 16. 手眼矩阵 T_arm2cam 的意义

脚本读取：

```python
arm_from_camera = load_matrix("T_arm2cam.txt", (4, 4))
```

它是 4×4 齐次变换矩阵：

```text
T = [ R  t
      0  1 ]
```

其中：

```text
R：3×3 旋转矩阵
t：3×1 平移向量
```

脚本使用方式是：

```python
ray_origin_arm = arm_from_camera[:3, 3]
ray_direction_arm = arm_from_camera[:3, :3] @ ray_camera
```

也就是说它把相机坐标系中的射线方向转换到机械臂基座坐标系。

---

## 17. wrap_pi：角度归一化

对应函数：

```python
def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
```

它把角度限制到：

```text
[-pi, pi)
```

也就是：

```text
-180° 到 +180°
```

为什么需要？

角度有周期性：

```text
181° 和 -179° 几乎是同一个方向
```

如果不归一化，代码可能误以为它们差了 360°。

---

## 18. 圆周平均：为什么角度不能直接平均

稳定检测里有这段：

```python
angles = np.arctan2((caps - bottoms)[:, 1], (caps - bottoms)[:, 0])
mean = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
```

角度不能直接平均。

例如：

```text
+179° 和 -179°
```

普通平均：

```text
(179 + -179) / 2 = 0°
```

但真实方向应该接近：

```text
180°
```

所以脚本把每个角度转换成单位圆上的点：

```text
cos(angle), sin(angle)
```

先平均这些点，再用 `atan2` 转回角度。

---

## 19. 四元数是什么

机械臂姿态用四元数表示：

```text
[qx, qy, qz, qw]
```

它表示三维旋转。

脚本里 CArm Pose 是：

```python
[x, y, z, qx, qy, qz, qw]
```

前 3 个是位置，后 4 个是姿态。

四元数比欧拉角稳定，能避免某些角度表示上的奇异问题。

---

## 20. 单位四元数 normalize_quat()

四元数表示旋转时，长度必须为 1：

```text
qx² + qy² + qz² + qw² = 1
```

脚本里：

```python
normalized = q / np.linalg.norm(q)
```

就是把四元数除以自己的长度。

如果长度接近 0：

```python
if np.linalg.norm(q) < 1e-12:
    raise ValueError("四元数长度不能为零")
```

因为零长度四元数没有旋转意义。

---

## 21. 四元数乘法 quat_multiply()

两个旋转不能直接相加。

如果想把两个旋转合成一个旋转，要做四元数乘法：

```python
quat_multiply(left, right)
```

在脚本17里最重要的用途是：

```text
先保持夹爪向下；
再绕基座 Z 轴转到试管方向。
```

注意顺序很重要：

```text
quat_multiply(A, B)
```

表示把两个旋转按指定顺序复合。旋转乘法通常不满足交换律：

```text
A * B 不一定等于 B * A
```

---

## 22. down_pose_quat() 如何生成抓取姿态

对应函数：

```python
def down_pose_quat(yaw: float, down_quaternion=DOWN_QUATERNION):
```

脚本先构造绕 Z 轴旋转 yaw 的四元数：

```python
half = wrap_pi(yaw) / 2.0
yaw_quaternion = (0.0, 0.0, math.sin(half), math.cos(half))
```

为什么用半角？

四元数表示“绕单位轴旋转 angle”时公式是：

```text
[axis_x*sin(angle/2),
 axis_y*sin(angle/2),
 axis_z*sin(angle/2),
 cos(angle/2)]
```

绕 Z 轴旋转时：

```text
axis = [0, 0, 1]
```

所以：

```text
[0, 0, sin(yaw/2), cos(yaw/2)]
```

最后：

```python
return quat_multiply(yaw_quaternion, down_quaternion)
```

得到最终抓取姿态。

---

## 23. 两个 yaw 候选为什么存在

代码：

```python
yaw_1 = wrap_pi(tube_yaw + math.radians(yaw_offset_deg))
"candidate_yaws": (yaw_1, wrap_pi(yaw_1 + math.pi))
```

夹爪近似对称，所以沿试管方向夹和反方向夹都可能夹住同一根试管：

```text
yaw_1
yaw_1 + 180°
```

如果让程序自动尝试这两个候选，它可能会选到和后面固定凹槽关节表不配套的腕部姿态。

脚本17不自动尝试，而是通过 POS/NEG 绑定：

```python
YELLOW_BRANCH_YAW_INDEX = {
    "POS": 1,
    "NEG": 2,
}
```

这样做是为了让抓取姿态和后面的固定关节表保持一致。

---

## 24. POS / NEG 分支的理论

脚本17用图像角度决定分支：

```python
if image_angle_deg > 0.0:
    return "POS"
return "NEG"
```

直观理解：

```text
图像里 B→C 朝一个方向倾斜 -> POS
图像里 B→C 朝另一个方向倾斜 -> NEG
```

POS 和 NEG 各有一套固定关节表：

```text
YELLOW_TUBE_POS_JOINTS
YELLOW_TUBE_NEG_JOINTS
PURPLE_TUBE_POS_JOINTS
PURPLE_TUBE_NEG_JOINTS
```

每套固定关节表都是现场标定出来的，不是脚本自动规划出来的。

---

## 25. 0° / ±180° 边界为什么危险

图像角度接近 0° 时：

```text
+0.001°
-0.001°
```

在图像里几乎一样，但代码会分到不同分支。

图像角度接近 ±180° 时也类似：

```text
+179.999°
-179.999°
```

它们几乎是同一个方向，但正负号可能因为微小抖动改变。

所以脚本17有：

```python
near_zero = abs(image_angle_deg) < BRANCH_BOUNDARY_DEG
near_180 = abs(abs(image_angle_deg) - 180.0) < BRANCH_BOUNDARY_DEG
```

当前脚本值是：

```python
BRANCH_BOUNDARY_DEG = 0.002
```

这表示保护范围非常窄，只有极其接近分界线才停止。

---

## 26. IK 是什么

IK 是 Inverse Kinematics，逆运动学。

问题是：

```text
我希望夹爪 TCP 到某个位置和姿态，
那六个关节应该是多少？
```

脚本调用：

```python
response = self.arm.inverse_kine(list(pose), list(seed), tool=TOOL)
```

其中：

```text
pose：目标 TCP 位姿
seed：求解初值，也就是从哪个关节姿态附近找解
```

同一个 TCP 位姿可能有多组关节解，所以 seed 很重要。

---

## 27. FK 是什么

FK 是 Forward Kinematics，正运动学。

问题是：

```text
我知道六个关节角，
夹爪 TCP 实际在哪里、朝哪？
```

脚本调用：

```python
pose = self.arm.forward_kine(list(joints), tool=TOOL)
```

脚本用 FK 做两件事：

```text
1. 从固定关节位读出 TCP 位置，用于打印；
2. 验证 IK 得到的关节解是否真的能回到目标 Pose。
```

---

## 28. IK/FK 一致性检查

脚本不是完全相信 SDK 的 IK 返回值，而是：

```python
actual = np.asarray(self.fk(joints))
target = np.asarray(pose, dtype=float)
```

然后检查位置误差：

```python
np.linalg.norm(actual[:3] - target[:3]) > 0.002
```

意思是：

```text
实际位置和目标位置相差超过 2 mm 就认为失败。
```

姿态误差用四元数点积：

```python
quaternion_dot = abs(float(np.dot(actual[3:], target[3:])))
angle_error = 2.0 * math.acos(float(np.clip(quaternion_dot, 0.0, 1.0)))
```

两个单位四元数越接近，点积越接近 1。

如果姿态误差超过 1°，脚本认为失败。

---

## 29. 固定关节表的意义

脚本17后半段不是动态 IK 到凹槽，而是直接使用固定关节。

例如：

```python
YELLOW_TUBE_POS_JOINTS = np.array(
    [
        [...],  # 盖子朝上
        [...],  # 凹槽上方
        [...],  # 凹槽释放
        [...],  # 松爪后撤回
    ],
    dtype=np.float64,
)
```

这表示：

```text
这些关节角已经在现场手动调过；
脚本只负责按顺序执行。
```

所以脚本17的稳定性很依赖这些固定关节表和实际现场位置一致。

---

## 30. 为什么抓取前半段用 IK，放槽后半段用固定关节

抓取位置取决于试管在桌面上的随机位置，所以必须用视觉 + IK。

但凹槽位置是固定的，现场已经标定过，所以后半段可以直接用固定关节。

这就是脚本17的混合策略：

```text
随机目标：视觉 + IK
固定凹槽：固定关节
```

它比全程动态规划更简单，也更符合你现在的项目风格。

---

# Python 语法部分

下面开始讲脚本17中出现的 Python 语法。每个语法都配一个小例子。

---

## 31. Shebang 是什么

脚本第一行：

```python
#!/usr/bin/env python3
```

这叫 shebang。

它表示：

```text
如果这个文件被当作可执行文件运行，
系统应该用 python3 来解释它。
```

你现在通常用：

```bash
python scripts/17_demo_two_tube_branch_yaw.py
```

所以这一行不是必须，但保留是好习惯。

---

## 32. 文件顶部三引号字符串

脚本开头：

```python
"""脚本17：先抓黄色试管到凹槽，再抓紫色试管到凹槽。"""
```

这是模块 docstring。

它的作用是：

```text
说明这个脚本做什么；
说明运行模式；
让读代码的人先有整体印象。
```

---

## 33. from __future__ import annotations

代码：

```python
from __future__ import annotations
```

它让类型注解延迟解析。

你可以简单理解成：

```text
让一些类型提示写法在 Python 3.10 里更顺滑。
```

它不影响机械臂逻辑，只影响 Python 如何处理类型提示。

---

## 34. import 导入模块

脚本导入了标准库：

```python
import argparse
import math
import time
from collections import deque
from pathlib import Path
```

也导入了第三方库：

```python
import cv2
import numpy as np
import yaml
```

常见用法：

```python
math.sin(angle)
time.sleep(0.5)
np.array([...])
cv2.imshow(...)
yaml.safe_load(file)
```

---

## 35. as np 是别名

代码：

```python
import numpy as np
```

意思是：

```text
以后用 np 代表 numpy。
```

所以：

```python
np.array([1, 2, 3])
```

等价于：

```python
numpy.array([1, 2, 3])
```

这是 NumPy 的常见写法。

---

## 36. 常量命名

脚本里很多大写名字：

```python
GRASP_Z = 0.165
BRANCH_BOUNDARY_DEG = 0.002
```

大写表示：

```text
这是现场参数或固定配置；
运行过程中通常不应该随便改。
```

后缀也有含义：

```text
_M   -> 米
_MM  -> 毫米
_DEG -> 角度
```

---

## 37. list 列表

列表用方括号：

```python
pose = [x, y, z, qx, qy, qz, qw]
```

特点：

```text
有顺序；
可以修改；
可以放多个值。
```

CArm Pose 就是一个 7 个元素的列表。

---

## 38. tuple 元组

元组用圆括号：

```python
DOWN_QUATERNION = (0.999575504, 0.008135427, 0.027844000, 0.002709061)
```

特点：

```text
有顺序；
通常不修改；
适合保存固定的一组值。
```

脚本返回两个 yaw 候选也用了元组：

```python
"candidate_yaws": (yaw_1, wrap_pi(yaw_1 + math.pi))
```

---

## 39. dict 字典

字典用大括号：

```python
YELLOW_BRANCH_YAW_INDEX = {
    "POS": 1,
    "NEG": 2,
}
```

它保存“键 -> 值”的关系。

读取：

```python
yaw_index = branch_yaw_index[groove_branch]
```

如果：

```python
groove_branch = "POS"
```

那么：

```python
yaw_index = 1
```

---

## 40. np.array 二维数组

固定关节表是二维数组：

```python
YELLOW_TUBE_POS_JOINTS = np.array(
    [
        [j1, j2, j3, j4, j5, j6],
        [j1, j2, j3, j4, j5, j6],
        [j1, j2, j3, j4, j5, j6],
        [j1, j2, j3, j4, j5, j6],
    ],
    dtype=np.float64,
)
```

它的形状是：

```text
4 行 × 6 列
```

取第 2 行：

```python
selected_joints[GROOVE_RELEASE_ROW]
```

如果：

```python
GROOVE_RELEASE_ROW = 2
```

就是取“凹槽释放”那组 6 个关节。

---

## 41. dtype=np.float64

代码：

```python
dtype=np.float64
```

意思是：

```text
数组里的数用 64 位浮点数保存。
```

机械臂关节角、坐标、四元数都是小数，所以用浮点数。

---

## 42. 函数 def

函数定义：

```python
def wrap_pi(angle: float) -> float:
    ...
```

含义：

```text
函数名：wrap_pi
输入参数：angle
返回类型提示：float
```

调用：

```python
new_angle = wrap_pi(angle)
```

函数的好处是：

```text
把一段有明确意义的逻辑封装起来；
以后重复使用；
主流程更容易读。
```

---

## 43. 默认参数

代码：

```python
def down_pose_quat(yaw: float, down_quaternion=DOWN_QUATERNION):
```

`down_quaternion=DOWN_QUATERNION` 是默认参数。

调用时可以只写：

```python
down_pose_quat(yaw)
```

Python 会自动使用默认的 `DOWN_QUATERNION`。

如果某天你想临时换一个向下姿态，也可以：

```python
down_pose_quat(yaw, another_down_quaternion)
```

---

## 44. 类型提示

例子：

```python
def normalize_quat(quaternion: Sequence[float]) -> tuple[float, ...]:
```

含义：

```text
quaternion 应该是一个浮点数序列；
函数返回一个浮点数元组。
```

`tuple[float, ...]` 里的 `...` 表示：

```text
元组里有若干个 float。
```

类型提示不会自动保证运行安全，但能帮助你和编辑器理解代码。

---

## 45. if 判断

代码：

```python
if image_angle_deg > 0.0:
    return "POS"

return "NEG"
```

意思是：

```text
如果图像角度大于 0，返回 POS；
否则返回 NEG。
```

另一个例子：

```python
if near_zero or near_180:
    raise RuntimeError(...)
```

表示：

```text
只要 near_zero 或 near_180 有一个是真的，就报错停止。
```

---

## 46. for 循环

脚本里常见：

```python
for center_px, bottom_px, cap_px in history:
    centers_list.append(center_px)
```

意思是：

```text
逐个取出 history 里的元素；
每个元素拆成 center_px、bottom_px、cap_px；
把 center_px 放进 centers_list。
```

这叫“序列解包”。

---

## 47. while True 无限循环

在相机检测中会看到：

```python
while True:
    frame = camera.read()
    ...
```

意思是：

```text
一直读取相机画面；
直到用户按 C 锁定，或按 Q/Esc 退出。
```

这种写法常用于实时相机程序。

---

## 48. deque 固定长度队列

代码：

```python
history = deque(maxlen=STABLE_FRAMES)
```

`deque` 是双端队列。

`maxlen=5` 的意思是：

```text
最多保存 5 个元素；
新元素进来时，如果满了，自动丢掉最老的。
```

脚本用它保存最近 5 帧检测结果。

---

## 49. None 的含义

脚本中很多函数可能返回 `None`：

```python
if result.boxes is None or result.masks is None:
    return None
```

`None` 表示：

```text
没有结果；
当前帧不可用；
暂时不能继续。
```

比如某一帧没有检测到黄色试管，`yellow_geometry()` 就会返回 `None`。

---

## 50. try / except

代码：

```python
try:
    approach_joints = robot.ik(...)
except RuntimeError as exc:
    raise RuntimeError(...) from exc
```

意思是：

```text
尝试执行 IK；
如果 IK 报 RuntimeError；
就包装成更具体的错误信息，再抛出。
```

`from exc` 的作用是保留原始错误原因，方便调试。

---

## 51. try / finally

主流程中：

```python
robot = SimpleCArm(args.ip)
try:
    ...
finally:
    robot.close()
```

意思是：

```text
只要连接了机械臂；
无论中途成功、失败、用户取消；
最后都执行 robot.close() 断开连接。
```

这是硬件脚本很重要的资源释放习惯。

---

## 52. class 和 self

脚本里：

```python
class RealSenseColor:
```

和：

```python
class SimpleCArm:
```

都是类。

类适合封装“有状态的设备”：

```text
RealSenseColor 保存相机 pipeline；
SimpleCArm 保存机械臂连接对象。
```

`self` 表示当前对象本身。

例如：

```python
self.pipeline = rs.pipeline()
```

意思是：

```text
把 pipeline 保存到当前 RealSenseColor 对象里；
后面的 read() 和 close() 都可以继续用。
```

---

## 53. 延迟导入

脚本里有：

```python
import pyrealsense2 as rs
```

但它在 `RealSenseColor.__init__()` 里面。

还有：

```python
from carm import Carm
```

但它在 `SimpleCArm.__init__()` 里面。

这种写法叫延迟导入。

好处：

```text
只有真正打开相机时才需要 pyrealsense2；
只有真正连接机械臂时才需要 carm；
运行 --help 时不必加载重硬件库。
```

---

## 54. pathlib.Path

代码：

```python
ROOT = Path(__file__).resolve().parents[1]
```

含义：

```text
__file__：当前脚本路径
resolve()：转成绝对路径
parents[1]：向上两级，得到工程根目录
```

`Path` 比字符串拼路径更清晰：

```python
ROOT / "runs"
```

表示工程根目录下的 `runs` 文件夹。

---

## 55. with open 自动关闭文件

脚本读取 YAML：

```python
with (CALIBRATION_DIR / "camera.yaml").open(encoding="utf-8") as file:
    camera = yaml.safe_load(file)
```

`with` 的作用是：

```text
打开文件；
读取完成后自动关闭；
即使中途发生异常，也会尽量关闭。
```

这是读写文件的推荐写法。

---

## 56. f-string

脚本里很多：

```python
print(f"起始零位最大关节误差：{zero_error:.3f}°")
```

`f"..."` 叫 f-string。

花括号里可以放变量：

```python
{zero_error}
```

`:.3f` 表示保留 3 位小数。

例子：

```python
x = 0.123456
print(f"{x:.3f}")
```

输出：

```text
0.123
```

---

## 57. 切片

脚本里：

```python
arm_from_camera[:3, 3]
arm_from_camera[:3, :3]
actual[:3]
actual[3:]
```

含义：

```text
[:3] 取前 3 个
[3:] 取从第 3 个开始到最后
```

对于 4×4 矩阵：

```python
arm_from_camera[:3, :3]
```

取左上角 3×3 旋转矩阵。

```python
arm_from_camera[:3, 3]
```

取右上角 3×1 平移向量。

---

## 58. @ 矩阵乘法

代码：

```python
ray_direction_arm = arm_from_camera[:3, :3] @ ray_camera
```

`@` 是矩阵乘法。

这里表示：

```text
把相机坐标系里的射线方向，
用旋转矩阵转换到机械臂基座坐标系。
```

另一个地方：

```python
projections = (body - center) @ axis
```

表示：

```text
把每个像素点投影到 PCA 主轴上。
```

---

## 59. axis=0 和 axis=1

例子：

```python
center = np.mean(body, axis=0)
```

`body` 是 N×2：

```text
[
  [x1, y1],
  [x2, y2],
  ...
]
```

`axis=0` 表示按列计算：

```text
所有 x 的平均
所有 y 的平均
```

所以得到：

```text
[mean_x, mean_y]
```

另一个例子：

```python
np.linalg.norm(centers - center, axis=1)
```

`axis=1` 表示每一行单独算距离。

---

## 60. bool 判断

脚本中：

```python
if not response_ok(response):
    raise RuntimeError(...)
```

`not` 表示取反。

如果 `response_ok(response)` 是 False，那么 `not False` 就是 True，于是报错。

---

## 61. isinstance()

脚本中：

```python
if isinstance(response, bool):
```

意思是：

```text
判断 response 是不是 bool 类型。
```

类似：

```python
isinstance(response, dict)
```

判断是不是字典。

脚本用它兼容 CArm SDK 不同返回格式。

---

## 62. argparse 命令行参数

脚本里：

```python
parser.add_argument("--check-ik", action="store_true", help="只检查 IK，不运动")
```

`action="store_true"` 的意思是：

```text
命令行写了 --check-ik，args.check_ik 就是 True；
没写，args.check_ik 就是 False。
```

比如：

```bash
python scripts/17_demo_two_tube_branch_yaw.py --check-ik
```

那么：

```python
args.check_ik == True
```

---

## 63. main() 和入口判断

脚本最后：

```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户取消。")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"程序终止：{exc}")
```

意思是：

```text
只有直接运行这个文件时，才执行 main()；
如果这个文件被别的 Python 文件 import，不会自动执行 main()。
```

这是 Python 脚本常见入口写法。

---

## 64. return 字典：为什么 build_one_tube_plan 返回很多东西

`build_one_tube_plan()` 返回：

```python
return {
    "tube_name": tube_name,
    "yaw_index": yaw_index,
    "grasp_yaw": yaw,
    ...
}
```

这是把一根试管执行所需的所有结果打包。

后面的执行函数只需要：

```python
plan["approach_joints"]
plan["grasp_pose"]
plan["cap_up_joints"]
plan["groove_release_joints"]
```

就能按计划运动。

这种写法比返回很多个单独变量更清楚。

---

## 65. 当前脚本17最重要的代码链路

你可以把脚本17的核心链路背下来：

```text
YOLO 掩膜
  -> tube_geometry()
  -> stable_geometry()
  -> make_solution()
  -> choose_groove_branch()
  -> branch_yaw_index[groove_branch]
  -> down_pose_quat(yaw)
  -> robot.ik()
  -> execute_one_tube_plan()
```

其中最容易混淆的是：

```text
image_angle_deg：只决定 POS/NEG
tube_yaw：用于抓取 yaw
candidate_yaws：两个等价夹取方向
branch_yaw_index：把 POS/NEG 和其中一个 yaw 绑定
```

---

## 66. 一句话总结

脚本17的数学本质是：

```text
用视觉几何确定试管在工作台上的位置和方向；
用手眼标定把像素转换为机械臂坐标；
用 yaw 和四元数生成抓取姿态；
用 IK/FK 检查抓取前半段；
用 POS/NEG 固定关节表完成现场标定好的放槽动作。
```

脚本17的 Python 本质是：

```text
用函数把每个步骤拆清楚；
用字典传递计划；
用二维数组保存固定关节；
用类管理相机和机械臂连接；
用 argparse 区分预览、检查和真实执行。
```
