# SCRIPT17 学习笔记：双试管连续抓取 + POS/NEG 绑定 yaw 候选

对应脚本：`scripts/17_demo_two_tube_branch_yaw.py`

这份文档按当前工程的初学者脚本风格整理，重点放在脚本 17 的核心逻辑：

1. 黄色试管和紫色试管一次视觉锁定；
2. 图像角度正负决定 POS / NEG 固定关节分支；
3. POS / NEG 分支和指定 yaw 候选绑定；
4. 接近 0° / ±180° 的分界线时停止，避免分支误选；
5. 连续执行黄色试管，再执行紫色试管。

如果你是第一次读脚本 17，建议顺序是：

```text
main()
  -> detect_and_lock()
  -> yellow_geometry() / purple_geometry()
  -> make_solution()
  -> build_one_tube_plan()
  -> execute_one_tube_plan()
  -> SimpleCArm 和四元数函数
```

---

## 1. 推荐运行命令

只做视觉预览，不连接机械臂：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial 135122076361
```

只读检查 IK，不使能、不运动：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial 135122076361 \
  --ip 10.42.0.101 \
  --check-ik
```

真实执行：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial 135122076361 \
  --ip 10.42.0.101 \
  --execute
```

真实执行前，脚本会要求输入一次完整确认文字：

```text
GRASP_TWO_TUBES yellow=(...) purple=(...)
```

确认后执行顺序是：

```text
黄色试管抓取 -> 黄色试管放入黄色凹槽 -> 黄色回零
紫色试管抓取 -> 紫色试管放入紫色凹槽 -> 紫色回零
```

---

## 2. 为什么脚本 17 要绑定 POS/NEG 和 yaw 候选

同一根试管通常有两个等价抓取 yaw：

```text
yaw 候选 1
yaw 候选 2 = yaw 候选 1 + 180°
```

但当前现场的凹槽动作使用固定关节表。固定关节表和机械臂腕部姿态是配套的，
所以脚本 17 采用更确定的规则：

```text
先根据图像角度选择 POS 或 NEG
再根据 POS / NEG 绑定固定的 yaw 候选
如果绑定的 yaw 不可达，直接报错，不自动换另一个
```

这样做是为了解决一个现场问题：

```text
抓取 yaw 候选不同，机械臂腕部姿态会不同；
后面固定凹槽关节表又依赖这种腕部姿态；
如果前面随便选了另一个 yaw，后面固定关节表可能就不配套。
```

所以脚本 17 的核心原则是：

```text
固定关节表和抓取 yaw 必须成套使用。
```

---

## 3. 脚本 17 的整体流程

脚本 17 可以看成一个线性流程：

```text
读取命令行参数
  -> 读取相机内参和手眼标定矩阵
  -> 打开 RealSense 彩色相机
  -> YOLO 同时识别黄色和紫色试管
  -> 计算两根试管的 B / C / 中心点
  -> 连续 5 帧稳定后按 C 锁定
  -> 连接 CArm
  -> 检查机械臂是否在零位
  -> 生成黄色试管计划
  -> 生成紫色试管计划
  -> 如果是 --check-ik：只打印结果，不运动
  -> 如果是 --execute：确认后先执行黄色，再执行紫色
  -> 断开机械臂连接
```

脚本里的 `main()` 就是在组织这条链路。

---

## 4. 三种运行模式

脚本 17 有三种模式。

### 视觉预览模式

不加 `--check-ik`，不加 `--execute`：

```bash
python scripts/17_demo_two_tube_branch_yaw.py --serial 135122076361
```

这个模式只做：

```text
打开相机
运行 YOLO
画黄色/紫色试管 B/C 点
显示图像角度和机械臂基座 yaw
```

不会连接机械臂。

### IK 检查模式

加 `--check-ik`：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial 135122076361 \
  --ip 10.42.0.101 \
  --check-ik
```

这个模式会连接 CArm，但是不使能、不运动。它主要检查：

```text
当前是否在零位
准备位是否有 IK
抓取上方是否有 IK
抓取下降位是否有 IK
POS/NEG 绑定的 yaw 是否可达
固定凹槽关节 FK 是否能读出 TCP
```

### 真实执行模式

加 `--execute`：

```bash
python scripts/17_demo_two_tube_branch_yaw.py \
  --serial 135122076361 \
  --ip 10.42.0.101 \
  --execute
```

这个模式会真实运动机械臂，所以脚本要求输入确认文字。

---

## 5. 关键坐标系和单位

脚本 17 同时使用图像坐标、相机坐标、机械臂基座坐标和关节坐标。

| 名称 | 代码变量 | 单位 | 含义 |
|---|---|---:|---|
| 图像坐标 | `u, v` / `x, y` | px | 图像左上角为原点，向右是 x 正方向，向下是 y 正方向 |
| 相机射线 | `ray_camera` | 方向 | 一个像素点反投影出来的相机视线 |
| 机械臂基座坐标 | `center_arm` 等 | m | CArm 最终使用的位置坐标 |
| 标定矩阵平移 | `T_arm2cam.txt` | mm | 外部手眼标定文件里的平移单位 |
| CArm Pose | `[x,y,z,qx,qy,qz,qw]` | m + 四元数 | 机械臂末端 TCP 位姿 |
| 固定关节 | `YELLOW_TUBE_POS_JOINTS` 等 | rad | 六个关节角 |
| 图像角度 | `image_angle_deg` | deg | 图像里 B→C 相对水平向右的角度 |
| 机械臂 yaw | `tube_yaw` | rad | 机械臂基座 XY 平面里的 B→C 方向 |

最重要的一点：

```text
image_angle_deg 不能直接当成机械臂 yaw。
```

`image_angle_deg` 只用来判断 POS / NEG 分支；真正让夹爪对齐试管的是 `tube_yaw`。

---

## 6. B 点、C 点和中心点

脚本约定：

```text
B 点：试管无盖端
C 点：靠近盖子的试管端
B→C：从无盖端指向盖子端的方向
center：试管管身中心
```

YOLO 分割会得到管身和盖子的掩膜。脚本先用管身掩膜找长轴，再用盖子掩膜判断哪一端是 C。

直观理解：

```text
管身长轴有两个端点；
哪个端点离盖子中心更近，哪个就是 C；
另一个就是 B。
```

---

## 7. YOLO 掩膜如何变成试管长轴

对应函数：

```python
tube_geometry(result, image_shape, body_class_id, cap_class_id)
```

黄色试管：

```python
yellow_geometry(result, image_shape)
```

紫色试管：

```python
purple_geometry(result, image_shape)
```

脚本先从 YOLO 结果里取类别 ID：

```python
class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
```

然后找到对应类别：

```text
p-body: 0
p-cap : 1
y-body: 2
y-cap : 3
```

接着把掩膜转换成像素点集合：

```python
body = mask_points(...)
cap = mask_points(...)
```

`body` 是一个 N×2 数组：

```text
[
  [x1, y1],
  [x2, y2],
  ...
]
```

每一行表示一个属于管身的像素点。

---

## 8. 为什么要保留最大连通区域

对应函数：

```python
largest_component(mask)
```

YOLO 掩膜有时会带一点噪点。比如真正试管是一大块，旁边还有几个小白点。

脚本使用：

```python
cv2.connectedComponentsWithStats(...)
```

把掩膜分成一块一块，只保留面积最大的那块。

目的：

```text
去掉零碎误检点，避免 PCA 长轴被噪点带偏。
```

---

## 9. PCA 如何找到试管长轴

管身像素点很多，脚本需要找出“这堆点最长的方向”。这就是 PCA 的用途。

核心代码：

```python
center = np.mean(body, axis=0)
covariance = np.cov((body - center).T)
values, vectors = np.linalg.eigh(covariance)
axis = vectors[:, int(np.argmax(values))]
```

含义：

```text
center：所有管身像素的平均位置
body - center：把点云移动到中心附近
covariance：统计这堆点往哪个方向分布更长
values：不同方向上的“展开程度”
vectors：这些方向本身
最大 values 对应的 vectors：试管长轴方向
```

然后用投影找两端：

```python
projections = (body - center) @ axis
end_1 = center + np.percentile(projections, 1) * axis
end_2 = center + np.percentile(projections, 99) * axis
```

这里不用最小值和最大值，而用 1% 和 99% 分位数，是为了减少边缘噪点影响。

---

## 10. 稳定检测为什么看最近 5 帧

对应函数：

```python
stable_geometry(history)
```

脚本不是看到一帧就锁定，而是要求最近 5 帧都稳定：

```python
STABLE_FRAMES = 5
MAX_CENTER_JITTER_PX = 3.0
MAX_ANGLE_JITTER_DEG = 3.0
```

稳定判断包括两件事：

```text
中心点不要抖太多；
B→C 角度不要抖太多。
```

这样可以避免某一帧 YOLO 掩膜抖动导致机械臂抓偏。

角度平均时不能直接普通平均。例如：

```text
+179° 和 -179° 实际方向几乎一样；
普通平均会得到 0°，这是错的。
```

所以脚本使用圆周平均：

```python
mean = atan2(mean(sin(angle)), mean(cos(angle)))
```

---

## 11. 像素点如何变成机械臂坐标

对应函数：

```python
pixel_to_arm(pixel, calibration)
```

一个 RGB 像素点本身没有深度，脚本用“工作台高度固定”这个条件求三维位置。

计算顺序：

```text
像素点 [u,v]
  -> 相机坐标系里的一条射线
  -> 用 T_arm2cam.txt 转到机械臂基座坐标
  -> 射线和固定工作台平面 z=TABLE_Z_ARM_MM 求交点
  -> 得到机械臂基座坐标中的点
```

核心公式：

```python
ray_camera = np.linalg.solve(intrinsic, [u, v, 1.0])
ray_origin_arm = arm_from_camera[:3, 3]
ray_direction_arm = arm_from_camera[:3, :3] @ ray_camera
scale = (TABLE_Z_ARM_MM - ray_origin_arm[2]) / ray_direction_arm[2]
arm_mm = ray_origin_arm + scale * ray_direction_arm
```

最后：

```python
return arm_mm[:3] / 1000.0
```

因为标定矩阵平移用毫米，CArm Pose 用米，所以要除以 1000。

---

## 12. 图像角度和机械臂 yaw 的区别

脚本 17 同时计算两个角度：

```python
image_angle_deg = math.degrees(math.atan2(delta_v, delta_u))
tube_yaw = math.atan2(direction[1], direction[0])
```

它们不是一回事。

### image_angle_deg

来自图像坐标：

```text
图像 x 轴向右
图像 y 轴向下
```

它主要用于：

```text
判断 POS / NEG 固定关节分支。
```

### tube_yaw

来自机械臂基座坐标：

```text
机械臂 X/Y 轴由机器人基座定义
```

它主要用于：

```text
让夹爪在工作台平面内对齐试管方向。
```

所以脚本17的关键思想是：

```text
image_angle_deg 决定走哪套固定关节；
tube_yaw 决定抓取时夹爪朝哪个方向。
```

---

## 13. 为什么有两个抓取 yaw 候选

对应代码：

```python
yaw_1 = wrap_pi(tube_yaw + math.radians(yaw_offset_deg))
"candidate_yaws": (yaw_1, wrap_pi(yaw_1 + math.pi))
```

原因是夹爪通常是近似对称的：

```text
夹爪沿试管方向夹住；
绕竖直 Z 轴转 180° 后，仍然可以夹住同一条试管。
```

所以有两个候选：

```text
yaw 候选 1：tube_yaw
yaw 候选 2：tube_yaw + 180°
```

如果自动尝试两个候选，程序可能会选到和后面固定关节表不配套的腕部姿态。
所以脚本 17 不自动竞争候选，而是由 POS / NEG 分支绑定候选。

---

## 14. 脚本17如何绑定 POS / NEG 和 yaw 候选

脚本17新增了两个字典：

```python
YELLOW_BRANCH_YAW_INDEX = {
    "POS": 1,
    "NEG": 2,
}

PURPLE_BRANCH_YAW_INDEX = {
    "POS": 1,
    "NEG": 2,
}
```

含义：

```text
黄色 POS 分支 -> 使用 yaw 候选 1
黄色 NEG 分支 -> 使用 yaw 候选 2

紫色 POS 分支 -> 使用 yaw 候选 1
紫色 NEG 分支 -> 使用 yaw 候选 2
```

这里的 `1` 和 `2` 是给人看的编号；代码取元组时要减 1：

```python
yaw = solution["candidate_yaws"][yaw_index - 1]
```

因为 Python 下标从 0 开始：

```text
候选 1 -> 下标 0
候选 2 -> 下标 1
```

如果现场验证发现某个分支应该换另一个 yaw，只需要改这个字典。

---

## 15. 0° / ±180° 分界线保护

脚本17新增：

```python
BRANCH_BOUNDARY_DEG = 0.02
```

对应函数：

```python
choose_groove_branch(tube_name, image_angle_deg)
```

核心代码：

```python
near_zero = abs(image_angle_deg) < BRANCH_BOUNDARY_DEG
near_180 = abs(abs(image_angle_deg) - 180.0) < BRANCH_BOUNDARY_DEG
```

如果 `near_zero` 或 `near_180` 为真，脚本直接报错停止。

为什么要这样？

因为 `+0.1°` 和 `-0.1°` 在图像里几乎一样，但代码会把它们分到不同分支：

```text
+0.1° -> POS
-0.1° -> NEG
```

如果这两个分支对应的固定关节完全不同，就可能选错。

当前 `0.02°` 是非常窄的保护范围：

```text
0.01° 会停止
0.03° 不会停止
-0.01° 会停止
-0.03° 不会停止
179.99° 会停止
179.95° 不会停止
```

这几乎等于“只保留极窄保险”。如果后续想更稳，可以把它改大，比如 `0.2` 或 `1.0`。

---

## 16. 固定关节表为什么是二维数组

脚本17有四套固定关节表：

```python
YELLOW_TUBE_POS_JOINTS
YELLOW_TUBE_NEG_JOINTS
PURPLE_TUBE_POS_JOINTS
PURPLE_TUBE_NEG_JOINTS
```

每套都是 4×6：

```text
4 行：四个动作位置
6 列：J1～J6 六个关节
```

例如：

```python
YELLOW_TUBE_POS_JOINTS = np.array(
    [
        [...],  # 第0行：盖子朝上
        [...],  # 第1行：凹槽上方
        [...],  # 第2行：凹槽释放
        [...],  # 第3行：松爪后撤回
    ],
    dtype=np.float64,
)
```

脚本用名字代替行号：

```python
CAP_UP_ROW = 0
GROOVE_ABOVE_ROW = 1
GROOVE_RELEASE_ROW = 2
GROOVE_RETREAT_ROW = 3
```

这样比直接写 `selected_joints[2]` 更好理解。

---

## 17. build_one_tube_plan() 做了什么

对应函数：

```python
build_one_tube_plan(...)
```

它不会真实运动机械臂，只是生成计划。

流程是：

```text
取视觉检测得到的试管中心 x/y
  -> 用 READY_JOINTS 的 FK 得到高空准备位参考姿态
  -> 把准备位 z 改成 rotate_z
  -> IK 检查高空准备位
  -> 用 image_angle_deg 判断 POS / NEG
  -> 根据 POS / NEG 选择固定关节表
  -> 根据 POS / NEG 选择绑定 yaw 候选
  -> 计算抓取上方 pose 和抓取 pose
  -> IK 检查抓取上方和抓取下降位置
  -> 返回 plan 字典
```

这里的重点是：

```python
groove_branch = choose_groove_branch(tube_name, image_angle_deg)
yaw_index = branch_yaw_index[groove_branch]
yaw = solution["candidate_yaws"][yaw_index - 1]
```

这三行就是脚本17的灵魂。

---

## 18. plan 字典里保存了什么

`build_one_tube_plan()` 返回一个字典：

```python
return {
    "tube_name": tube_name,
    "yaw_index": yaw_index,
    "grasp_yaw": yaw,
    "rotate_z": rotate_z,
    "groove_branch": groove_branch,
    "groove_mode": ...,
    "ready_joints": ready_joints,
    "approach_pose": approach_pose,
    "approach_joints": approach_joints,
    "grasp_pose": grasp_pose,
    "cap_up_joints": ...,
    "groove_above_pose": ...,
    "groove_above_joints": ...,
    "groove_release_joints": ...,
    "groove_retreat_joints": ...,
}
```

可以把它理解成：

```text
一根试管从抓取到放槽需要的所有运动目标。
```

后面的 `execute_one_tube_plan()` 不再重新计算，只按这个 plan 发送动作。

---

## 19. execute_one_tube_plan() 的动作顺序

对应函数：

```python
execute_one_tube_plan(robot, plan)
```

当前脚本17执行 11 步：

```text
1. 到试管高空准备位并张开夹爪
2. 从高空准备位直线下降抓取
3. 等待 0.5 秒后闭爪
4. 直线抬回高空准备位
5. 按 POS/NEG 分支移动到固定盖朝上关节1
6. 提示盖子已经朝上
7. 移动到固定凹槽高空位
8. 移动到固定凹槽释放位
9. 打开夹爪
10. 移动到固定撤离关节位
11. 回到六关节零位
```

其中第 2 步和第 4 步是直线运动：

```python
robot.move_line(...)
```

其他大部分是关节运动：

```python
robot.move_joints(...)
```

直线运动适合靠近和离开试管，因为它更容易保持 TCP 在一条竖直线上移动。

固定关节运动适合凹槽阶段，因为这些位置已经由现场手动标定验证过。

---

## 20. 当前脚本里的释放位暂停

你之前讨论过“到凹槽释放位后按 Enter 再打开爪子”。

当前脚本17里有这一行，但它是注释状态：

```python
# input(f"{tube_name} 已到凹槽释放位。确认可以松爪后按 Enter；取消请按 Ctrl+C。")
```

所以当前脚本实际运行时：

```text
到凹槽释放位后，会直接执行打开夹爪。
```

如果你想启用暂停，需要把这一行前面的 `#` 去掉，并确认它放在第 8 步运动完成之后、第 9 步打开夹爪之前。

---

## 21. CArm 包装类 SimpleCArm 做了什么

对应类：

```python
class SimpleCArm:
```

它把第三方 `carm` SDK 包成几个更容易读的方法：

```text
ready()          使能、设置模式、设置速度、设置工具、碰撞配置
fk()             正运动学：关节 -> TCP Pose
ik()             逆运动学：TCP Pose -> 关节
move_joints()    关节运动
move_line()      笛卡尔直线运动
open_gripper()   张开夹爪
close_gripper()  闭合夹爪
close()          断开连接
```

这样主流程不需要直接面对 SDK 的细节。

---

## 22. IK 和 FK 是什么

FK 是正运动学：

```text
已知六个关节角 -> 算出夹爪 TCP 在哪里、朝哪
```

对应：

```python
robot.fk(joints)
```

IK 是逆运动学：

```text
已知希望 TCP 到某个位置和姿态 -> 求六个关节角
```

对应：

```python
robot.ik(pose, seed)
```

脚本17在 `ik()` 里还做了检查：

```text
IK 是否返回 6 个关节
关节是否是有限数字
关节是否太靠近限位
单段关节变化是否太大
用 FK 反算验证位置误差是否小于 2 mm
用四元数点积验证姿态误差是否小于 1°
```

注意：IK/FK 通过不等于“路径一定没有碰撞”。它只说明数学上可达，并做了基本一致性检查。

---

## 23. MAX_JOINT_STEP_DEG 为什么是 130

当前脚本17里：

```python
MAX_JOINT_STEP_DEG = 130.0
```

同时抓取上方 IK 也写了：

```python
approach_joints = robot.ik(approach_pose, ready_joints, 130.0)
```

它的意思是：

```text
从 seed 关节到目标关节，
六个关节里变化最大的那个不能超过 130°。
```

这个限制不是机器人本体的物理限位，而是脚本为了避免单段跳太大设置的安全阈值。

脚本17把这个阈值设得比较宽，是因为绑定固定 yaw 后，有些姿态从准备位到抓取上方的关节变化会更大。

---

## 24. 速度设置怎么看

在 `SimpleCArm.ready()` 中，当前脚本17实际调用：

```python
self.arm.set_speed_level(6, 1000)
```

第一个参数 `6` 是速度等级，范围 0～10。

第二个参数 `1000` 是速度变化的过渡周期数，可以理解为速度从当前等级变到目标等级时的平滑程度。

它不是：

```text
不是 1000 秒
不是速度 1000
不是力 1000
```

更接近：

```text
过渡周期越大，速度变化越平滑；
过渡周期越小，速度变化越直接。
```

真实 demo 调速度时，通常优先改第一个参数，不要随便大幅改第二个参数。

---

## 25. 四元数在脚本里做什么

CArm Pose 使用：

```python
[x, y, z, qx, qy, qz, qw]
```

后四个数是四元数，用来表示夹爪姿态。

脚本里有一个固定的夹爪向下姿态：

```python
DOWN_QUATERNION
```

然后用：

```python
down_pose_quat(yaw)
```

得到：

```text
夹爪保持向下，同时在桌面平面内旋转到目标 yaw。
```

这就是抓试管时的姿态。

---

## 26. wrap_pi() 是什么

对应函数：

```python
wrap_pi(angle)
```

它把角度限制到：

```text
[-pi, pi)
```

也就是：

```text
-180° 到 +180°
```

为什么需要它？

因为角度有周期性：

```text
190° 和 -170° 是同一个方向附近；
如果不 wrap，代码可能误以为它们差了 360°。
```

---

## 27. Python 语法：字典 dict

脚本17大量使用字典。

例如：

```python
YELLOW_BRANCH_YAW_INDEX = {
    "POS": 1,
    "NEG": 2,
}
```

含义是：

```text
键 "POS" 对应值 1
键 "NEG" 对应值 2
```

读取时：

```python
yaw_index = branch_yaw_index[groove_branch]
```

如果 `groove_branch` 是 `"POS"`，就得到 `1`。

另一个重要字典是 `solution`：

```python
solution["center_arm"]
solution["image_angle_deg"]
solution["tube_yaw"]
solution["candidate_yaws"]
```

它保存视觉检测后得到的结果。

---

## 28. Python 语法：二维 np.array

固定关节表是二维 NumPy 数组：

```python
selected_joints[GROOVE_RELEASE_ROW]
```

如果 `GROOVE_RELEASE_ROW = 2`，就是取第 2 行。

因为 Python 从 0 开始计数：

```text
第0行：盖子朝上
第1行：凹槽上方
第2行：凹槽释放
第3行：撤回
```

`.copy()` 的作用是复制一份数组，避免后续如果修改返回值，影响原始固定关节表。

---

## 29. Python 语法：try / except / finally

脚本17里最重要的资源释放逻辑在 `main()`：

```python
robot = SimpleCArm(args.ip)
try:
    ...
finally:
    robot.close()
```

意思是：

```text
只要连接了机械臂，
无论中途成功、失败、用户取消，
最后都尽量断开连接。
```

这对硬件脚本很重要。

`except RuntimeError as exc` 用来捕获某一段失败原因，并把它包装成更容易理解的错误提示。

---

## 30. Python 语法：argparse

脚本17通过：

```python
argparse.ArgumentParser(...)
```

定义命令行参数。

主要参数：

```text
--model            指定 YOLO 权重
--serial           指定 RealSense 相机序列号
--yaw-offset-deg   抓取 yaw 额外偏移，默认 0
--rotate-z         抓取前高空准备高度，当前默认 0.350 m
--ip               CArm 控制器 IP
--check-ik         只检查 IK
--execute          真实执行
```

`action="store_true"` 的意思是：

```text
命令行出现这个参数时，值为 True；
没出现时，值为 False。
```

---

## 31. main() 应该怎么读

`main()` 是脚本17最推荐先读的入口。

它的结构很清楚：

```text
读取参数
读取标定
视觉锁定黄色和紫色
打印两个目标
如果只是预览：结束
连接机械臂
检查零位
生成黄色计划
生成紫色计划
打印计划
如果 execute：确认后执行黄色，再执行紫色
断开连接
```

这也是脚本17比很多复杂工程更容易学的地方：主流程基本是线性的。

---

## 32. 真机安全理解

脚本17有几个安全点：

```text
默认只预览，不连接机械臂；
--check-ik 只读检查，不使能、不运动；
--execute 才真实运动；
真实执行前要求输入确认文字；
执行前检查机械臂是否从零位开始；
连接机械臂后用 finally 保证断开连接；
IK/FK 做位置和姿态一致性检查。
```

但也要清楚：

```text
脚本17没有完整避障规划；
IK 通过不等于不会碰撞；
固定关节表依赖现场标定；
相机、工作台、凹槽、夹爪、试管位置变化后都需要重新验证。
```

---

## 33. 学习时建议你重点看哪些变量

建议你先在脚本中搜索这些名字：

```text
YELLOW_TUBE_POS_JOINTS
YELLOW_TUBE_NEG_JOINTS
PURPLE_TUBE_POS_JOINTS
PURPLE_TUBE_NEG_JOINTS
YELLOW_BRANCH_YAW_INDEX
PURPLE_BRANCH_YAW_INDEX
BRANCH_BOUNDARY_DEG
image_angle_deg
tube_yaw
candidate_yaws
choose_groove_branch
build_one_tube_plan
execute_one_tube_plan
```

如果这些理解了，脚本17的核心就理解了一大半。

---

## 34. 小练习

你可以做几个安全的小练习，先不要真实运动：

1. 运行 `--check-ik`，观察黄色和紫色分别选择 POS 还是 NEG。
2. 把试管稍微转过 0° 附近，观察是否触发边界保护。
3. 临时把 `BRANCH_BOUNDARY_DEG` 改成 `1.0`，观察保护范围变大。
4. 只看打印结果，比较 `image_angle_deg` 和 `B→C yaw` 为什么不同。
5. 看 `YELLOW_BRANCH_YAW_INDEX`，思考如果把 POS 从 1 改成 2 会发生什么。

---

## 35. 一句话总结

脚本17的本质是：

```text
视觉负责找两根试管的位置和方向；
IK 负责抓取前半段可达性；
POS/NEG 固定关节表负责凹槽放置；
分支和 yaw 候选绑定，避免抓取姿态与固定放槽姿态不配套。
```
