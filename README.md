# LibArgus ROS2 同进程管线

## PC Operator UI

`argus_operator_ui` 是运行在 PC 上的 PySide6 客户端。它只订阅
`/camera/image/compressed` 与 `/camera/inference/result`，不会订阅只能在 Orin 同进程使用的
`/camera/image/yuv`，也不会在 PC 启动 YOLO 或 Qwen 模型。UI 在本地完成 JPEG 解码、检测框与
实例 mask 绘制，并通过 `/camera/inference/qwen` Action 与 Orin 上的 Qwen3-VL 交互。

> 当前 Orin 的 `argus_inference/src/inference_node.cpp` 只发布 bbox、类别与置信度，尚未填充
> message 中的 mask 字段。因此当前实机画面只出现 bbox 是正常行为；PC 客户端已支持合法 mask，
> 可用 `--demo` 立即验证透明 mask 叠加。

### PC 依赖与编译

PC 与 Orin 必须使用完全一致的 `argus_interfaces` 源码。Ubuntu 24.04 / ROS 2 Jazzy 上安装：

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy python3-pytest
python3 -m pip install --user PySide6
./scripts/build_pc.sh
source install/setup.bash
```

构建脚本显式使用 `/usr/bin/python3`，避免已激活的 Conda Python（尤其不同 minor 版本）干扰
ROS 2 Jazzy 的 Python 3.12 生成工具。

如果系统启用了 PEP 668，推荐创建虚拟环境并允许访问 ROS 系统包：

```bash
python3 -m venv --system-site-packages .venv-ui
source .venv-ui/bin/activate
pip install PySide6
./scripts/build_pc.sh
source install/setup.bash
```

启动 UI：

```bash
ros2 run argus_operator_ui argus_operator_ui
ros2 launch argus_operator_ui operator_ui.launch.py
```

topic 与 Action 名称可通过 ROS 参数覆盖：

```bash
ros2 run argus_operator_ui argus_operator_ui --ros-args \
  -p image_topic:=/camera/image/compressed \
  -p result_topic:=/camera/inference/result \
  -p qwen_action_name:=/camera/inference/qwen
```

无 Orin 时可用合成画面验证 UI、bbox、mask 和退出流程：

```bash
ros2 run argus_operator_ui argus_operator_ui --demo
ros2 run argus_operator_ui argus_operator_ui --demo --demo-video /path/to/video.mp4
```

### Orin 编译与启动

在 Orin 仓库中执行（脚本跳过仅供 PC 使用的 UI package）：

```bash
./scripts/build_orin.sh
source install/setup.bash
ros2 launch argus_bringup argus_pipeline.launch.py
```

### ROS2 网络配置

两台机器应在同一局域网，使用相同的 ROS domain 和兼容的 RMW 实现：

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
```

先在 PC 验证发现和数据流：

```bash
ros2 topic info /camera/image/compressed -v
ros2 topic hz /camera/image/compressed
ros2 action info /camera/inference/qwen
```

图像与 YOLO 订阅均为 best-effort/volatile/keep-last，网络抖动时会丢旧帧而不累积延迟。
若跨机无法发现，检查两端 `ROS_DOMAIN_ID`、防火墙、组播/VLAN、RMW 实现，以及系统时间；
本客户端的图像/结果匹配使用消息内同源 `header.stamp`，不依赖两台主机墙上时钟完全同步。

### 线程与同步

- Qt 主线程只更新控件与缩放已经生成的 `QImage`。
- ROS worker 独占 node、executor、订阅 callback 和 Action client，短周期检查服务可用性。
- 图像 worker 解码 JPEG、匹配消息、绘制 overlay 并转换 `QImage`；待处理图像槽始终只保留最新帧。
- `FrameSynchronizer` 以 `sec * 1e9 + nanosec` 精确匹配，图像和结果缓存默认各 8 项、1.5 秒过期。
- 新 JPEG 立即显示原图；对应结果稍后抵达时再显示同一帧 overlay，YOLO 跳帧不会阻塞视频。

### UI 常见问题

- Qwen 不可用：确认 Orin 的 Action Server 已启动，并执行 `ros2 action list -t`。
- 只有原图没有框：检查 `/camera/inference/result`；没有检测结果不代表图像流离线。
- 有框没有 mask：当前 Orin 发布逻辑尚未填写 mask，这是已知状态，可先用 demo 模式验证客户端。
- UI 启动时报 Qt platform plugin 错误：确认 PC 有图形会话和 `DISPLAY`/Wayland 环境；SSH 场景使用
  X11 转发或在本地桌面启动。
- 关闭较慢：客户端会请求取消活跃 goal；服务端底层推理可能不能立即中断，但 worker join 有上限，
  不会无限等待。

这是一个 Jetson **LibArgus** 与 ROS2 的同进程图像管线：相机 node 创建并持有 NVMM
YUV buffer pool，ISP 直接写入这些 surface；推理 node 和可视化 node 通过自定义消息共享
同一个 dma-buf lease，不把采集图像复制到 CPU `sensor_msgs/Image`。

## 图像链路

LibArgus 的硬件采集路径为：

```text
MIPI CSI-2 sensor
        │
        ▼
  VI / CSI receiver
        │
        └── ISP（去马赛克、白平衡、降噪、色彩校正等）
                └── NV12 block-linear NVMM buffer pool
                        ├── YOLOv8-seg TensorRT 推理 → /camera/inference/overlay/compressed
                        └── JPEG 编码 → /camera/image/compressed
```

## 代码主线

`argus_camera` 按下面的 LibArgus 对象关系组织：

1. `CameraProvider` / `ICameraProvider`：连接 Argus 服务、枚举摄像头。
2. `CaptureSession` / `ICaptureSession`：针对一个摄像头创建采集会话。
3. `BufferOutputStream`：以 `BUFFER_TYPE_EGL_IMAGE` 将采集结果写入应用预分配的 NV12 block-linear NVMM surface。
4. `IBufferOutputStream`：获取完成的 `Buffer`，并在最后一个消费者释放后归还给 Argus。
5. `ArgusFramePacket`：通过 ROS2 `TypeAdapter` 在进程内传递 dma-buf shared lease；对外只转换并发布元数据。
6. 可视化 node：直接将同一个 dma-buf 交给 JPEG 编码器。

LibArgus 的对象通常通过 `UniqueObj<T>` 管理；同一个对象的不同能力通过 `interface_cast<IXXX>()` 获取。这是学习 LibArgus 时最重要的两个惯用模式。

## Package 结构

```text
src/argus_interfaces       ROS message 定义
src/argus_transport        进程内 ArgusFramePacket 和 dma-buf lease
src/argus_camera           Argus camera component
src/argus_inference        YOLOv8-seg TensorRT inference component
src/argus_visualization    JPEG 输出 component
src/argus_bringup          component container launch
```

三个业务 node 位于不同 package，但由同一个 `component_container_mt` 进程加载，因此
`argus_transport` 中的 `ArgusFramePacket` 能够共享同一个 Argus buffer lease。相机组件
加载后立即开始采集，不依赖某个固定数量的消费者；在消费者连接前产生的帧会被正常丢弃。

## 编译

本示例默认使用 Jetson Multimedia API 的安装路径：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Debug
source install/setup.bash
```

如果 SDK 在其他位置，可以在对应 package 的 CMake 配置中覆盖路径：

```bash
colcon build --symlink-install --cmake-args \
  -DARGUS_ROOT=/path/to/jetson_multimedia_api/argus \
  -DARGUS_INCLUDE_DIR=/path/to/argus/include
```

CMake 会链接 `nvargus_socketclient`。在当前 Jetson 系统上，它位于 `/usr/lib/aarch64-linux-gnu/nvidia/`；CMake 已包含常见的 `nvidia`、`tegra` 和架构库目录搜索路径。

## 运行前检查

- 在 Jetson 上运行，而不是普通 x86 Linux 主机；
- `nvargus-daemon` 已启动；
- 摄像头排线、驱动和设备树配置正常；
- 没有其它程序占用摄像头；
- 如果通过 SSH 运行，JPEG 编码通常仍可工作，但预览/显示类 EGL 示例可能需要图形会话。

## 常见问题

`CameraProvider::create()` 返回空：通常是 `nvargus-daemon` 未运行、驱动未加载，或当前平台不是 Jetson。

`createCaptureSession` 返回 `STATUS_UNAVAILABLE`：摄像头被其它进程占用，先退出 `nvarguscamerasrc`、`nvgstcapture-1.0` 等程序。

`acquireFrame` 超时：检查 sensor mode、摄像头连接和 Argus 日志；也可以先用 `v4l2-ctl --list-devices` 确认设备是否出现。

`当前 sensor mode 不支持 YUV 输出`：检查 sensor mode、摄像头类型和 Argus override 配置。

## ROS2 同进程多节点连续采集链路

连续采集现在由一个进程中的三个 ROS2 node 组成：

```text
argus_camera_node
    └── /camera/image/yuv (argus_interfaces/msg/ArgusYuvFrame)
          ├── yuv_inference_node
          └── yuv_visualization_node
                    └── /camera/image/compressed (sensor_msgs/CompressedImage)
```

三个 node 使用 `MultiThreadedExecutor` 和 ROS2 intra-process communication。camera node
发布 `argus_transport::ArgusFramePacket`，packet 内部通过 `shared_ptr` 持有 dma-buf 与
`Argus::Buffer` 的归还 lease；inference 和 visualization 收到的是同一个 NVMM YUV surface，
不复制 YUV 数据。ROS2 `TypeAdapter` 只在需要 ROS 消息或跨进程传输时复制
时间戳、源帧序号、分辨率和 stride 等元数据。

每个订阅者的 intra-process 队列各自持有一份 packet，但只增加 shared pointer 引用计数。
消息被处理或从 keep-last 队列淘汰后，最后一个引用析构时会自动调用
`IBufferOutputStream::releaseBuffer()`，将该 buffer 归还给 Argus。

native dma-buf lease 是进程内句柄，不能直接拿到另一个独立进程中使用；跨进程订阅者只能收到
`ArgusYuvFrame` 的元数据。如果未来需要跨进程传输 YUV，需要另行实现 dma-buf fd/共享内存
传输协议或采用支持 GPU buffer 的消息类型。

`yuv_inference_node` 只加载已构建好的 YOLOv8-seg TensorRT engine，不包含 ONNX 解析或
engine 构建逻辑。它直接将采集 dma-buf 作为 VIC 的 YUV 输入，转为可复用的 RGBA dmabuf；
随后通过 `NvBufSurfaceMapEglImage` 和 CUDA EGL interop 取得 RGBA
device pointer，由 CUDA kernel 完成双线性 letterbox、RGB 排列、CHW 与 `[0, 1]` 归一化，
并直接写入 TensorRT input buffer。推理输入没有 host-to-device 复制；推理完成后仅将 640×640
RGBA surface 映射到 CPU，用于生成可视化 overlay JPEG。
检测头在节点内回传 CPU 完成类别 NMS 与坐标反变换；mask prototype 始终保留在 GPU，CUDA kernel
使用 NMS 保留目标的系数直接解码、缩放并二值化各自的 ROI 掩码；检测头和用于生成 overlay 的
结果会回到 CPU。
可视化节点直接使用同一采集 dma-buf 进行 JPEG 编码。

可视化发布默认关闭；设置 `enable_overlay:=true` 后，推理结果以可视化专用的
`sensor_msgs/msg/CompressedImage` 发布到 `/camera/inference/overlay/compressed`，图像为 JPEG
格式：在推理用的 640×640 RGBA surface 上叠加实例掩码、边界框、类别和置信度。发布器使用
`best_effort`、`KeepLast(1)` QoS，网络拥塞时丢弃旧帧而不积压延迟；可通过 `overlay_quality`
（默认 90）调整 JPEG 质量。该 topic 可直接在 RViz2 中使用 Image display 订阅，适合跨局域网
实时预览。

默认 engine 是 `/home/royfan/yolov8_trt/yolov8s-seg-640.engine`，profile 固定为
`1x3x640x640`。engine 必须在目标 Jetson 上离线构建；节点启动时只反序列化并加载该文件，
若文件缺失或与目标平台/TensorRT 版本不兼容，节点会立即报错退出。可通过以下参数调整：

```text
input_topic, output_topic, enable_overlay, overlay_quality
engine_path, input_size
timing_log_every_n_frames
confidence_threshold, iou_threshold
```

推理节点只保留一帧正在处理的 RGBA surface 和一帧最新待处理的 YUV handle。模型执行期间到达的
新帧会替换旧的待处理帧，避免积压导致推理结果滞后。

本机离线生成默认 engine 使用的命令如下；`--noTF32` 避开当前 TensorRT 10.16 安装中 FP16
CASK tactic 的 shader 断言：

```bash
trtexec --onnx=/home/royfan/yolov8_trt/yolov8s-seg.onnx \
  --saveEngine=/home/royfan/yolov8_trt/yolov8s-seg-640.engine \
  --noTF32 --memPoolSize=workspace:1024 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --builderOptimizationLevel=0 --avgTiming=1 --skipInference
```

不同 GPU 或 TensorRT 版本生成的 engine 不应互相复用；请在部署目标上运行上面的离线命令，
然后通过 `engine_path` 指向生成文件。

启动整个管线：

```bash
ros2 launch argus_bringup argus_pipeline.launch.py
```

## Qwen 按需推理 Action

`qwen_description_node` 不再按固定帧间隔自动推理。节点持续缓存
`/camera/image/compressed` 的最新一帧，只在收到 `/camera/inference/qwen` action goal 时执行一次
Qwen3-VL 推理。请求中的 `prompt` 会作为本次推理提示词，不再使用节点内置的固定提示词：

```bash
ros2 action send_goal /camera/inference/qwen argus_interfaces/action/QwenInference \
  "{prompt: '请用中文描述画面中的人正在做什么', include_detection_context: false}" \
  --feedback
```

`include_detection_context=true` 时，节点会把最新 YOLO 检测候选类别、置信度和边界框追加到
本次提示词；设为 `false` 时，Qwen 只接收调用方提供的提示词和最新图像。Action result 中的 `result`
包含 `description`、`success`、`error_message`、`inference_ms`、图像 header，以及可用时的检测
帧号和候选类别。为兼容已有消费者，同一结果仍会发布到
`/camera/inference/qwen_description`。尚未收到图像或提示词为空时不会执行推理，响应中的
`success` 为 `false` 并通过 `error_message` 说明原因。执行期间 feedback 的 `stage` 会依次报告
`preparing` 和 `inferencing`；同一时刻只接受一个 goal，已有推理运行时新 goal 会被拒绝。

Action 名称可通过 `action_name` 参数修改；生成长度和采样仍分别由
`max_generate_length`、`temperature` 参数控制。

组件默认按 inference、visualization、camera 顺序加载；相机组件不等待订阅者发现即可开始
采集。相机节点不再通过 ROS node parameter 接收配置，而是默认读取 `argus_camera` 安装目录
下的 `config/argus_camera.yaml`。修改配置文件后重启节点即可生效，也可以通过环境变量
`ARGUS_CAMERA_CONFIG` 指定其他 YAML 文件：

```bash
ARGUS_CAMERA_CONFIG=/path/to/argus_camera.yaml \
  ros2 launch argus_bringup argus_pipeline.launch.py
```

`camera_index` 和 `sensor_mode_index` 分别选择 LibArgus 枚举到的摄像头与 sensor mode；
`frame_count=0` 表示持续采集；`frame_rate=0` 表示使用 sensor mode 默认帧率，否则按请求的
FPS 设置采集帧周期（超出传感器支持范围时由 Argus 取最接近值）。配置文件字段如下：

```text
topic, frame_id, frame_count, camera_index, sensor_mode_index, capture_buffer_count, frame_rate
saturation, exposure_compensation, isp_digital_gain
denoise_mode (off|fast|hq), denoise_strength
edge_enhance_mode (off|fast|hq), edge_enhance_strength
manual_white_balance, white_balance_gains: [r, g_even, g_odd, b]
```

这些配置在组件启动时读取并应用到 Argus request；运行中修改文件不会自动改变当前采集，
需要重启节点。

查看可视化输出：

```bash
ros2 topic echo /camera/image/compressed --once
ros2 run rqt_image_view rqt_image_view
```

自定义 YUV handle topic 使用 best-effort/keep-last（相机发布端深度为 4，JPEG 可视化节点
深度为 1）；intra-process communication 要求使用 keep-last 历史策略。JPEG 输出使用
`SensorDataQoS`（best effort）。采集 NVMM pool 默认包含 4 个 buffer，可使用
`capture_buffer_count` 修改；
应结合实际处理时延、下游队列深度与允许的帧数延迟调整。

YUV packet 使用 keep-last 队列；如果消费者处理速度低于采集速度，最旧的 packet 会被淘汰，
其 shared owner 随之归还 Argus Buffer，避免 NVMM buffer 永久滞留。慢消费者耗尽 pool 时会
对采集形成背压，因此 pool 深度应与下游队列和处理耗时匹配。

消息头时间戳取自 `ICaptureMetadata::getSensorTimestamp()`，即 Argus 报告的传感器时间戳
（纳秒）。该时间通常属于系统单调时钟域，不等同于 ROS 的 `/clock` 时间；与其他传感器
做时间同步时，应确保它们使用相同时间基准或在上层完成时钟转换。

`ArgusYuvFrame.frame_number` 由采集节点写入 `ICaptureMetadata::getCaptureId()`；推理和
可视化等下游节点只读取/透传该源帧序号，不在各自节点重新计数。`CompressedImage` 是标准
消息且没有帧序号字段，需通过其继承的时间戳与 YUV packet 关联。

不同 Jetson/传感器对 ISP 数字增益、色彩饱和度和手动白平衡的支持范围可能不同；如果驱动
拒绝某项启动参数，程序会打印 Argus status 并退出。
