# LibArgus ROS2 同进程管线

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

组件默认按 inference、visualization、camera 顺序加载；相机组件不等待订阅者发现即可开始
采集。以下常用参数可在 launch 命令中传入：

```bash
ros2 launch argus_bringup argus_pipeline.launch.py \
  camera_index:=0 sensor_mode_index:=0 frame_count:=100 \
  capture_buffer_count:=8 frame_rate:=30 frame_id:=camera
```

`camera_index` 和 `sensor_mode_index` 分别选择 LibArgus 枚举到的摄像头与 sensor mode；
`frame_count=0` 表示持续采集；`frame_rate=0` 表示使用 sensor mode 默认帧率，否则按请求的
FPS 设置采集帧周期（超出传感器支持范围时由 Argus 取最接近值）。采集节点还提供以下启动参数，
适合写入 ROS 参数 YAML：

```text
topic, frame_id, frame_count, camera_index, sensor_mode_index, capture_buffer_count, frame_rate
saturation, exposure_compensation, isp_digital_gain
denoise_mode (off|fast|hq), denoise_strength
edge_enhance_mode (off|fast|hq), edge_enhance_strength
manual_white_balance, white_balance_gains: [r, g_even, g_odd, b]
```

这些参数在组件启动时应用到 Argus request；运行中不再读取终端标准输入。

查看可视化输出：

```bash
ros2 topic echo /camera/image/compressed --once
ros2 run rqt_image_view rqt_image_view
```

自定义 YUV handle topic 使用 reliable/keep-last（深度 8）；intra-process communication
要求使用 keep-last 历史策略。JPEG 输出使用 `SensorDataQoS`（best effort）。采集 NVMM
pool 默认包含 8 个 buffer，可使用 `capture_buffer_count` 修改；应结合实际处理时延、下游
队列深度与允许的帧数延迟调整。

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
