import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CameraDemoApp());
}

class CameraDemoApp extends StatefulWidget {
  const CameraDemoApp({super.key});

  @override
  State<CameraDemoApp> createState() => _CameraDemoAppState();
}

class _CameraDemoAppState extends State<CameraDemoApp> {
  List<CameraDescription> _cameras = const [];
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    if (_cameras.isEmpty) {
      _loadCameras();
    }
  }

  Future<void> _loadCameras() async {
    try {
      final cameras = await availableCameras();
      if (!mounted) {
        return;
      }
      setState(() {
        _cameras = cameras;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Camera error: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    Widget home;
    if (_loading) {
      home = const Scaffold(body: Center(child: CircularProgressIndicator()));
    } else if (_error != null) {
      home = Scaffold(
        body: Center(
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    } else {
      home = CameraHome(cameras: _cameras);
    }

    return MaterialApp(
      title: 'Camera Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: Colors.black),
      home: home,
    );
  }
}

class CameraHome extends StatefulWidget {
  const CameraHome({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<CameraHome> createState() => _CameraHomeState();
}

class _CameraHomeState extends State<CameraHome> {
  CameraController? _controller;
  Metrics _metrics = Metrics.initial();
  String? _error;
  bool _isStreaming = false;
  DateTime _lastMetricsUpdate = DateTime.fromMillisecondsSinceEpoch(0);
  Uint8List? _prevLuma;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _stopImageStream();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() {
        _error = 'No camera found on this device.';
      });
      return;
    }

    final camera = widget.cameras.first;

    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller.initialize();
      _controller = controller;
      if (mounted) {
        setState(() {});
      }
      await _startImageStream();
    } on CameraException catch (e) {
      setState(() {
        _error = 'Camera error: ${e.code}';
      });
    }
  }

  Future<void> _startImageStream() async {
    final controller = _controller;
    if (controller == null || _isStreaming) {
      return;
    }

    try {
      await controller.startImageStream(_onImage);
      if (mounted) {
        setState(() {
          _isStreaming = true;
        });
      } else {
        _isStreaming = true;
      }
    } on CameraException catch (_) {
      // Ignore stream errors for the demo; preview still works.
    }
  }

  Future<void> _stopImageStream() async {
    final controller = _controller;
    if (controller == null || !_isStreaming) {
      return;
    }

    try {
      await controller.stopImageStream();
    } on CameraException catch (_) {
      // Ignore.
    } finally {
      if (mounted) {
        setState(() {
          _isStreaming = false;
        });
      } else {
        _isStreaming = false;
      }
    }
  }

  void _onImage(CameraImage image) {
    final now = DateTime.now();
    if (now.difference(_lastMetricsUpdate) <
        const Duration(milliseconds: 120)) {
      return;
    }
    _lastMetricsUpdate = now;

    if (!mounted) {
      return;
    }

    final yPlane = image.planes[0];
    final packed = _packLumaPlane(
      yPlane.bytes,
      image.width,
      image.height,
      yPlane.bytesPerRow,
    );

    final mat = cv.Mat.fromList(
      image.height,
      image.width,
      cv.MatType.CV_8UC1,
      packed,
    );
    try {
      final (mean, stddev) = cv.meanStdDev(mat);
      try {
        final (gradX, gradY) = _computeGradients(mat);
        double angleDeg;
        double tiltVerticalDeg;
        try {
          angleDeg = _dominantAngleDegrees(gradX, gradY);
          tiltVerticalDeg =
              _verticalTiltDegrees(gradX, gradY, image.width, image.height);
        } finally {
          gradX.dispose();
          gradY.dispose();
        }

        final shakeRaw = _frameDifference(packed, _prevLuma);
        _prevLuma = Uint8List.fromList(packed);

        setState(() {
          _metrics = _metrics.withStats(
            mean: mean.val1,
            stddev: stddev.val1,
            angleDeg: angleDeg,
            tiltVerticalDeg: tiltVerticalDeg,
            shakeRaw: shakeRaw,
          );
        });
      } finally {
        mean.dispose();
        stddev.dispose();
      }
    } finally {
      mat.dispose();
    }
  }

  (cv.Mat, cv.Mat) _computeGradients(cv.Mat src) {
    final gradX = cv.sobel(src, cv.MatType.CV_32F, 1, 0, ksize: 3);
    final gradY = cv.sobel(src, cv.MatType.CV_32F, 0, 1, ksize: 3);
    return (gradX, gradY);
  }

  double _dominantAngleDegrees(cv.Mat gradX, cv.Mat gradY) {
    final gx = _matToFloat32(gradX);
    final gy = _matToFloat32(gradY);
    final length = gx.length < gy.length ? gx.length : gy.length;

    const sampleStep = 4;
    double sumXX = 0;
    double sumYY = 0;
    double sumXY = 0;
    for (var i = 0; i < length; i += sampleStep) {
      final dx = gx[i];
      final dy = gy[i];
      sumXX += dx * dx;
      sumYY += dy * dy;
      sumXY += dx * dy;
    }

    if (sumXX + sumYY == 0) {
      return 0;
    }

    final angleRad = 0.5 * math.atan2(2 * sumXY, sumXX - sumYY);
    return angleRad * 180 / math.pi;
  }

  double _verticalTiltDegrees(
    cv.Mat gradX,
    cv.Mat gradY,
    int width,
    int height,
  ) {
    if (width <= 0 || height <= 0) {
      return 0;
    }

    final gx = _matToFloat32(gradX);
    final gy = _matToFloat32(gradY);
    final length = gx.length < gy.length ? gx.length : gy.length;
    final total = width * height;
    final usable = length < total ? length : total;
    final half = height ~/ 2;

    const sampleStep = 4;
    double top = 0;
    double bottom = 0;
    for (var i = 0; i < usable; i += sampleStep) {
      final mag = gx[i].abs() + gy[i].abs();
      final y = i ~/ width;
      if (y < half) {
        top += mag;
      } else {
        bottom += mag;
      }
    }

    final denom = top + bottom;
    if (denom == 0) {
      return 0;
    }

    final imbalance = (bottom - top) / denom;
    return imbalance * 30;
  }

  Float32List _matToFloat32(cv.Mat mat) {
    final data = mat.data;
    final length = mat.total * mat.channels;
    return Float32List.view(data.buffer, data.offsetInBytes, length);
  }

  double _frameDifference(Uint8List current, Uint8List? prev) {
    if (prev == null || prev.length != current.length) {
      return 0;
    }

    const sampleStep = 4;
    var sum = 0;
    var count = 0;
    for (var i = 0; i < current.length; i += sampleStep) {
      sum += (current[i] - prev[i]).abs();
      count++;
    }
    if (count == 0) {
      return 0;
    }
    return sum / count;
  }

  Uint8List _packLumaPlane(
    Uint8List bytes,
    int width,
    int height,
    int rowStride,
  ) {
    if (rowStride == width) {
      return bytes;
    }

    final packed = Uint8List(width * height);
    var dst = 0;
    for (var y = 0; y < height; y++) {
      final srcStart = y * rowStride;
      packed.setRange(dst, dst + width, bytes, srcStart);
      dst += width;
    }
    return packed;
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        body: Center(
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: CameraPreview(controller)),
          Positioned(
            left: 16,
            top: 16,
            child: SafeArea(child: MetricsPanel(metrics: _metrics)),
          ),
          Positioned(
            right: 16,
            top: 16,
            child: SafeArea(child: StreamBadge(isStreaming: _isStreaming)),
          ),
        ],
      ),
    );
  }
}

class Metrics {
  const Metrics({
    required this.sharpness,
    required this.angle,
    required this.tiltVertical,
    required this.brightness,
    required this.shake,
    required this.distance,
  });

  final double sharpness;
  final double angle;
  final double tiltVertical;
  final double brightness;
  final double shake;
  final double distance;

  static Metrics initial() {
    return const Metrics(
      sharpness: 72,
      angle: 3,
      tiltVertical: 0,
      brightness: 64,
      shake: 0.8,
      distance: 1.2,
    );
  }

  Metrics withStats({
    required double mean,
    required double stddev,
    required double angleDeg,
    required double tiltVerticalDeg,
    required double shakeRaw,
  }) {
    final brightnessTarget = _mapToPercent(mean, 0, 255);
    final sharpnessTarget = _mapToPercent(stddev, 0, 64);
    final angleTarget = _clamp(angleDeg, -45, 45);
    final tiltVerticalTarget = _clamp(tiltVerticalDeg, -30, 30);
    final shakeTarget = _mapToPercent(shakeRaw, 0, 30) / 100 * 5;
    final distanceTarget =
        _mapRange(100 - sharpnessTarget, 0, 100, 0.2, 3.5);

    return Metrics(
      sharpness: _smooth(sharpness, sharpnessTarget, 0.25),
      angle: _smooth(angle, angleTarget, 0.2),
      tiltVertical: _smooth(tiltVertical, tiltVerticalTarget, 0.2),
      brightness: _smooth(brightness, brightnessTarget, 0.25),
      shake: _smooth(shake, _clamp(shakeTarget, 0, 5), 0.25),
      distance: _smooth(
        distance,
        _clamp(distanceTarget, 0.2, 3.5),
        0.2,
      ),
    );
  }

  double _mapRange(
    double value,
    double inMin,
    double inMax,
    double outMin,
    double outMax,
  ) {
    if (inMax <= inMin) {
      return outMin;
    }
    final t = _clamp((value - inMin) / (inMax - inMin), 0, 1);
    return outMin + (outMax - outMin) * t;
  }

  double _mapToPercent(double value, double min, double max) {
    if (max <= min) {
      return 0;
    }
    final clamped = _clamp(value, min, max);
    return ((clamped - min) / (max - min)) * 100;
  }

  double _smooth(double current, double target, double alpha) {
    return current + (target - current) * alpha;
  }

  double _clamp(double value, double min, double max) {
    if (value < min) {
      return min;
    }
    if (value > max) {
      return max;
    }
    return value;
  }
}

class MetricsPanel extends StatelessWidget {
  const MetricsPanel({super.key, required this.metrics});

  final Metrics metrics;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 200,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Realtime Metrics',
            style: TextStyle(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.2,
            ),
          ),
          const SizedBox(height: 10),
          MetricRow(
            label: 'Nét',
            value: '${metrics.sharpness.toStringAsFixed(0)}%',
          ),
          MetricRow(
            label: 'Nghiêng ngang',
            value: '${metrics.angle.toStringAsFixed(1)} deg',
          ),
          MetricRow(
            label: 'Nghiêng dọc',
            value: '${metrics.tiltVertical.toStringAsFixed(1)} deg',
          ),
          MetricRow(
            label: 'Sáng',
            value: '${metrics.brightness.toStringAsFixed(0)}%',
          ),
          MetricRow(label: 'Rung', value: metrics.shake.toStringAsFixed(2)),
          MetricRow(
            label: 'Cách',
            value: '${metrics.distance.toStringAsFixed(2)} m',
          ),
        ],
      ),
    );
  }
}

class MetricRow extends StatelessWidget {
  const MetricRow({super.key, required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.85),
              fontSize: 13,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}

class StreamBadge extends StatelessWidget {
  const StreamBadge({super.key, required this.isStreaming});

  final bool isStreaming;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: isStreaming ? Colors.greenAccent : Colors.redAccent,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            isStreaming ? 'Stream ON' : 'Stream OFF',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
