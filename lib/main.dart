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
          tiltVerticalDeg = _verticalTiltDegrees(
            gradX,
            gradY,
            image.width,
            image.height,
          );
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
            left: 12,
            right: 12,
            bottom: 12,
            child: SafeArea(top: false, child: MetricsPanel(metrics: _metrics)),
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
    required this.frontal,
    required this.brightness,
    required this.shake,
    required this.distance,
    required this.isSharpnessOk,
    required this.isAngleOk,
    required this.isTiltVerticalOk,
    required this.isFrontalOk,
    required this.isBrightnessOk,
    required this.isShakeOk,
    required this.isDistanceOk,
  });

  final double sharpness;
  final double angle;
  final double tiltVertical;
  final double frontal;
  final double brightness;
  final double shake;
  final double distance;
  final bool isSharpnessOk;
  final bool isAngleOk;
  final bool isTiltVerticalOk;
  final bool isFrontalOk;
  final bool isBrightnessOk;
  final bool isShakeOk;
  final bool isDistanceOk;

  bool get isCaptureReady =>
      isSharpnessOk &&
      isAngleOk &&
      isTiltVerticalOk &&
      isFrontalOk &&
      isBrightnessOk &&
      isShakeOk &&
      isDistanceOk;

  static const double _sharpnessMin = 60;
  static const double _angleMax = 5;
  static const double _tiltVerticalMax = 5;
  static const double _frontalMin = 80;
  static const double _brightnessMin = 30;
  static const double _brightnessMax = 85;
  static const double _shakeMax = 1.5;
  static const double _distanceMin = 0.5;
  static const double _distanceMax = 2.0;

  static Metrics initial() {
    return const Metrics(
      sharpness: 72,
      angle: 3,
      tiltVertical: 0,
      frontal: 88,
      brightness: 64,
      shake: 0.8,
      distance: 1.2,
      isSharpnessOk: true,
      isAngleOk: true,
      isTiltVerticalOk: true,
      isFrontalOk: true,
      isBrightnessOk: true,
      isShakeOk: true,
      isDistanceOk: true,
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
    final frontalTarget = _frontalScore(angleDeg, tiltVerticalDeg);
    final shakeTarget = _mapToPercent(shakeRaw, 0, 30) / 100 * 5;
    final distanceTarget = _mapRange(100 - sharpnessTarget, 0, 100, 0.2, 3.5);

    final nextSharpness = _smooth(sharpness, sharpnessTarget, 0.25);
    final nextAngle = _smooth(angle, angleTarget, 0.2);
    final nextTiltVertical = _smooth(tiltVertical, tiltVerticalTarget, 0.2);
    final nextFrontal = _smooth(frontal, frontalTarget, 0.2);
    final nextBrightness = _smooth(brightness, brightnessTarget, 0.25);
    final nextShake = _smooth(shake, _clamp(shakeTarget, 0, 5), 0.25);
    final nextDistance = _smooth(
      distance,
      _clamp(distanceTarget, 0.2, 3.5),
      0.2,
    );

    return Metrics(
      sharpness: nextSharpness,
      angle: nextAngle,
      tiltVertical: nextTiltVertical,
      frontal: nextFrontal,
      brightness: nextBrightness,
      shake: nextShake,
      distance: nextDistance,
      isSharpnessOk: nextSharpness >= _sharpnessMin,
      isAngleOk: nextAngle.abs() <= _angleMax,
      isTiltVerticalOk: nextTiltVertical.abs() <= _tiltVerticalMax,
      isFrontalOk: nextFrontal >= _frontalMin,
      isBrightnessOk:
          nextBrightness >= _brightnessMin && nextBrightness <= _brightnessMax,
      isShakeOk: nextShake <= _shakeMax,
      isDistanceOk:
          nextDistance >= _distanceMin && nextDistance <= _distanceMax,
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

  double _frontalScore(double angleDeg, double tiltVerticalDeg) {
    const maxAngle = 45.0;
    const maxTilt = 30.0;
    final angleRatio = (angleDeg.abs() / maxAngle);
    final tiltRatio = (tiltVerticalDeg.abs() / maxTilt);
    final normalized = _clamp((angleRatio + tiltRatio) / 2, 0, 1);
    return (1 - normalized) * 100;
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
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Builder(
        builder: (context) {
          final items = <Widget>[
            MetricRow(
              label: 'Nét',
              value: '${metrics.sharpness.toStringAsFixed(0)}%',
              isValid: metrics.isSharpnessOk,
            ),
            MetricRow(
              label: 'Ngang',
              value: '${metrics.angle.toStringAsFixed(1)} deg',
              isValid: metrics.isAngleOk,
            ),
            MetricRow(
              label: 'Dọc',
              value: '${metrics.tiltVertical.toStringAsFixed(1)} deg',
              isValid: metrics.isTiltVerticalOk,
            ),
            MetricRow(
              label: 'Thẳng',
              value: '${metrics.frontal.toStringAsFixed(0)}%',
              isValid: metrics.isFrontalOk,
            ),
            MetricRow(
              label: 'Sáng',
              value: '${metrics.brightness.toStringAsFixed(0)}%',
              isValid: metrics.isBrightnessOk,
            ),
            MetricRow(
              label: 'Rung',
              value: metrics.shake.toStringAsFixed(2),
              isValid: metrics.isShakeOk,
            ),
            MetricRow(
              label: 'Cách',
              value: '${metrics.distance.toStringAsFixed(2)} m',
              isValid: metrics.isDistanceOk,
            ),
          ];

          final firstRow = items.take(4).toList();
          final secondRow = items.skip(4).toList();
          while (secondRow.length < 4) {
            secondRow.add(const _MetricPlaceholder());
          }

          Widget cell(Widget child) {
            return Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
                child: child,
              ),
            );
          }

          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(children: firstRow.map(cell).toList()),
              const SizedBox(height: 8),
              Row(children: secondRow.map(cell).toList()),
            ],
          );
        },
      ),
    );
  }
}

class MetricRow extends StatelessWidget {
  const MetricRow({
    super.key,
    required this.label,
    required this.value,
    required this.isValid,
  });

  final String label;
  final String value;
  final bool isValid;

  @override
  Widget build(BuildContext context) {
    final statusColor = isValid ? Colors.greenAccent : Colors.redAccent;
    return SizedBox(
      height: _metricItemHeight,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: statusColor.withValues(alpha: 0.8)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.8),
                fontSize: 11,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

const double _metricItemHeight = 60;

class _MetricPlaceholder extends StatelessWidget {
  const _MetricPlaceholder();

  @override
  Widget build(BuildContext context) {
    return const SizedBox(height: _metricItemHeight);
  }
}
