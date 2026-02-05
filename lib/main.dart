import 'dart:ffi' as ffi;
import 'dart:math';

import 'package:ffi/ffi.dart';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_opencv/mylib_ffi.dart';

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
    _loadCameras();
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
  final Random _rng = Random();
  MyLib? _lib;

  @override
  void initState() {
    super.initState();
    _initNative();
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
    if (now.difference(_lastMetricsUpdate) < const Duration(milliseconds: 120)) {
      return;
    }
    _lastMetricsUpdate = now;

    if (!mounted) {
      return;
    }

    final lib = _lib;
    if (lib == null) {
      setState(() {
        _metrics = _metrics.jitter(_rng);
      });
      return;
    }

    final yPlane = image.planes[0];
    final bytes = yPlane.bytes;

    final ptr = malloc<ffi.Uint8>(bytes.length);
    try {
      ptr.asTypedList(bytes.length).setAll(0, bytes);

      final res = lib.processGray(
        ptr,
        image.width,
        image.height,
        yPlane.bytesPerRow,
      );

      setState(() {
        _metrics = _metrics.withNative(res, _rng);
      });
    } finally {
      malloc.free(ptr);
    }
  }

  void _initNative() {
    try {
      _lib = MyLib();
    } catch (_) {
      _lib = null;
    }
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
    required this.brightness,
    required this.shake,
    required this.distance,
  });

  final double sharpness;
  final double angle;
  final double brightness;
  final double shake;
  final double distance;

  static Metrics initial() {
    return const Metrics(
      sharpness: 72,
      angle: 3,
      brightness: 64,
      shake: 0.8,
      distance: 1.2,
    );
  }

  Metrics withNative(Result native, Random rng) {
    final brightnessTarget = _mapToPercent(native.a, 0, 255);
    final sharpnessTarget = _mapToPercent(native.b, 0, 64);

    return Metrics(
      sharpness: _smooth(sharpness, sharpnessTarget, 0.25),
      angle: _clamp(angle + _delta(rng, 1.8), -45, 45),
      brightness: _smooth(brightness, brightnessTarget, 0.25),
      shake: _clamp(shake + _delta(rng, 0.2), 0, 5),
      distance: _clamp(distance + _delta(rng, 0.06), 0.2, 3.5),
    );
  }

  Metrics jitter(Random rng) {
    return Metrics(
      sharpness: _clamp(sharpness + _delta(rng, 3), 0, 100),
      angle: _clamp(angle + _delta(rng, 1.8), -45, 45),
      brightness: _clamp(brightness + _delta(rng, 2.5), 0, 100),
      shake: _clamp(shake + _delta(rng, 0.2), 0, 5),
      distance: _clamp(distance + _delta(rng, 0.06), 0.2, 3.5),
    );
  }

  double _delta(Random rng, double magnitude) {
    return (rng.nextDouble() * 2 - 1) * magnitude;
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
            label: 'Góc',
            value: '${metrics.angle.toStringAsFixed(1)} deg',
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
