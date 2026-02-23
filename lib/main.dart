import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'models/detection.dart';
import 'models/metrics.dart';
import 'services/detection_service.dart';
import 'services/metrics_service.dart';
import 'widgets/detection_painter.dart';
import 'widgets/axes_overlay.dart';
import 'widgets/metrics_panel.dart';

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
  static const int _minInferenceIntervalMs = 60;
  static const int _metricsMaxDim = 360;
  static const Duration _metricsInterval = Duration(milliseconds: 150);
  static const Duration _detectionHold = Duration(milliseconds: 250);
  static const double _scoreThreshold = 0.2;
  static const double _nmsThreshold = 0.9;
  static const int _maxDetections = 100;

  CameraController? _controller;
  Metrics _metrics = Metrics.initial();
  String? _error;
  bool _isStreaming = false;

  late final MetricsService _metricsService;
  late final DetectionService _detectionService;
  List<Detection> _detections = const [];
  DateTime _lastInference = DateTime.fromMillisecondsSinceEpoch(0);
  int _lastInferenceMs = 0;
  DateTime _lastMetrics = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastNonEmptyDetection = DateTime.fromMillisecondsSinceEpoch(0);
  List<Detection> _lastNonEmptyDetections = const [];
  Size? _lastNonEmptyImageSize;
  Size? _analysisImageSize;
  String? _modelError;
  @override
  void initState() {
    super.initState();
    _metricsService = MetricsService(maxDim: _metricsMaxDim);
    _detectionService = DetectionService(
      modelAsset: 'assets/best_float32.tflite',
      scoreThreshold: _scoreThreshold,
      nmsThreshold: _nmsThreshold,
      maxDetections: _maxDetections,
    );
    _initializeCamera();
    _loadModel();
  }

  @override
  void dispose() {
    _stopImageStream();
    _detectionService.dispose();
    _metricsService.dispose();
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

  Future<void> _loadModel() async {
    try {
      if (!mounted) {
        return;
      }

      await _detectionService.load();
      setState(() {
        _modelError = _detectionService.modelError;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _modelError = 'Model error: $e';
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
    if (!mounted) {
      return;
    }

    if (!_detectionService.isBusy &&
        now.difference(_lastMetrics) >= _metricsInterval) {
      _lastMetrics = now;
      try {
        final nextMetrics = _metricsService.process(
          current: _metrics,
          image: image,
        );
        if (mounted) {
          setState(() {
            _metrics = nextMetrics;
          });
        }
      } catch (_) {
        // Ignore metric errors to avoid blocking the preview.
      }
    }
    _maybeRunDetection(image, now);
  }

  void _maybeRunDetection(CameraImage image, DateTime now) {
    if (!_detectionService.isReady || _detectionService.isBusy) {
      return;
    }
    final minIntervalMs = _lastInferenceMs <= 0
        ? _minInferenceIntervalMs
        : math.max(_minInferenceIntervalMs, _lastInferenceMs);
    if (now.difference(_lastInference).inMilliseconds < minIntervalMs) {
      return;
    }
    _lastInference = now;
    _runDetection(image);
  }

  Future<void> _runDetection(CameraImage image) async {
    try {
      final start = DateTime.now();
      final result = await _detectionService.run(image);
      if (!mounted) {
        return;
      }
      if (result == null) {
        setState(() {
          _modelError = _detectionService.modelError;
        });
        return;
      }
      var nextDetections = result.detections;
      var nextImageSize = result.imageSize;
      if (nextDetections.isNotEmpty) {
        _lastNonEmptyDetection = start;
        _lastNonEmptyDetections = nextDetections;
        _lastNonEmptyImageSize = nextImageSize;
        final top = nextDetections.first;
        debugPrint(
          'TF detect: ${nextDetections.length} objects. '
          'Top id=${top.classId}, '
          'score=${(top.score * 100).toStringAsFixed(1)}%',
        );
      } else if (start.difference(_lastNonEmptyDetection) <= _detectionHold &&
          _lastNonEmptyDetections.isNotEmpty) {
        nextDetections = _lastNonEmptyDetections;
        if (_lastNonEmptyImageSize != null) {
          nextImageSize = _lastNonEmptyImageSize!;
        }
      }
      setState(() {
        _analysisImageSize = nextImageSize;
        _detections = nextDetections;
        _lastInferenceMs = result.inferenceMs;
        _modelError = _detectionService.modelError;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _modelError = 'Inference error: $e';
      });
    }
  }

  DeviceOrientation _getApplicableOrientation(CameraController controller) {
    return controller.value.isRecordingVideo
        ? controller.value.recordingOrientation!
        : (controller.value.previewPauseOrientation ??
            controller.value.lockedCaptureOrientation ??
            controller.value.deviceOrientation);
  }

  int _deviceOrientationToDegrees(DeviceOrientation orientation) {
    switch (orientation) {
      case DeviceOrientation.portraitUp:
        return 0;
      case DeviceOrientation.landscapeLeft:
        return 90;
      case DeviceOrientation.portraitDown:
        return 180;
      case DeviceOrientation.landscapeRight:
        return 270;
    }
  }

  int _overlayQuarterTurns(CameraController controller) {
    final orientation = _getApplicableOrientation(controller);
    final deviceDegrees = _deviceOrientationToDegrees(orientation);
    final sensorOrientation = controller.description.sensorOrientation;
    final isFront =
        controller.description.lensDirection == CameraLensDirection.front;
    final rotationDegrees = isFront
        ? (sensorOrientation + deviceDegrees) % 360
        : (sensorOrientation - deviceDegrees + 360) % 360;
    return (rotationDegrees ~/ 90) % 4;
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

    final rotateOverlay =
        !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
    final quarterTurns =
        rotateOverlay ? _overlayQuarterTurns(controller) : 0;

    return Material(
      child: Stack(
        alignment: Alignment.topCenter,
        children: [
          SafeArea(
            child: CameraPreview(
              controller,
              child: CustomPaint(
                painter: DetectionPainter(
                  detections: _detections,
                  imageSize: _analysisImageSize,
                  quarterTurns: rotateOverlay ? quarterTurns : 0,
                  isFrontCamera:
                      controller.description.lensDirection ==
                      CameraLensDirection.front,
                  inferenceMs: _lastInferenceMs,
                  modelError: _modelError,
                ),
              ),
            ),
          ),
          Positioned.fill(
            child: SafeArea(
              child: IgnorePointer(
                child: ValueListenableBuilder<Offset>(
                  valueListenable: _metricsService.sensorUpListenable,
                  builder: (context, sensorUp, _) {
                    return AxesOverlay(sensorUp: sensorUp);
                  },
                ),
              ),
            ),
          ),
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
