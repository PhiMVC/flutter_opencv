import 'dart:async';
import 'dart:math' as math;
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:sensors_plus/sensors_plus.dart';

import '../models/metrics.dart';

class MetricsService {
  MetricsService({
    required this.maxDim,
    this.smoothing = 0.3,
    this.maxTiltAngle = 45,
    this.sensorInterval = SensorInterval.gameInterval,
    this.axesSmoothing = 0.35,
  }) {
    _startSensors();
  }

  final int maxDim;
  final double smoothing;
  final double maxTiltAngle;
  final Duration sensorInterval;
  final double axesSmoothing;

  Uint8List? _prevLuma;
  StreamSubscription<AccelerometerEvent>? _tiltSubscription;
  bool _sensorFailed = false;
  double _tiltRollDeg = 0;
  double _tiltPitchDeg = 0;
  Offset _sensorUpDir = const Offset(0, -1);
  final ValueNotifier<Offset> _sensorUpNotifier = ValueNotifier<Offset>(
    const Offset(0, -1),
  );
  bool _calibrated = false;
  int _calibrationCount = 0;
  double _calibrationRollSum = 0;
  double _calibrationPitchSum = 0;
  double _rollZero = 0;
  double _pitchZero = 0;
  static const double _zeroThreshold = 3.0;
  static const double _zeroAlpha = 0.08;
  static const double _minUpProjection = 0.08;

  ValueListenable<Offset> get sensorUpListenable => _sensorUpNotifier;

  Metrics process({required Metrics current, required CameraImage image}) {
    _startSensors();
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
      final metricsMat = _resizeForMetrics(mat);
      final ownsMetricsMat = !identical(metricsMat, mat);
      try {
        final (mean, stddev) = cv.meanStdDev(metricsMat);
        try {
          final angleDeg = _tiltRollDeg;
          final tiltVerticalDeg = _tiltPitchDeg;

          final shakeRaw = _frameDifference(packed, _prevLuma);
          _updatePrevLuma(packed);

          return current.withStats(
            mean: mean.val1,
            stddev: stddev.val1,
            angleDeg: angleDeg,
            tiltVerticalDeg: tiltVerticalDeg,
            shakeRaw: shakeRaw,
          );
        } finally {
          mean.dispose();
          stddev.dispose();
        }
      } finally {
        if (ownsMetricsMat) {
          metricsMat.dispose();
        }
      }
    } finally {
      mat.dispose();
    }
  }

  void _updatePrevLuma(Uint8List packed) {
    if (_prevLuma == null || _prevLuma!.length != packed.length) {
      _prevLuma = Uint8List(packed.length);
    }
    _prevLuma!.setRange(0, packed.length, packed);
  }

  cv.Mat _resizeForMetrics(cv.Mat src) {
    final maxSide = math.max(src.width, src.height);
    if (maxSide <= maxDim) {
      return src;
    }
    final scale = maxDim / maxSide;
    final targetW = math.max(1, (src.width * scale).round());
    final targetH = math.max(1, (src.height * scale).round());
    return cv.resize(src, (targetW, targetH));
  }

  void dispose() {
    _tiltSubscription?.cancel();
    _tiltSubscription = null;
    _sensorUpNotifier.dispose();
  }

  void _startSensors() {
    if (_sensorFailed || _tiltSubscription != null) {
      return;
    }

    try {
      _tiltSubscription = accelerometerEventStream(
        samplingPeriod: sensorInterval,
      ).listen(
        _handleAccelerometer,
        onError: (_) {
          _sensorFailed = true;
          _tiltSubscription?.cancel();
          _tiltSubscription = null;
        },
      );
    } catch (_) {
      _sensorFailed = true;
    }
  }

  void _handleAccelerometer(AccelerometerEvent event) {
    final ax = event.x;
    final ay = event.y;
    final az = event.z;

    final gravity = math.sqrt(ax * ax + ay * ay + az * az);
    if (gravity == 0) {
      return;
    }

    final nx = ax / gravity;
    final ny = ay / gravity;
    final projectedUp = Offset(-nx, -ny);
    final projLength = projectedUp.distance;
    if (projLength >= _minUpProjection) {
      final targetUpDevice = projectedUp / projLength;
      // Convert device coords (y up) to canvas coords (y down).
      final targetUpCanvas = Offset(targetUpDevice.dx, -targetUpDevice.dy);
      final smoothedUp =
          Offset.lerp(_sensorUpDir, targetUpCanvas, axesSmoothing) ??
          targetUpCanvas;
      final normalizedUp = _normalizeOffset(smoothedUp);
      _sensorUpDir = normalizedUp;
      _sensorUpNotifier.value = normalizedUp;
    }

    // Map gravity to roll/pitch when the phone is held upright (portrait).
    // Using -ay as the vertical reference keeps 0° when the phone is straight.
    // Invert roll to match intuitive left/right tilt direction.
    var rollTarget = -math.atan2(ax, -ay) * 180 / math.pi;
    var pitchTarget = math.atan2(az, -ay) * 180 / math.pi;

    if (!_calibrated) {
      _calibrationRollSum += rollTarget;
      _calibrationPitchSum += pitchTarget;
      _calibrationCount++;
      if (_calibrationCount >= 25) {
        _rollZero = _calibrationRollSum / _calibrationCount;
        _pitchZero = _calibrationPitchSum / _calibrationCount;
        _calibrated = true;
      }
      return;
    }

    if (rollTarget.abs() < _zeroThreshold &&
        pitchTarget.abs() < _zeroThreshold) {
      _rollZero = _smooth(_rollZero, rollTarget, _zeroAlpha);
      _pitchZero = _smooth(_pitchZero, pitchTarget, _zeroAlpha);
    }

    rollTarget -= _rollZero;
    pitchTarget -= _pitchZero;

    final smoothedRoll = _smooth(_tiltRollDeg, rollTarget, smoothing);
    final smoothedPitch = _smooth(_tiltPitchDeg, pitchTarget, smoothing);

    _tiltRollDeg = _clamp(smoothedRoll, -maxTiltAngle, maxTiltAngle);
    _tiltPitchDeg = _clamp(smoothedPitch, -maxTiltAngle, maxTiltAngle);
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

  Offset _normalizeOffset(Offset value) {
    final length = value.distance;
    if (length <= 0) {
      return const Offset(0, -1);
    }
    return value / length;
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
}
