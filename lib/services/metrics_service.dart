import 'dart:async';
import 'dart:isolate';
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
    this.useBackgroundMeanStdDev = false,
    this.profilingEnabled = false,
    this.profilingEveryN = 30,
  }) {
    _startSensors();
  }

  final int maxDim;
  final double smoothing;
  final double maxTiltAngle;
  final Duration sensorInterval;
  final double axesSmoothing;
  final bool useBackgroundMeanStdDev;
  final bool profilingEnabled;
  final int profilingEveryN;

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
  int _profileCount = 0;

  ValueListenable<Offset> get sensorUpListenable => _sensorUpNotifier;

  Metrics process({required Metrics current, required CameraImage image}) {
    _startSensors();
    final shouldProfile = _shouldProfile();
    final totalSw = shouldProfile ? Stopwatch() : null;
    totalSw?.start();
    final packSw = shouldProfile ? Stopwatch() : null;
    packSw?.start();
    final yPlane = image.planes[0];
    final packed = _packLumaPlane(
      yPlane.bytes,
      image.width,
      image.height,
      yPlane.bytesPerRow,
    );
    packSw?.stop();

    final matSw = shouldProfile ? Stopwatch() : null;
    matSw?.start();
    final mat = cv.Mat.fromList(
      image.height,
      image.width,
      cv.MatType.CV_8UC1,
      packed,
    );
    matSw?.stop();
    try {
      final resizeSw = shouldProfile ? Stopwatch() : null;
      resizeSw?.start();
      final metricsMat = _resizeForMetrics(mat);
      resizeSw?.stop();
      final ownsMetricsMat = !identical(metricsMat, mat);
      try {
        final statsSw = shouldProfile ? Stopwatch() : null;
        statsSw?.start();
        final (mean, stddev) = cv.meanStdDev(metricsMat);
        statsSw?.stop();
        try {
          final angleDeg = _tiltRollDeg;
          final tiltVerticalDeg = _tiltPitchDeg;

          final shakeSw = shouldProfile ? Stopwatch() : null;
          shakeSw?.start();
          final shakeRaw = _frameDifference(packed, _prevLuma);
          _updatePrevLuma(packed);
          shakeSw?.stop();

          final result = current.withStats(
            mean: mean.val1,
            stddev: stddev.val1,
            angleDeg: angleDeg,
            tiltVerticalDeg: tiltVerticalDeg,
            shakeRaw: shakeRaw,
          );

          if (shouldProfile) {
            totalSw?.stop();
            _logProfile(
              'Profile: pack=${packSw?.elapsedMilliseconds ?? 0}ms '
              'mat=${matSw?.elapsedMilliseconds ?? 0}ms '
              'resize=${resizeSw?.elapsedMilliseconds ?? 0}ms '
              'stats=${statsSw?.elapsedMilliseconds ?? 0}ms '
              'shake=${shakeSw?.elapsedMilliseconds ?? 0}ms '
              'total=${totalSw?.elapsedMilliseconds ?? 0}ms',
            );
          }

          return result;
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

  Future<Metrics> processAsync({
    required Metrics current,
    required CameraImage image,
  }) async {
    _startSensors();
    final shouldProfile = _shouldProfile();
    final totalSw = shouldProfile ? Stopwatch() : null;
    totalSw?.start();
    final packSw = shouldProfile ? Stopwatch() : null;
    packSw?.start();
    final yPlane = image.planes[0];
    final packed = _packLumaPlane(
      yPlane.bytes,
      image.width,
      image.height,
      yPlane.bytesPerRow,
    );
    packSw?.stop();

    final shakeSw = shouldProfile ? Stopwatch() : null;
    shakeSw?.start();
    final shakeRaw = _frameDifference(packed, _prevLuma);
    _updatePrevLuma(packed);
    shakeSw?.stop();

    final meanSw = shouldProfile ? Stopwatch() : null;
    final meanStd =
        useBackgroundMeanStdDev
            ? await compute(_meanStdDevWorker, <String, Object?>{
              'bytes': TransferableTypedData.fromList([
                Uint8List.fromList(packed),
              ]),
              'width': image.width,
              'height': image.height,
              'maxDim': maxDim,
            })
            : _meanStdDevLocal(packed, image.width, image.height);
    meanSw?.stop();

    final angleDeg = _tiltRollDeg;
    final tiltVerticalDeg = _tiltPitchDeg;

    final result = current.withStats(
      mean: (meanStd['mean'] ?? 0).toDouble(),
      stddev: (meanStd['stddev'] ?? 0).toDouble(),
      angleDeg: angleDeg,
      tiltVerticalDeg: tiltVerticalDeg,
      shakeRaw: shakeRaw,
    );

    if (shouldProfile) {
      totalSw?.stop();
      _logProfile(
        'Profile(async): pack=${packSw?.elapsedMilliseconds ?? 0}ms '
        'meanStd=${meanSw?.elapsedMilliseconds ?? 0}ms '
        'shake=${shakeSw?.elapsedMilliseconds ?? 0}ms '
        'total=${totalSw?.elapsedMilliseconds ?? 0}ms',
      );
    }

    return result;
  }

  Map<String, double> _meanStdDevLocal(Uint8List bytes, int width, int height) {
    return _meanStdDevFromBytes(bytes, width, height, maxDim);
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

  bool _shouldProfile() {
    if (!profilingEnabled) return false;
    _profileCount++;
    if (profilingEveryN <= 0) {
      return _profileCount == 1;
    }
    return _profileCount % profilingEveryN == 0;
  }

  void _logProfile(String message) {
    if (!profilingEnabled) return;
    debugPrint('[MetricsService] $message');
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

Map<String, double> _meanStdDevWorker(Map<String, Object?> args) {
  final bytes =
      (args['bytes'] as TransferableTypedData).materialize().asUint8List();
  final width = args['width'] as int? ?? 0;
  final height = args['height'] as int? ?? 0;
  final maxDim = args['maxDim'] as int? ?? 0;
  return _meanStdDevFromBytes(bytes, width, height, maxDim);
}

Map<String, double> _meanStdDevFromBytes(
  Uint8List bytes,
  int width,
  int height,
  int maxDim,
) {
  if (width <= 0 || height <= 0 || bytes.isEmpty) {
    return const {'mean': 0, 'stddev': 0};
  }

  var targetW = width;
  var targetH = height;
  final maxSide = math.max(width, height);
  if (maxDim > 0 && maxSide > maxDim) {
    final scale = maxDim / maxSide;
    targetW = math.max(1, (width * scale).round());
    targetH = math.max(1, (height * scale).round());
  }

  final count = targetW * targetH;
  if (count <= 0) {
    return const {'mean': 0, 'stddev': 0};
  }

  double sum = 0;
  double sumSq = 0;
  for (var y = 0; y < targetH; y++) {
    final srcY = ((y * height) / targetH).round().clamp(0, height - 1);
    final row = srcY * width;
    for (var x = 0; x < targetW; x++) {
      final srcX = ((x * width) / targetW).round().clamp(0, width - 1);
      final value = bytes[row + srcX].toDouble();
      sum += value;
      sumSq += value * value;
    }
  }

  final mean = sum / count;
  final variance = (sumSq / count) - (mean * mean);
  final stddev = math.sqrt(math.max(0, variance));

  return {'mean': mean, 'stddev': stddev};
}
