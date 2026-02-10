import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../models/metrics.dart';

class MetricsService {
  MetricsService({required this.maxDim});

  final int maxDim;
  Uint8List? _prevLuma;

  Metrics process({
    required Metrics current,
    required CameraImage image,
  }) {
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
          final (gradX, gradY) = _computeGradients(metricsMat);
          double angleDeg;
          double tiltVerticalDeg;
          try {
            angleDeg = _dominantAngleDegrees(gradX, gradY);
            tiltVerticalDeg = _verticalTiltDegrees(
              gradX,
              gradY,
              metricsMat.width,
              metricsMat.height,
            );
          } finally {
            gradX.dispose();
            gradY.dispose();
          }

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
}
