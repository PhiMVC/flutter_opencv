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
