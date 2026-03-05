class Metrics {
  const Metrics({
    required this.sharpness,
    required this.angle,
    required this.tiltVertical,
    required this.brightness,
    required this.darkness,
    required this.shake,
    required this.isSharpnessOk,
    required this.isAngleOk,
    required this.isTiltVerticalOk,
    required this.isBrightnessOk,
    required this.isDarknessOk,
    required this.isShakeOk,
  });

  final double sharpness;
  final double angle;
  final double tiltVertical;
  final double brightness;
  final double darkness;
  final double shake;
  final bool isSharpnessOk;
  final bool isAngleOk;
  final bool isTiltVerticalOk;
  final bool isBrightnessOk;
  final bool isDarknessOk;
  final bool isShakeOk;

  bool get isCaptureReady =>
      isSharpnessOk &&
      isAngleOk &&
      isTiltVerticalOk &&
      isBrightnessOk &&
      isShakeOk;

  static const double _sharpnessMin = 60;
  static const double _angleMax = 5;
  static const double _tiltVerticalMax = 5;
  static const double _brightnessMin = 30;
  static const double _brightnessMax = 85;
  static const double _darknessMax = 70;
  static const double _shakeMax = 1.5;

  static Metrics initial() {
    return const Metrics(
      sharpness: 72,
      angle: 3,
      tiltVertical: 0,
      brightness: 64,
      darkness: 36,
      shake: 0.8,
      isSharpnessOk: true,
      isAngleOk: true,
      isTiltVerticalOk: true,
      isBrightnessOk: true,
      isDarknessOk: true,
      isShakeOk: true,
    );
  }

  Metrics withStats({
    required double mean,
    required double stddev,
    required double angleDeg,
    required double tiltVerticalDeg,
    required double shakeRaw,
    double? darknessPercent,
  }) {
    final brightnessTarget = _mapToPercent(mean, 0, 255);
    final darknessTarget = _clamp(
      darknessPercent ?? (100 - brightnessTarget),
      0,
      100,
    );
    final sharpnessTarget = _mapToPercent(stddev, 0, 64);
    final angleTarget = _clamp(angleDeg, -45, 45);
    final tiltVerticalTarget = _clamp(tiltVerticalDeg, -30, 30);
    final shakeTarget = _mapToPercent(shakeRaw, 0, 30) / 100 * 5;

    final nextSharpness = _smooth(sharpness, sharpnessTarget, 0.25);
    final nextAngle = _smooth(angle, angleTarget, 0.2);
    final nextTiltVertical = _smooth(tiltVertical, tiltVerticalTarget, 0.2);
    final nextBrightness = _smooth(brightness, brightnessTarget, 0.25);
    final nextDarkness = _smooth(darkness, darknessTarget, 0.25);
    final nextShake = _smooth(shake, _clamp(shakeTarget, 0, 5), 0.25);

    return Metrics(
      sharpness: nextSharpness,
      angle: nextAngle,
      tiltVertical: nextTiltVertical,
      brightness: nextBrightness,
      darkness: nextDarkness,
      shake: nextShake,
      isSharpnessOk: nextSharpness >= _sharpnessMin,
      isAngleOk: nextAngle.abs() <= _angleMax,
      isTiltVerticalOk: nextTiltVertical.abs() <= _tiltVerticalMax,
      isBrightnessOk:
          nextBrightness >= _brightnessMin && nextBrightness <= _brightnessMax,
      isDarknessOk: nextDarkness <= _darknessMax,
      isShakeOk: nextShake <= _shakeMax,
    );
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
