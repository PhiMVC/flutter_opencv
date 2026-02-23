import 'dart:math' as math;

import 'package:flutter/material.dart';

class AxesOverlay extends StatelessWidget {
  const AxesOverlay({
    super.key,
    required this.sensorUp,
  });

  final Offset sensorUp;

  @override
  Widget build(BuildContext context) {
    return SizedBox.expand(
      child: CustomPaint(
        painter: _AxesOverlayPainter(sensorUp: sensorUp),
      ),
    );
  }
}

class _AxesOverlayPainter extends CustomPainter {
  _AxesOverlayPainter({required this.sensorUp});

  final Offset sensorUp;

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) {
      return;
    }

    final center = size.center(Offset.zero);
    final baseLength = math.min(size.width, size.height) * 0.18;

    final screenPaint =
        Paint()
          ..color = Colors.white.withValues(alpha: 0.35)
          ..strokeWidth = 1.5
          ..style = PaintingStyle.stroke;
    final sensorPaint =
        Paint()
          ..color = Colors.lightGreenAccent.withValues(alpha: 0.85)
          ..strokeWidth = 2
          ..style = PaintingStyle.stroke;

    final screenX = const Offset(1, 0);
    final screenY = const Offset(0, -1);
    final sensorY = _normalizeOffset(sensorUp);
    final sensorX = Offset(-sensorY.dy, sensorY.dx);

    _drawAxes(
      canvas: canvas,
      center: center,
      length: baseLength,
      xAxis: screenX,
      yAxis: screenY,
      paint: screenPaint,
    );

    _drawAxes(
      canvas: canvas,
      center: center,
      length: baseLength * 1.05,
      xAxis: sensorX,
      yAxis: sensorY,
      paint: sensorPaint,
    );

    final centerPaint =
        Paint()
          ..color = Colors.white.withValues(alpha: 0.6)
          ..style = PaintingStyle.fill;
    canvas.drawCircle(center, 2.4, centerPaint);
  }

  void _drawAxes({
    required Canvas canvas,
    required Offset center,
    required double length,
    required Offset xAxis,
    required Offset yAxis,
    required Paint paint,
  }) {
    final xDir = _normalizeOffset(xAxis);
    final yDir = _normalizeOffset(yAxis);

    final xStart = center - xDir * length;
    final xEnd = center + xDir * length;
    final yStart = center - yDir * length;
    final yEnd = center + yDir * length;

    canvas.drawLine(xStart, xEnd, paint);
    canvas.drawLine(yStart, yEnd, paint);

  }

  Offset _normalizeOffset(Offset value) {
    final length = value.distance;
    if (length <= 0) {
      return const Offset(0, -1);
    }
    return value / length;
  }

  @override
  bool shouldRepaint(covariant _AxesOverlayPainter oldDelegate) {
    return oldDelegate.sensorUp != sensorUp;
  }
}
