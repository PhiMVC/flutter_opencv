import 'package:flutter/material.dart';

import '../models/detection.dart';

class DetectionPainter extends CustomPainter {
  DetectionPainter({
    required this.detections,
    required this.imageSize,
    required this.quarterTurns,
    required this.isFrontCamera,
    required this.inferenceMs,
    required this.modelError,
  });

  final List<Detection> detections;
  final Size? imageSize;
  final int quarterTurns;
  final bool isFrontCamera;
  final int inferenceMs;
  final String? modelError;

  @override
  void paint(Canvas canvas, Size size) {
    final imgSize = imageSize;
    if (imgSize == null) {
      _drawStatus(canvas, size);
      return;
    }

    final orientedSize = _rotatedSize(imgSize, quarterTurns);
    final scaleX = size.width / orientedSize.width;
    final scaleY = size.height / orientedSize.height;

    final boxPaint =
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Colors.lightGreenAccent;

    for (final det in detections) {
      Rect rect = det.rect;
      if (quarterTurns != 0) {
        rect = _rotateRect(rect, imgSize, quarterTurns);
      }
      if (isFrontCamera) {
        rect = Rect.fromLTRB(
          orientedSize.width - rect.right,
          rect.top,
          orientedSize.width - rect.left,
          rect.bottom,
        );
      }

      final drawRect = Rect.fromLTRB(
        rect.left * scaleX,
        rect.top * scaleY,
        rect.right * scaleX,
        rect.bottom * scaleY,
      );
      canvas.drawRect(drawRect, boxPaint);
      _drawLabel(canvas, drawRect, det);
    }

    _drawStatus(canvas, size);
  }

  void _drawLabel(Canvas canvas, Rect rect, Detection det) {
    final label = 'ID ${det.classId} ${(det.score * 100).toStringAsFixed(1)}%';
    final textPainter = TextPainter(
      text: TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final padding = 4.0;
    var left = rect.left;
    var top = rect.top - textPainter.height - padding * 2;
    if (top < 0) {
      top = rect.top + padding;
    }
    if (left + textPainter.width + padding * 2 > rect.right) {
      left = rect.right - textPainter.width - padding * 2;
    }
    if (left < 0) {
      left = 0;
    }

    final bgRect = Rect.fromLTWH(
      left,
      top,
      textPainter.width + padding * 2,
      textPainter.height + padding * 2,
    );
    final bgPaint = Paint()..color = Colors.black.withValues(alpha: 0.6);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(4)),
      bgPaint,
    );
    textPainter.paint(canvas, Offset(left + padding, top + padding));
  }

  void _drawStatus(Canvas canvas, Size size) {
    final lines = <String>[];
    if (modelError != null) {
      lines.add(modelError!);
    } else if (inferenceMs > 0) {
      lines.add('TF Lite: ${inferenceMs}ms');
    }
    if (lines.isEmpty) {
      return;
    }

    final textPainter = TextPainter(
      text: TextSpan(
        text: lines.join('\n'),
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: size.width - 20);

    final padding = 6.0;
    final bgRect = Rect.fromLTWH(
      8,
      8,
      textPainter.width + padding * 2,
      textPainter.height + padding * 2,
    );
    final bgPaint = Paint()..color = Colors.black.withValues(alpha: 0.5);
    canvas.drawRRect(
      RRect.fromRectAndRadius(bgRect, const Radius.circular(6)),
      bgPaint,
    );
    textPainter.paint(canvas, Offset(8 + padding, 8 + padding));
  }

  Size _rotatedSize(Size size, int quarterTurns) {
    final turns = quarterTurns % 4;
    if (turns.isOdd) {
      return Size(size.height, size.width);
    }
    return size;
  }

  Rect _rotateRect(Rect rect, Size size, int quarterTurns) {
    final turns = quarterTurns % 4;
    if (turns == 0) {
      return rect;
    }

    final points =
        <Offset>[
          rect.topLeft,
          rect.topRight,
          rect.bottomRight,
          rect.bottomLeft,
        ].map((p) => _rotatePoint(p, size, turns)).toList();

    var minX = points.first.dx;
    var maxX = points.first.dx;
    var minY = points.first.dy;
    var maxY = points.first.dy;
    for (final p in points.skip(1)) {
      if (p.dx < minX) minX = p.dx;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dy > maxY) maxY = p.dy;
    }

    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  Offset _rotatePoint(Offset p, Size size, int quarterTurns) {
    switch (quarterTurns % 4) {
      case 1:
        return Offset(size.height - p.dy, p.dx);
      case 2:
        return Offset(size.width - p.dx, size.height - p.dy);
      case 3:
        return Offset(p.dy, size.width - p.dx);
      default:
        return p;
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.quarterTurns != quarterTurns ||
        oldDelegate.isFrontCamera != isFrontCamera ||
        oldDelegate.inferenceMs != inferenceMs ||
        oldDelegate.modelError != modelError;
  }
}
