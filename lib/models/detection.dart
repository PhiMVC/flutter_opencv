import 'dart:ui';

class Detection {
  const Detection({
    required this.rect,
    required this.score,
    required this.classId,
  });

  final Rect rect;
  final double score;
  final int classId;
}
