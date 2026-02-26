import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

/// Simple luma-based embedding placeholder.
/// Thay thế bằng TFLite embedding model khi đã có.
class ImageEmbeddingService {
  ImageEmbeddingService({this.targetSize = 112});

  final int targetSize;

  Future<Float32List?> computeEmbedding(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final codec = await ui.instantiateImageCodec(
        bytes,
        targetWidth: targetSize,
        targetHeight: targetSize,
      );
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      image.dispose();
      codec.dispose();
      if (data == null) {
        return null;
      }

      final rgba = data.buffer.asUint8List();
      final out = Float32List(targetSize * targetSize);
      var outIndex = 0;
      for (var i = 0; i < rgba.length; i += 4) {
        final r = rgba[i].toDouble();
        final g = rgba[i + 1].toDouble();
        final b = rgba[i + 2].toDouble();
        final luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        out[outIndex++] = luma;
      }

      // Mean-center + standardize to avoid high cosine similarity
      // for unrelated positive-only vectors.
      double sum = 0;
      for (var i = 0; i < out.length; i++) {
        sum += out[i];
      }
      final mean = sum / out.length;
      double sumSq = 0;
      for (var i = 0; i < out.length; i++) {
        final v = out[i] - mean;
        out[i] = v;
        sumSq += v * v;
      }
      final std = math.sqrt(sumSq / out.length);
      if (std > 0) {
        for (var i = 0; i < out.length; i++) {
          out[i] = out[i] / std;
        }
      }
      return out;
    } catch (_) {
      return null;
    }
  }
}
