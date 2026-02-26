import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

/// Simple luma-based embedding placeholder.
/// Bộ embedding tạm dựa trên độ sáng (luma).
///
/// Replace this with a real TFLite embedding model when available.
/// Khi có model TFLite thật, hãy thay thế phần tạo embedding ở đây.
class ImageEmbeddingService {
  /// Target resize size for square input (targetSize x targetSize).
  /// Kích thước resize đầu vào dạng vuông (targetSize x targetSize).
  ImageEmbeddingService({this.targetSize = 112});

  final int targetSize;

  /// Compute a simple embedding from an image file.
  /// Tính embedding đơn giản từ file ảnh.
  ///
  /// Pipeline:
  /// 1) Decode + resize to targetSize.
  /// 1) Giải mã + resize về targetSize.
  /// 2) Convert RGBA -> luma (grayscale).
  /// 2) Chuyển RGBA -> luma (xám).
  /// 3) Mean-center + standardize to reduce cosine bias.
  /// 3) Trừ mean + chuẩn hóa theo std để giảm bias cosine.
  Future<Float32List?> computeEmbedding(File file) async {
    try {
      // Read image bytes from disk.
      // Đọc bytes ảnh từ bộ nhớ.
      final bytes = await file.readAsBytes();
      // Decode + resize to a fixed square size.
      // Giải mã + resize về kích thước vuông cố định.
      final codec = await ui.instantiateImageCodec(
        bytes,
        targetWidth: targetSize,
        targetHeight: targetSize,
      );
      final frame = await codec.getNextFrame();
      final image = frame.image;
      // Extract raw RGBA bytes.
      // Trích xuất bytes RGBA thô.
      final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      image.dispose();
      codec.dispose();
      if (data == null) {
        return null;
      }

      final rgba = data.buffer.asUint8List();
      // One float per pixel (luma).
      // Mỗi pixel là một float (luma).
      final out = Float32List(targetSize * targetSize);
      var outIndex = 0;
      for (var i = 0; i < rgba.length; i += 4) {
        final r = rgba[i].toDouble();
        final g = rgba[i + 1].toDouble();
        final b = rgba[i + 2].toDouble();
        // BT.601 luma, normalized to [0..1].
        // Luma BT.601, chuẩn hóa về [0..1].
        final luma = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
        out[outIndex++] = luma;
      }

      // Mean-center + standardize to avoid high cosine similarity
      // for unrelated positive-only vectors.
      // Trừ mean + chuẩn hóa std để tránh cosine cao do vector toàn dương.
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
      // Return null on any decode/IO error.
      // Trả về null nếu lỗi decode/IO.
      return null;
    }
  }
}
