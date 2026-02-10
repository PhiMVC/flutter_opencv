import 'package:flutter/material.dart';

import '../models/metrics.dart';

class MetricsPanel extends StatelessWidget {
  const MetricsPanel({super.key, required this.metrics});

  final Metrics metrics;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Builder(
        builder: (context) {
          final items = <Widget>[
            MetricRow(
              label: 'Nét',
              value: '${metrics.sharpness.toStringAsFixed(0)}%',
              isValid: metrics.isSharpnessOk,
            ),
            MetricRow(
              label: 'Ngang',
              value: '${metrics.angle.toStringAsFixed(1)} deg',
              isValid: metrics.isAngleOk,
            ),
            MetricRow(
              label: 'Dọc',
              value: '${metrics.tiltVertical.toStringAsFixed(1)} deg',
              isValid: metrics.isTiltVerticalOk,
            ),
            MetricRow(
              label: 'Thẳng',
              value: '${metrics.frontal.toStringAsFixed(0)}%',
              isValid: metrics.isFrontalOk,
            ),
            MetricRow(
              label: 'Sáng',
              value: '${metrics.brightness.toStringAsFixed(0)}%',
              isValid: metrics.isBrightnessOk,
            ),
            MetricRow(
              label: 'Rung',
              value: metrics.shake.toStringAsFixed(2),
              isValid: metrics.isShakeOk,
            ),
            MetricRow(
              label: 'Cách',
              value: '${metrics.distance.toStringAsFixed(2)} m',
              isValid: metrics.isDistanceOk,
            ),
          ];

          final firstRow = items.take(4).toList();
          final secondRow = items.skip(4).toList();
          while (secondRow.length < 4) {
            secondRow.add(const _MetricPlaceholder());
          }

          Widget cell(Widget child) {
            return Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 3),
                child: child,
              ),
            );
          }

          return Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(children: firstRow.map(cell).toList()),
              const SizedBox(height: 8),
              Row(children: secondRow.map(cell).toList()),
            ],
          );
        },
      ),
    );
  }
}

class MetricRow extends StatelessWidget {
  const MetricRow({
    super.key,
    required this.label,
    required this.value,
    required this.isValid,
  });

  final String label;
  final String value;
  final bool isValid;

  @override
  Widget build(BuildContext context) {
    final statusColor = isValid ? Colors.greenAccent : Colors.redAccent;
    return SizedBox(
      height: _metricItemHeight,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: statusColor.withValues(alpha: 0.8)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.8),
                fontSize: 11,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

const double _metricItemHeight = 60;

class _MetricPlaceholder extends StatelessWidget {
  const _MetricPlaceholder();

  @override
  Widget build(BuildContext context) {
    return const SizedBox(height: _metricItemHeight);
  }
}
