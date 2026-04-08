/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Hash ROLLUP (hierarchical grouping sets) via libcudf {@code groupby::groupby::rollup}.
 *
 * <p>Column order in the result table: {@code [key columns..., spark_grouping_id (INT64),
 * aggregate outputs...]}. Aggregation packing matches {@link Table.GroupByOperation#aggregate}:
 * one native instance per vararg, in order; the JNI layer merges consecutive ops on the same value
 * column into one {@code aggregation_request}.
 *
 * <p>The Spark plugin typically calls {@code com.nvidia.spark.rapids.jni.Rollup}, which delegates
 * here so native handle lifetime stays inside this package.
 *
 * <p>{@code rolledUpKeyIndicesAmongKeys} are zero-based positions within {@code keyColumnIndices},
 * not absolute table column indices. {@code keySorted}, {@code keysDescending}, and
 * {@code keysNullSmallest} are reserved: the current libcudf implementation uses the hash path only
 * ({@code sorted::NO}).
 */
public final class GroupByRollup {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private GroupByRollup() {}

  /**
   * @param ignoreNullKeys maps to libcudf {@code null_policy} ({@code true} &rarr; exclude rows with
   *                       null in active keys, {@code false} &rarr; include)
   */
  public static Table rollup(
      Table table,
      int[] keyColumnIndices,
      int[] rolledUpKeyIndicesAmongKeys,
      boolean ignoreNullKeys,
      boolean keySorted,
      boolean[] keysDescending,
      boolean[] keysNullSmallest,
      GroupByAggregationOnColumn... aggregates) {
    if (keyColumnIndices == null || keyColumnIndices.length == 0) {
      throw new IllegalArgumentException("keyColumnIndices must be non-null and non-empty");
    }
    if (aggregates == null || aggregates.length == 0) {
      throw new IllegalArgumentException("aggregates must be non-null and non-empty");
    }
    int[] rolled =
        rolledUpKeyIndicesAmongKeys == null ? new int[0] : rolledUpKeyIndicesAmongKeys;
    for (int r : rolled) {
      if (r < 0 || r >= keyColumnIndices.length) {
        throw new IllegalArgumentException(
            "rolledUpKeyIndicesAmongKeys must reference key positions in [0, "
                + keyColumnIndices.length
                + "): invalid index "
                + r);
      }
    }
    boolean[] kd = keysDescending == null ? new boolean[0] : keysDescending;
    boolean[] knf = keysNullSmallest == null ? new boolean[0] : keysNullSmallest;
    final int gidExtra = 1;

    int keysLength = keyColumnIndices.length;
    int[] aggColumnIndexes = new int[aggregates.length];
    long[] aggOperationInstances = new long[aggregates.length];
    try {
      for (int i = 0; i < aggregates.length; i++) {
        GroupByAggregationOnColumn agg = aggregates[i];
        aggColumnIndexes[i] = agg.getColumnIndex();
        aggOperationInstances[i] = agg.getWrapped().getWrapped().createNativeInstance();
      }

      try (Table out =
          new Table(
              nativeRollup(
                  table.getNativeView(),
                  keyColumnIndices,
                  rolled,
                  aggColumnIndexes,
                  aggOperationInstances,
                  ignoreNullKeys,
                  keySorted,
                  kd,
                  knf))) {
        ColumnVector[] finalCols = new ColumnVector[keysLength + gidExtra + aggregates.length];
        for (int k = 0; k < keysLength; k++) {
          finalCols[k] = out.getColumn(k);
        }
        finalCols[keysLength] = out.getColumn(keysLength);
        int aggBase = keysLength + gidExtra;
        for (int i = 0; i < aggregates.length; i++) {
          finalCols[aggBase + i] = out.getColumn(aggBase + i);
        }
        return new Table(finalCols);
      }
    } finally {
      Aggregation.close(aggOperationInstances);
    }
  }

  private static native long[] nativeRollup(
      long tableView,
      int[] keyColumnIndices,
      int[] rolledUpKeyIndicesAmongKeys,
      int[] aggColumnIndexes,
      long[] aggOperationInstances,
      boolean ignoreNullKeys,
      boolean keySorted,
      boolean[] keysDescending,
      boolean[] keysNullSmallest) throws CudfException;
}
