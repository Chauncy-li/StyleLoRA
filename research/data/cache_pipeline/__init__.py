"""Raw DB split and cache-generation entrypoints.

Run order:
  1. python -m research.data.cache_pipeline.build_raw_db_split
  2. python -m research.data.cache_pipeline.build_cache_train_val_split
  3. python -m research.data.cache_pipeline.process_cache_split
"""
